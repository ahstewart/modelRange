import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image/image.dart' as img;
import '../../core/data_models/inference_result_model.dart';
import '../../core/services/inferenceService.dart';
import '../../core/services/stats_service.dart';
import '../../core/providers/stats_providers.dart';

class TextToImageWidget extends ConsumerStatefulWidget {
  final String modelName;
  final String pipelinePath;
  final bool isLocalFile;
  final String? localDir;
  final String? modelVersionId;
  final String? modelDisplayName;

  const TextToImageWidget({
    super.key,
    required this.modelName,
    required this.pipelinePath,
    this.isLocalFile = false,
    this.localDir,
    this.modelVersionId,
    this.modelDisplayName,
  });

  @override
  ConsumerState<TextToImageWidget> createState() => _TextToImageWidgetState();
}

class _TextToImageWidgetState extends ConsumerState<TextToImageWidget> {
  late final InferenceService _inferenceObject;
  late final Future<void> _initFuture;

  final TextEditingController _promptController = TextEditingController();
  bool _isLoading = false;
  Uint8List? _imagePngBytes;
  Timer? _elapsedTimer;
  int _elapsedSeconds = 0;

  @override
  void initState() {
    super.initState();
    _inferenceObject = InferenceService(
      modelPath: widget.modelName,
      pipelinePath: widget.pipelinePath,
      isLocalFile: widget.isLocalFile,
      localDir: widget.localDir,
      modelVersionId: widget.modelVersionId,
      modelDisplayName: widget.modelDisplayName,
      statsService: widget.modelVersionId != null ? StatsService() : null,
    );
    _initFuture = _inferenceObject.initialize();
  }

  @override
  void dispose() {
    _elapsedTimer?.cancel();
    _inferenceObject.dispose();
    _promptController.dispose();
    super.dispose();
  }

  Future<void> _runGeneration() async {
    final text = _promptController.text.trim();
    if (text.isEmpty || _isLoading) return;
    if (!_inferenceObject.isReady) return;

    setState(() {
      _isLoading = true;
      _imagePngBytes = null;
      _elapsedSeconds = 0;
    });
    _elapsedTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      setState(() => _elapsedSeconds++);
    });

    await Future.delayed(const Duration(milliseconds: 200));

    try {
      final inputMap = <String, dynamic>{};
      for (final input in _inferenceObject.modelPipeline!.inputs) {
        inputMap[input.name] = text;
      }

      final result = await _inferenceObject.performInference(inputMap);
      final firstResult = result.values.firstOrNull;

      if (firstResult is GeneratedImageResult) {
        final pngBytes = img.encodePng(firstResult.image);
        setState(() => _imagePngBytes = Uint8List.fromList(pngBytes));
      } else if (firstResult != null) {
        if (kDebugMode) {
          debugPrint(
            '[TextToImage] Unexpected result type: ${firstResult.runtimeType}',
          );
        }
      }
    } catch (e) {
      if (kDebugMode) debugPrint('[TextToImage] Inference error: $e');
    } finally {
      _elapsedTimer?.cancel();
      _elapsedTimer = null;
      if (widget.modelVersionId != null) {
        ref
            .read(telemetryServiceProvider)
            .syncIfEligible(
              optedIn: ref.read(telemetryOptInProvider),
              authToken: null,
            )
            .ignore();
      }
      setState(() => _isLoading = false);
    }
  }

  String _formatElapsed(int s) {
    final m = s ~/ 60;
    final sec = s % 60;
    return '$m:${sec.toString().padLeft(2, '0')}';
  }

  Widget _buildLoadingCard(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      key: const ValueKey('loading-card'),
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 16),
      decoration: BoxDecoration(
        color: cs.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: cs.outline.withValues(alpha: 0.4)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const CircularProgressIndicator(),
          const SizedBox(height: 16),
          Text(
            'Generating image\u2026',
            style: Theme.of(
              context,
            ).textTheme.bodyLarge?.copyWith(fontWeight: FontWeight.w500),
          ),
          const SizedBox(height: 4),
          Text(
            _formatElapsed(_elapsedSeconds),
            style: Theme.of(
              context,
            ).textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<void>(
      future: _initFuture,
      builder: (context, snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return const Scaffold(
            body: Center(child: CircularProgressIndicator()),
          );
        }
        if (snapshot.hasError) {
          return const Scaffold(
            body: Center(
              child: Text(
                'Model not loaded. Check logs.',
                style: TextStyle(color: Colors.orange, fontSize: 16),
              ),
            ),
          );
        }
        return _buildMainContent(context);
      },
    );
  }

  Widget _buildMainContent(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: Text(
          _inferenceObject.modelPipeline?.metadata[0].model_name ??
              widget.modelDisplayName ??
              'Text to Image',
        ),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: 8),
              TextField(
                controller: _promptController,
                enabled: !_isLoading,
                maxLines: 4,
                minLines: 2,
                decoration: InputDecoration(
                  hintText: 'Describe the image you want to generate\u2026',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  contentPadding: const EdgeInsets.all(12),
                ),
              ),
              const SizedBox(height: 16),
              AnimatedSwitcher(
                duration: const Duration(milliseconds: 300),
                child:
                    _isLoading
                        ? _buildLoadingCard(context)
                        : ElevatedButton.icon(
                          key: const ValueKey('generate-btn'),
                          onPressed: _runGeneration,
                          icon: const Icon(Icons.image_outlined),
                          label: const Text('Generate Image'),
                        ),
              ),
              const SizedBox(height: 24),
              if (_imagePngBytes != null) ...[
                Text('Output', style: theme.textTheme.titleSmall),
                const SizedBox(height: 8),
                Card(
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                    side: BorderSide(color: cs.outline),
                  ),
                  clipBehavior: Clip.antiAlias,
                  child: Image.memory(_imagePngBytes!, fit: BoxFit.contain),
                ),
              ],
              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }
}
