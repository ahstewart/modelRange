import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import '../../core/data_models/inference_result_model.dart';
import '../../core/services/inferenceService.dart';
import '../../core/services/stats_service.dart';
import '../../core/providers/stats_providers.dart';
import '../../core/widgets/image_inference_shell.dart';

class SemanticSegmentationWidget extends ConsumerStatefulWidget {
  final String modelName;
  final String pipelinePath;
  final bool isLocalFile;
  final String? localDir;
  final String? modelVersionId;
  final String? modelDisplayName;

  const SemanticSegmentationWidget({
    super.key,
    required this.modelName,
    required this.pipelinePath,
    this.isLocalFile = false,
    this.localDir,
    this.modelVersionId,
    this.modelDisplayName,
  });

  @override
  ConsumerState<SemanticSegmentationWidget> createState() =>
      _SemanticSegmentationWidgetState();
}

class _SemanticSegmentationWidgetState
    extends ConsumerState<SemanticSegmentationWidget> {
  late final InferenceService _svc;
  late final Future<void> _initFuture;

  File? _pickedImageFile;
  img.Image? _pickedImage;
  SegmentationResult? _result;
  Uint8List? _maskPngBytes;
  bool _isLoading = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _svc = InferenceService(
      modelPath: widget.modelName,
      pipelinePath: widget.pipelinePath,
      isLocalFile: widget.isLocalFile,
      localDir: widget.localDir,
      modelVersionId: widget.modelVersionId,
      modelDisplayName: widget.modelDisplayName,
      statsService: widget.modelVersionId != null ? StatsService() : null,
    );
    _initFuture = _svc.initialize();
  }

  @override
  void dispose() {
    _svc.dispose();
    super.dispose();
  }

  Future<void> _pickImage(ImageSource source) async {
    if (_isLoading) return;

    try {
      final XFile? picked = await ImagePicker().pickImage(source: source);
      if (picked == null) return;

      final bytes = await picked.readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) {
        setState(() => _error = 'Could not decode image.');
        return;
      }
      setState(() {
        _pickedImageFile = File(picked.path);
        _pickedImage = decoded;
        _result = null;
        _maskPngBytes = null;
        _error = null;
      });
    } catch (e) {
      if (kDebugMode) debugPrint('[Segmentation] Pick error: $e');
      return;
    }

    if (!_svc.isReady) return;

    setState(() => _isLoading = true);
    try {
      final inputName = _svc.modelPipeline!.inputs.first.name;
      final results = await _svc.performInference({inputName: _pickedImage});

      final first = results.values.firstOrNull;
      if (first is SegmentationResult) {
        final pngBytes = img.encodePng(first.mask);
        setState(() {
          _result = first;
          _maskPngBytes = pngBytes;
        });
      } else {
        setState(
          () => _error = 'Unexpected result type: ${first?.runtimeType}',
        );
      }
    } catch (e) {
      if (kDebugMode) debugPrint('[Segmentation] Inference error: $e');
      setState(() => _error = e.toString());
    } finally {
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

  Widget _buildImageArea() {
    if (_pickedImageFile == null) {
      return Container(
        height: 250,
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey),
          borderRadius: BorderRadius.circular(12),
          color: Colors.grey[200],
        ),
        child: const Center(
          child: Text(
            'No image selected.',
            style: TextStyle(color: Colors.grey),
          ),
        ),
      );
    }
    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: Stack(
        children: [
          Image.file(
            _pickedImageFile!,
            width: double.infinity,
            fit: BoxFit.contain,
          ),
          if (_maskPngBytes != null)
            Opacity(
              opacity: 0.6,
              child: Image.memory(
                _maskPngBytes!,
                width: double.infinity,
                fit: BoxFit.contain,
              ),
            ),
        ],
      ),
    );
  }

  Widget? _buildResults() {
    if (_error != null) {
      return Text('Error: $_error', style: const TextStyle(color: Colors.red));
    }
    if (_result != null) {
      return _buildLegend(_result!);
    }
    return null;
  }

  /// Returns up to [maxClasses] most-frequent class indices from the mask.
  List<(int, int)> _getTopClasses(
    SegmentationResult result, {
    int maxClasses = 10,
  }) {
    final palette = result.palette;
    final Map<int, int> freq = {};
    for (final pixel in result.mask) {
      final r = pixel.r.toInt();
      final g = pixel.g.toInt();
      final b = pixel.b.toInt();
      for (int i = 0; i < palette.length; i++) {
        if (palette[i][0] == r && palette[i][1] == g && palette[i][2] == b) {
          freq[i] = (freq[i] ?? 0) + 1;
          break;
        }
      }
    }
    final sorted =
        freq.entries.toList()..sort((a, b) => b.value.compareTo(a.value));
    return sorted.take(maxClasses).map((e) => (e.key, e.value)).toList();
  }

  Widget _buildLegend(SegmentationResult result) {
    final topClasses = _getTopClasses(result);
    final labels = result.labels;
    final palette = result.palette;

    return Wrap(
      spacing: 12,
      runSpacing: 6,
      children:
          topClasses.map((entry) {
            final classIdx = entry.$1;
            final rgb = palette[classIdx % palette.length];
            final color = Color.fromARGB(255, rgb[0], rgb[1], rgb[2]);
            final label =
                (labels != null && classIdx < labels.length)
                    ? labels[classIdx]
                    : 'Class $classIdx';
            return Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 16,
                  height: 16,
                  decoration: BoxDecoration(
                    color: color,
                    borderRadius: BorderRadius.circular(3),
                  ),
                ),
                const SizedBox(width: 4),
                Text(label, style: const TextStyle(fontSize: 12)),
              ],
            );
          }).toList(),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ImageInferenceShell(
      title: 'Semantic Segmentation',
      initFuture: _initFuture,
      isRunning: _isLoading,
      onImagePicked: _pickImage,
      imageArea: _buildImageArea(),
      results: _buildResults(),
    );
  }
}
