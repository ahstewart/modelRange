import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
//import 'package:freezed_annotation/freezed_annotation.dart';
//import 'package:flutter/foundation.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';
import 'features/image_classification/image_classification.dart';
import 'core/services/inferenceService.dart';
import 'core/data_models/pipeline.dart';
import 'core/data_models/api_models.dart';
import 'core/providers/model_providers.dart';
import 'package:yaml/yaml.dart';
import 'dart:convert';
import 'features/object_detection/object_detection.dart';
import 'features/text_generation/text_generation.dart';
import 'features/semantic_segmentation/semantic_segmentation.dart';
import 'features/automatic_speech_recognition/asr_widget.dart';
import 'features/image_to_text/image_to_text_widget.dart';
import 'features/text_to_image/text_to_image_widget.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'core/services/stats_service.dart';
import 'core/providers/stats_providers.dart';
import 'core/data_models/inference_stat.dart';
import 'core/providers/auth_providers.dart';
import 'core/services/api_service.dart';
import 'core/config/app_config.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

// Returns free bytes available on the filesystem containing [dirPath], or null on failure.
Future<int?> _getFreeDiskBytes(String dirPath) async {
  try {
    final result = await Process.run('df', ['-B1', dirPath]);
    if (result.exitCode != 0) return null;
    final lines = (result.stdout as String).trim().split('\n');
    if (lines.length < 2) return null;
    // Typical df -B1 output: Filesystem 1B-blocks Used Available Use% Mounted
    final parts = lines.last.trim().split(RegExp(r'\s+'));
    if (parts.length >= 4) return int.tryParse(parts[3]);
  } catch (_) {}
  return null;
}

String _formatBytes(int bytes) {
  if (bytes >= 1024 * 1024 * 1024) {
    return '${(bytes / 1024 / 1024 / 1024).toStringAsFixed(1)} GB';
  }
  if (bytes >= 1024 * 1024) {
    return '${(bytes / 1024 / 1024).toStringAsFixed(1)} MB';
  }
  if (bytes >= 1024) return '${(bytes / 1024).toStringAsFixed(0)} KB';
  return '$bytes B';
}

Future<void> main() async {
  runZonedGuarded(
    () async {
      WidgetsFlutterBinding.ensureInitialized();

      await Supabase.initialize(
        url: AppConfig.supabaseUrl,
        anonKey: AppConfig.supabaseAnonKey,
      );

      await DeviceInfoHelper.init();
      final prefs = await SharedPreferences.getInstance();

      FlutterError.onError = (FlutterErrorDetails details) {
        FlutterError.presentError(details); // Still show red screen in debug
        debugPrint('Caught by global error handler: ${details.exception}');
      };

      runApp(
        ProviderScope(
          overrides: [sharedPreferencesProvider.overrideWithValue(prefs)],
          child: const MyApp(),
        ),
      );
    },
    (error, stackTrace) {
      debugPrint('Uncaught async error: $error');
    },
  );
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Jacana',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: const ColorScheme.light(
          // Primary: Chestnut Brown
          primary: Color(0xFF8C3A21),
          onPrimary: Colors.white,
          primaryContainer: Color(0xFFFAE5D8),
          onPrimaryContainer: Color(0xFF3F190D),
          // Secondary: Golden Amber
          secondary: Color(0xFFE5A91A),
          onSecondary: Color(0xFF1A1A1A),
          secondaryContainer: Color(0xFFFEF0C3),
          onSecondaryContainer: Color(0xFF4F3A07),
          // Tertiary: Wing Shield Teal — FAB, highlighted chips, tonal buttons
          tertiary: Color(0xFF0D9488),
          onTertiary: Colors.white,
          tertiaryContainer: Color(0xFFCCFBF1),
          onTertiaryContainer: Color(0xFF134E4A),
          // Surfaces
          surface: Colors.white,
          onSurface: Color(0xFF1A1A1A),
          surfaceContainerHighest: Color(0xFFF5F5F5),
          // Accent: Slate Blue for secondary text / inactive
          onSurfaceVariant: Color(0xFF526870),
          // Borders
          outline: Color(0xFFC5D3D8),
          error: Color(0xFFDC2626),
          onError: Colors.white,
        ),
        scaffoldBackgroundColor: const Color(0xFFF5F5F5),
        cardTheme: CardThemeData(
          elevation: 0,
          color: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
            side: const BorderSide(color: Color(0xFFC5D3D8)),
          ),
          margin: EdgeInsets.zero,
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF8C3A21),
          foregroundColor: Colors.white,
          elevation: 0,
          scrolledUnderElevation: 1,
          shadowColor: Color(0x1A000000),
          titleTextStyle: TextStyle(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.w700,
          ),
        ),
        navigationBarTheme: NavigationBarThemeData(
          backgroundColor: Colors.white,
          indicatorColor: const Color(0xFFFAE5D8),
          elevation: 0,
          shadowColor: Colors.transparent,
          surfaceTintColor: Colors.transparent,
          iconTheme: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.selected)) {
              return const IconThemeData(color: Color(0xFF8C3A21));
            }
            return const IconThemeData(color: Color(0xFF829CA5));
          }),
          labelTextStyle: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.selected)) {
              return const TextStyle(
                color: Color(0xFF8C3A21),
                fontSize: 12,
                fontWeight: FontWeight.w600,
              );
            }
            return const TextStyle(color: Color(0xFF829CA5), fontSize: 12);
          }),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF8C3A21),
            foregroundColor: Colors.white,
            elevation: 0,
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 11),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
            textStyle: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(
            foregroundColor: const Color(0xFF8C3A21),
            side: const BorderSide(color: Color(0xFF8C3A21)),
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 11),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
            textStyle: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
        textButtonTheme: TextButtonThemeData(
          style: TextButton.styleFrom(foregroundColor: const Color(0xFF8C3A21)),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(10),
            borderSide: const BorderSide(color: Color(0xFFC5D3D8)),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(10),
            borderSide: const BorderSide(color: Color(0xFFC5D3D8)),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(10),
            borderSide: const BorderSide(color: Color(0xFF8C3A21), width: 2),
          ),
          contentPadding: const EdgeInsets.symmetric(
            horizontal: 12,
            vertical: 10,
          ),
        ),
        chipTheme: ChipThemeData(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
          labelPadding: const EdgeInsets.symmetric(horizontal: 2),
          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 0),
        ),
        dividerTheme: const DividerThemeData(
          color: Color(0xFFC5D3D8),
          thickness: 1,
        ),
        switchTheme: SwitchThemeData(
          thumbColor: WidgetStateProperty.resolveWith(
            (states) =>
                states.contains(WidgetState.selected)
                    ? const Color(0xFF8C3A21)
                    : null,
          ),
          trackColor: WidgetStateProperty.resolveWith(
            (states) =>
                states.contains(WidgetState.selected)
                    ? const Color(0xFFFCE08A)
                    : null,
          ),
        ),
      ),
      home: const ModelList(title: 'Jacana'),
    );
  }
}

class _SelectedModelNotifier extends Notifier<MLModel?> {
  @override
  MLModel? build() => null;
}

final selectedModelProvider =
    NotifierProvider<_SelectedModelNotifier, MLModel?>(
      _SelectedModelNotifier.new,
    );

// Provider to track which model's modal is open
class _SelectedModelModalNotifier extends Notifier<MLModel?> {
  @override
  MLModel? build() => null;
}

final selectedModelModalProvider =
    NotifierProvider<_SelectedModelModalNotifier, MLModel?>(
      _SelectedModelModalNotifier.new,
    );

// Model for storing downloaded model metadata
class DownloadedModel {
  final String modelId;
  final String modelName;
  final String versionName;
  final String versionId;

  /// Pipeline model_task (e.g. "image_classification") — used for inference routing.
  final String category;

  /// MLModel category (e.g. "utility", "diagnostic") — used for filtering.
  final String modelCategory;

  final DateTime downloadedAt;
  final String localPath;

  DownloadedModel({
    required this.modelId,
    required this.modelName,
    required this.versionName,
    required this.versionId,
    required this.category,
    required this.modelCategory,
    required this.downloadedAt,
    required this.localPath,
  });

  Map<String, dynamic> toJson() => {
    'modelId': modelId,
    'modelName': modelName,
    'versionName': versionName,
    'versionId': versionId,
    'category': category,
    'modelCategory': modelCategory,
    'downloadedAt': downloadedAt.toIso8601String(),
    'localPath': localPath,
  };

  static DownloadedModel? tryFromJson(Map<String, dynamic> json) {
    try {
      return DownloadedModel(
        modelId: json['modelId'] as String? ?? '',
        modelName: json['modelName'] as String? ?? 'Unknown',
        versionName: json['versionName'] as String? ?? '',
        versionId: json['versionId'] as String? ?? '',
        category: json['category'] as String? ?? 'unknown',
        modelCategory: json['modelCategory'] as String? ?? 'other',
        downloadedAt:
            json['downloadedAt'] != null
                ? DateTime.tryParse(json['downloadedAt'] as String) ??
                    DateTime.now()
                : DateTime.now(),
        localPath: json['localPath'] as String? ?? '',
      );
    } catch (e) {
      debugPrint('[DownloadedModel] Skipping bad entry: $e');
      return null;
    }
  }
}

final sharedPreferencesProvider = Provider<SharedPreferences>(
  (_) =>
      throw UnimplementedError(
        'sharedPreferencesProvider must be overridden in main()',
      ),
);

class DownloadedModelsState {
  final List<DownloadedModel> models;
  final bool isLoaded;
  const DownloadedModelsState({required this.models, required this.isLoaded});
}

// Provider for managing downloaded models
final downloadedModelsProvider =
    NotifierProvider<DownloadedModelsNotifier, DownloadedModelsState>(
      DownloadedModelsNotifier.new,
    );

class DownloadedModelsNotifier extends Notifier<DownloadedModelsState> {
  static const _prefsKey = 'downloaded_models_v1';
  late SharedPreferences _prefs;
  bool _disposed = false;

  @override
  DownloadedModelsState build() {
    _disposed = false;
    _prefs = ref.watch(sharedPreferencesProvider);
    ref.onDispose(() => _disposed = true);
    Future.microtask(_initialize);
    return const DownloadedModelsState(models: [], isLoaded: false);
  }

  Future<void> _initialize() async {
    await _loadDownloadedModels();
    await _scanDiskForMissingModels();
    if (!_disposed)
      state = DownloadedModelsState(models: state.models, isLoaded: true);
  }

  Future<void> _loadDownloadedModels() async {
    try {
      final raw = _prefs.getString(_prefsKey);
      if (raw == null) return;
      final list = (jsonDecode(raw) as List).cast<Map<String, dynamic>>();
      if (!_disposed)
        state = DownloadedModelsState(
          models:
              list
                  .map(DownloadedModel.tryFromJson)
                  .whereType<DownloadedModel>()
                  .toList(),
          isLoaded: false,
        );
    } catch (e, st) {
      debugPrint('[DownloadedModels] Failed to load from prefs: $e\n$st');
    }
  }

  Future<void> _scanDiskForMissingModels() async {
    try {
      final appDir = await getApplicationDocumentsDirectory();
      final root = Directory('${appDir.path}/downloaded_models');
      if (!await root.exists()) return;

      final knownIds = state.models.map((m) => m.versionId).toSet();
      final recovered = <DownloadedModel>[];

      await for (final entity in root.list()) {
        if (entity is! Directory) continue;
        final versionId = entity.path.split(Platform.pathSeparator).last;
        if (knownIds.contains(versionId)) continue;

        final specFile = File('${entity.path}/pipeline_spec.json');
        if (!await specFile.exists()) continue;

        try {
          final spec =
              jsonDecode(await specFile.readAsString()) as Map<String, dynamic>;
          final metadata =
              (spec['metadata'] as List?)?.first as Map<String, dynamic>?;
          final task = metadata?['model_task'] as String? ?? 'unknown';
          final modelName = metadata?['model_name'] as String? ?? 'Unknown';

          recovered.add(
            DownloadedModel(
              modelId: '',
              modelName: modelName,
              versionName: '',
              versionId: versionId,
              category: task,
              modelCategory: 'other',
              downloadedAt: (await entity.stat()).modified,
              localPath: entity.path,
            ),
          );
        } catch (e) {
          debugPrint(
            '[DownloadedModels] Could not recover $versionId from disk: $e',
          );
        }
      }

      if (recovered.isNotEmpty && !_disposed) {
        state = DownloadedModelsState(
          models: [...state.models, ...recovered],
          isLoaded: false,
        );
        await _persist();
        debugPrint(
          '[DownloadedModels] Recovered ${recovered.length} model(s) from disk.',
        );
      }
    } catch (e, st) {
      debugPrint('[DownloadedModels] Disk scan failed: $e\n$st');
    }
  }

  Future<void> _persist() async {
    try {
      await _prefs.setString(
        _prefsKey,
        jsonEncode(state.models.map((m) => m.toJson()).toList()),
      );
    } catch (e, st) {
      debugPrint('[DownloadedModels] Failed to persist: $e\n$st');
    }
  }

  void addDownloadedModel(DownloadedModel model) {
    state = DownloadedModelsState(
      models: [...state.models, model],
      isLoaded: true,
    );
    _persist();
  }

  void removeDownloadedModel(String versionId) {
    state = DownloadedModelsState(
      models: state.models.where((m) => m.versionId != versionId).toList(),
      isLoaded: true,
    );
    _persist();
  }
}

// Model for tracking an in-progress model download
class InProgressDownload {
  final String versionId;
  final String modelName;
  final String versionName;
  final int completedSteps;
  final int totalSteps;

  InProgressDownload({
    required this.versionId,
    required this.modelName,
    required this.versionName,
    required this.completedSteps,
    required this.totalSteps,
  });

  double get progress => totalSteps > 0 ? completedSteps / totalSteps : 0.0;

  InProgressDownload copyWith({int? completedSteps}) => InProgressDownload(
    versionId: versionId,
    modelName: modelName,
    versionName: versionName,
    completedSteps: completedSteps ?? this.completedSteps,
    totalSteps: totalSteps,
  );
}

final inProgressDownloadsProvider = NotifierProvider<
  InProgressDownloadsNotifier,
  Map<String, InProgressDownload>
>(InProgressDownloadsNotifier.new);

class InProgressDownloadsNotifier
    extends Notifier<Map<String, InProgressDownload>> {
  @override
  Map<String, InProgressDownload> build() => {};

  void startDownload(InProgressDownload download) {
    state = {...state, download.versionId: download};
  }

  void updateProgress(String versionId, int completedSteps) {
    final current = state[versionId];
    if (current == null) return;
    state = {
      ...state,
      versionId: current.copyWith(completedSteps: completedSteps),
    };
  }

  void finishDownload(String versionId) {
    final updated = Map<String, InProgressDownload>.from(state)
      ..remove(versionId);
    state = updated;
  }
}

// Widget containing the model tile list
IconData _taskIcon(String task) {
  if (task.contains('classif')) return Icons.image_search;
  if (task.contains('detect')) return Icons.center_focus_strong;
  if (task.contains('segment')) return Icons.blur_circular;
  if (task.contains('text') || task.contains('generat')) {
    return Icons.text_fields;
  }
  if (task.contains('speech') ||
      task.contains('audio') ||
      task.contains('asr')) {
    return Icons.mic;
  }
  return Icons.memory;
}

/// Returns (containerColor, onContainerColor) for a task type.
/// Teal for text generation, blue for speech, amber for image, primary for others.
(Color, Color) _taskColors(String task, ColorScheme cs) {
  if (task.contains('text') || task.contains('generat')) {
    return (const Color(0xFFCCFBF1), const Color(0xFF134E4A)); // teal container
  }
  if (task.contains('speech') ||
      task.contains('audio') ||
      task.contains('asr')) {
    return (const Color(0xFFDBEAFE), const Color(0xFF1E3A8A)); // blue container
  }
  if (task == 'image_to_text' || task == 'image-to-text') {
    return (
      const Color(0xFFEDE9FE),
      const Color(0xFF3730A3),
    ); // violet container
  }
  if (task == 'text_to_image' || task == 'text-to-image') {
    return (const Color(0xFFFCE7F3), const Color(0xFF831843)); // pink container
  }
  return (cs.primaryContainer, cs.onPrimaryContainer);
}

/// Returns the best pipeline status for a model, using the pre-computed
/// backend field best_version_status.
String _bestStatus(MLModel model) => model.best_version_status ?? 'missing';

String _friendlyTask(String category) => category
    .split('_')
    .map((w) => w.isEmpty ? '' : '${w[0].toUpperCase()}${w.substring(1)}')
    .join(' ');

String _formatCount(int n) {
  if (n >= 1000000) return '${(n / 1000000).toStringAsFixed(1)}M';
  if (n >= 1000) return '${(n / 1000).toStringAsFixed(1)}K';
  return '$n';
}

class Models extends ConsumerStatefulWidget {
  const Models({super.key});

  @override
  ConsumerState<Models> createState() => _ModelsState();
}

class _ModelsState extends ConsumerState<Models> {
  final _searchController = TextEditingController();
  String _searchQuery = '';
  String? _selectedTask;
  String? _selectedStatus;
  String _sortOrder = 'downloads'; // 'downloads' | 'rating' | 'newest'

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final modelsAsync = ref.watch(supportedModelsProvider);

    return modelsAsync.when(
      data: (modelList) {
        if (modelList.isEmpty) {
          return const Center(child: Text('No supported models available'));
        }

        // Collect unique tasks, sorted alphabetically
        final tasks =
            modelList
                .map((m) => m.task)
                .whereType<String>()
                .where((t) => t.isNotEmpty)
                .toSet()
                .toList()
              ..sort();

        // Apply search, task filters, and hide unsupported models
        var filtered =
            modelList.where((m) {
              if (_bestStatus(m) == 'missing') return false;
              final q = _searchQuery.toLowerCase();
              final matchesSearch =
                  q.isEmpty ||
                  m.name.toLowerCase().contains(q) ||
                  m.description.toLowerCase().contains(q) ||
                  (m.task?.toLowerCase().contains(q) ?? false);
              final matchesTask =
                  _selectedTask == null || m.task == _selectedTask;
              final matchesStatus =
                  _selectedStatus == null || _bestStatus(m) == _selectedStatus;
              return matchesSearch && matchesTask && matchesStatus;
            }).toList();

        // Apply sort
        switch (_sortOrder) {
          case 'rating':
            filtered.sort(
              (a, b) => b.rating_weighted_avg.compareTo(a.rating_weighted_avg),
            );
          case 'newest':
            filtered.sort((a, b) => b.created_at.compareTo(a.created_at));
          default:
            filtered.sort(
              (a, b) =>
                  b.total_download_count.compareTo(a.total_download_count),
            );
        }

        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Search bar
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
              child: TextField(
                controller: _searchController,
                decoration: InputDecoration(
                  hintText: 'Search models\u2026',
                  prefixIcon: const Icon(Icons.search),
                  suffixIcon:
                      _searchQuery.isNotEmpty
                          ? IconButton(
                            icon: const Icon(Icons.clear),
                            onPressed: () {
                              _searchController.clear();
                              setState(() => _searchQuery = '');
                            },
                          )
                          : null,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  isDense: true,
                  contentPadding: const EdgeInsets.symmetric(vertical: 10),
                ),
                onChanged: (v) => setState(() => _searchQuery = v),
              ),
            ),
            // Sort button
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 0, 8, 0),
              child: Row(
                children: [
                  const Spacer(),
                  PopupMenuButton<String>(
                    icon: const Icon(Icons.sort),
                    tooltip: 'Sort by',
                    initialValue: _sortOrder,
                    onSelected: (v) => setState(() => _sortOrder = v),
                    itemBuilder:
                        (_) => const [
                          PopupMenuItem(
                            value: 'downloads',
                            child: Text('Most Downloaded'),
                          ),
                          PopupMenuItem(
                            value: 'rating',
                            child: Text('Top Rated'),
                          ),
                          PopupMenuItem(value: 'newest', child: Text('Newest')),
                        ],
                  ),
                ],
              ),
            ),
            // Task filter chips (only shown when tasks are available)
            if (tasks.isNotEmpty)
              Padding(
                padding: const EdgeInsets.fromLTRB(12, 0, 12, 4),
                child: SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Row(
                    children: [
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 4),
                        child: FilterChip(
                          label: const Text('All Tasks'),
                          selected: _selectedTask == null,
                          onSelected:
                              (_) => setState(() => _selectedTask = null),
                        ),
                      ),
                      for (final task in tasks)
                        Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 4),
                          child: FilterChip(
                            label: Text(_friendlyTask(task)),
                            selected: _selectedTask == task,
                            onSelected:
                                (selected) => setState(
                                  () => _selectedTask = selected ? task : null,
                                ),
                          ),
                        ),
                    ],
                  ),
                ),
              ),
            // Status filter chips
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 0, 12, 4),
              child: SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: Row(
                  children: [
                    for (final entry in const [
                      (null, 'All'),
                      ('verified', 'Verified'),
                      ('pending', 'Pending'),
                    ])
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 4),
                        child: FilterChip(
                          label: Text(entry.$2),
                          selected: _selectedStatus == entry.$1,
                          onSelected:
                              (_) => setState(() => _selectedStatus = entry.$1),
                        ),
                      ),
                  ],
                ),
              ),
            ),
            // Model list or empty state
            if (filtered.isEmpty)
              const Expanded(
                child: Center(child: Text('No models match your search.')),
              )
            else
              Expanded(
                child: ListView.separated(
                  padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
                  itemCount: filtered.length,
                  separatorBuilder: (_, __) => const SizedBox(height: 8),
                  itemBuilder: (context, index) {
                    final model = filtered[index];
                    return _ModelCard(
                      model: model,
                      onTap: () => _showModelDetailsModal(context, model),
                    );
                  },
                ),
              ),
          ],
        );
      },
      loading: () => const Center(child: CircularProgressIndicator()),
      error:
          (err, stack) => _ApiErrorWidget(
            err: err,
            onRetry: () => ref.invalidate(supportedModelsProvider),
          ),
    );
  }

  void _showModelDetailsModal(BuildContext context, MLModel model) {
    showDialog(
      context: context,
      builder: (context) => ModelDetailsModal(model: model),
    );
  }
}

// ── API error widget ────────────────────────────────────────────────────────

class _ApiErrorWidget extends StatelessWidget {
  const _ApiErrorWidget({required this.err, required this.onRetry});

  final Object err;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    final bool isOffline =
        err is ApiConnectionException ||
        err is ApiTimeoutException ||
        err.toString().toLowerCase().contains('socket') ||
        err.toString().toLowerCase().contains('connection refused');

    final String message = switch (err) {
      ApiConnectionException() =>
        "Couldn't reach the Jacana server.\nMake sure the backend is running and try again.",
      ApiTimeoutException() =>
        'The server took too long to respond.\nCheck your connection and try again.',
      _ when isOffline =>
        "Couldn't connect to the server.\nCheck your connection and try again.",
      _ => 'Something went wrong loading models.\nPlease try again.',
    };

    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              isOffline ? Icons.cloud_off_outlined : Icons.error_outline,
              size: 56,
              color: cs.outline,
            ),
            const SizedBox(height: 16),
            Text(
              message,
              textAlign: TextAlign.center,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: cs.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 24),
            FilledButton.tonal(onPressed: onRetry, child: const Text('Retry')),
          ],
        ),
      ),
    );
  }
}

class _ModelCard extends StatelessWidget {
  final MLModel model;
  final VoidCallback onTap;

  const _ModelCard({required this.model, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final taskLabel =
        model.task != null && model.task!.isNotEmpty
            ? _friendlyTask(model.task!)
            : _friendlyTask(model.category);
    final taskKey = (model.task ?? model.category).toLowerCase();
    final icon = _taskIcon(taskKey);
    final (containerColor, onContainerColor) = _taskColors(taskKey, cs);

    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          child: Row(
            children: [
              // Icon box — teal for text gen, blue for speech, primary for others
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: containerColor,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(icon, color: onContainerColor, size: 22),
              ),
              const SizedBox(width: 12),
              // Text block
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            model.name,
                            style: theme.textTheme.bodyMedium?.copyWith(
                              fontWeight: FontWeight.w600,
                              color: cs.onSurface,
                            ),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                        const SizedBox(width: 6),
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 6,
                            vertical: 2,
                          ),
                          decoration: BoxDecoration(
                            color: containerColor,
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: Text(
                            taskLabel,
                            style: TextStyle(
                              fontSize: 10,
                              fontWeight: FontWeight.w600,
                              color: onContainerColor,
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 2),
                    Text(
                      model.description,
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: cs.onSurfaceVariant,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 4),
                    Row(
                      children: [
                        // Downloads
                        Icon(
                          Icons.download_outlined,
                          size: 12,
                          color: cs.onSurfaceVariant,
                        ),
                        const SizedBox(width: 3),
                        Text(
                          _formatCount(model.total_download_count),
                          style: theme.textTheme.labelSmall?.copyWith(
                            color: cs.onSurfaceVariant,
                          ),
                        ),
                        // Rating
                        if (model.total_ratings > 0) ...[
                          const SizedBox(width: 8),
                          Icon(
                            Icons.star_rounded,
                            size: 12,
                            color: Colors.amber[700],
                          ),
                          const SizedBox(width: 3),
                          Text(
                            model.rating_weighted_avg.toStringAsFixed(1),
                            style: theme.textTheme.labelSmall?.copyWith(
                              color: cs.onSurfaceVariant,
                            ),
                          ),
                        ],
                        // Version count
                        if (model.version_count > 0) ...[
                          const SizedBox(width: 8),
                          Icon(
                            Icons.layers_outlined,
                            size: 12,
                            color: cs.onSurfaceVariant,
                          ),
                          const SizedBox(width: 3),
                          Text(
                            '${model.version_count}',
                            style: theme.textTheme.labelSmall?.copyWith(
                              color: cs.onSurfaceVariant,
                            ),
                          ),
                        ],
                        // Status dot
                        ...[
                          const SizedBox(width: 8),
                          Builder(
                            builder: (context) {
                              final status = _bestStatus(model);
                              final color =
                                  status == 'verified'
                                      ? Colors.green[600]!
                                      : status == 'pending'
                                      ? Colors.amber[700]!
                                      : cs.outlineVariant;
                              final label =
                                  status == 'verified'
                                      ? 'Verified'
                                      : status == 'pending'
                                      ? 'Pending'
                                      : 'Missing';
                              return Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Container(
                                    width: 6,
                                    height: 6,
                                    decoration: BoxDecoration(
                                      color: color,
                                      shape: BoxShape.circle,
                                    ),
                                  ),
                                  const SizedBox(width: 3),
                                  Text(
                                    label,
                                    style: theme.textTheme.labelSmall?.copyWith(
                                      color: color,
                                    ),
                                  ),
                                ],
                              );
                            },
                          ),
                        ],
                        // File size
                        if (model.file_size_bytes > 0) ...[
                          const SizedBox(width: 8),
                          Icon(
                            Icons.sd_storage_outlined,
                            size: 12,
                            color: cs.onSurfaceVariant,
                          ),
                          const SizedBox(width: 3),
                          Text(
                            _formatBytes(model.file_size_bytes),
                            style: theme.textTheme.labelSmall?.copyWith(
                              color: cs.onSurfaceVariant,
                            ),
                          ),
                        ],
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 4),
              Icon(Icons.chevron_right, size: 18, color: cs.onSurfaceVariant),
            ],
          ),
        ),
      ),
    );
  }
}

// Modal widget for displaying model details and versions
class ModelDetailsModal extends ConsumerStatefulWidget {
  final MLModel model;

  const ModelDetailsModal({required this.model, super.key});

  @override
  ConsumerState<ModelDetailsModal> createState() => _ModelDetailsModalState();
}

class _ModelDetailsModalState extends ConsumerState<ModelDetailsModal> {
  @override
  Widget build(BuildContext context) {
    final versionsAsync = ref.watch(modelVersionsProvider(widget.model.id));

    return Dialog(
      insetPadding: const EdgeInsets.all(16.0),
      child: versionsAsync.when(
        data: (versions) {
          final supportedVersions = versions.where((v) => v.isUsable).toList();

          if (supportedVersions.isEmpty) {
            return AlertDialog(
              title: const Text('Model Details'),
              content: const Text('No supported versions available'),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text('Close'),
                ),
              ],
            );
          }

          final theme = Theme.of(context);
          final cs = theme.colorScheme;

          return AlertDialog(
            contentPadding: EdgeInsets.zero,
            content: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // ── Header ──────────────────────────────────────────
                  Padding(
                    padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // Task badge
                        Wrap(
                          spacing: 6,
                          runSpacing: 4,
                          children: [
                            if (widget.model.task != null &&
                                widget.model.task!.isNotEmpty)
                              Chip(
                                label: Text(
                                  _friendlyTask(widget.model.task!),
                                  style: TextStyle(
                                    color: cs.onSecondaryContainer,
                                    fontSize: 12,
                                    fontWeight: FontWeight.w500,
                                  ),
                                ),
                                backgroundColor: cs.secondaryContainer,
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 4,
                                ),
                                materialTapTargetSize:
                                    MaterialTapTargetSize.shrinkWrap,
                              ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        // Model name
                        Text(
                          widget.model.name,
                          style: theme.textTheme.titleLarge?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 8),
                        // Description
                        Text(
                          widget.model.description,
                          style: theme.textTheme.bodyMedium?.copyWith(
                            color: cs.onSurfaceVariant,
                          ),
                        ),
                        const SizedBox(height: 12),
                        // Aggregate stats row
                        Row(
                          children: [
                            const Icon(Icons.download_outlined, size: 16),
                            const SizedBox(width: 4),
                            Text(
                              _formatCount(widget.model.total_download_count),
                              style: theme.textTheme.labelMedium,
                            ),
                            const SizedBox(width: 20),
                            const Icon(Icons.star_outline, size: 16),
                            const SizedBox(width: 4),
                            Text(
                              widget.model.total_ratings > 0
                                  ? '${widget.model.rating_weighted_avg.toStringAsFixed(1)} (${_formatCount(widget.model.total_ratings)})'
                                  : 'No ratings yet',
                              style: theme.textTheme.labelMedium,
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                      ],
                    ),
                  ),
                  const Divider(height: 1),
                  // ── Versions ────────────────────────────────────────
                  Padding(
                    padding: const EdgeInsets.fromLTRB(20, 16, 20, 4),
                    child: Text(
                      'Versions (${supportedVersions.length})',
                      style: theme.textTheme.titleSmall,
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(12, 0, 12, 8),
                    child: Column(
                      children:
                          supportedVersions
                              .map(
                                (version) => _VersionTile(
                                  version: version,
                                  modelName: widget.model.name,
                                  modelCategory: widget.model.category,
                                ),
                              )
                              .toList(),
                    ),
                  ),
                ],
              ),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Close'),
              ),
            ],
          );
        },
        loading:
            () => AlertDialog(
              content: const Center(child: CircularProgressIndicator()),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text('Close'),
                ),
              ],
            ),
        error:
            (err, stack) => AlertDialog(
              title: const Text('Error'),
              content: Text('Error loading versions: $err'),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text('Close'),
                ),
              ],
            ),
      ),
    );
  }
}

class _VersionTile extends ConsumerStatefulWidget {
  final ModelVersion version;
  final String modelName;
  final String modelCategory;

  const _VersionTile({
    required this.version,
    required this.modelName,
    required this.modelCategory,
  });

  @override
  ConsumerState<_VersionTile> createState() => _VersionTileState();
}

class _VersionTileState extends ConsumerState<_VersionTile> {
  @override
  Widget build(BuildContext context) {
    final downloaded = ref.watch(downloadedModelsProvider).models;
    final isDownloaded = downloaded.any(
      (m) => m.versionId == widget.version.id,
    );

    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8.0),
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              widget.version.version_name,
              style: Theme.of(context).textTheme.titleSmall,
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                const Icon(Icons.download_outlined, size: 14),
                const SizedBox(width: 4),
                Text(
                  _formatCount(widget.version.download_count),
                  style: Theme.of(context).textTheme.labelSmall,
                ),
                const SizedBox(width: 16),
                const Icon(Icons.star_outline, size: 14),
                const SizedBox(width: 4),
                Text(
                  widget.version.num_ratings > 0
                      ? '${widget.version.rating_avg.toStringAsFixed(1)} (${_formatCount(widget.version.num_ratings)})'
                      : 'No ratings',
                  style: Theme.of(context).textTheme.labelSmall,
                ),
                const SizedBox(width: 16),
                Text(
                  '${(widget.version.file_size_bytes / 1024 / 1024).toStringAsFixed(1)} MB',
                  style: Theme.of(context).textTheme.labelSmall,
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              widget.version.license_type,
              style: Theme.of(context).textTheme.labelSmall?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child:
                  isDownloaded
                      ? ElevatedButton.icon(
                        onPressed: () {
                          final dm = ref
                              .read(downloadedModelsProvider)
                              .models
                              .firstWhere(
                                (m) => m.versionId == widget.version.id,
                              );
                          final nav = Navigator.of(context);
                          nav.pop(); // close dialog
                          _launchInference(nav.context, dm);
                        },
                        icon: const Icon(Icons.play_arrow),
                        label: const Text('Run'),
                      )
                      : ElevatedButton.icon(
                        onPressed: _handleDownload,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF0D9488),
                          foregroundColor: Colors.white,
                        ),
                        icon: const Icon(Icons.download),
                        label: const Text('Download'),
                      ),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _handleDownload() async {
    final version = widget.version;
    final modelName = widget.modelName;

    // Show storage-aware confirmation before doing anything.
    final confirmed = await showDialog<bool>(
      context: context,
      builder:
          (_) => _StorageConfirmDialog(version: version, modelName: modelName),
    );
    if (confirmed != true || !mounted) return;

    // Capture everything before closing — widget will be disposed after the pop.
    final messenger = ScaffoldMessenger.of(context);
    final api = ref.read(apiServiceProvider);
    final notifier = ref.read(downloadedModelsProvider.notifier);
    final inProgressNotifier = ref.read(inProgressDownloadsProvider.notifier);

    // Compute the asset list here so we know totalSteps before closing.
    final assetKeys = <String>[
      'tflite',
      if (version.assets.labels != null) 'labels',
      if (version.assets.anchors != null) 'anchors',
      if (version.assets.tokenizer != null) 'tokenizer',
      if (version.assets.vocab != null) 'vocab',
    ];
    final totalSteps =
        assetKeys.length + (version.pipeline_spec != null ? 1 : 0);

    // Register with in-progress provider so the My AI tab shows the card immediately.
    inProgressNotifier.startDownload(
      InProgressDownload(
        versionId: version.id,
        modelName: modelName,
        versionName: version.version_name,
        completedSteps: 0,
        totalSteps: totalSteps,
      ),
    );

    // Close the parent dialog and navigate directly to the My AI tab.
    Navigator.pop(context);
    ref.read(selectedIndexProvider.notifier).state = 1;
    messenger.showSnackBar(
      SnackBar(
        content: Text('Downloading $modelName\u2026'),
        duration: const Duration(seconds: 2),
      ),
    );

    try {
      final appDir = await getApplicationDocumentsDirectory();
      final versionDir = Directory(
        '${appDir.path}/downloaded_models/${version.id}',
      );
      debugPrint(
        '[Download] Starting: $modelName v${version.version_name} (${version.id})',
      );
      debugPrint('[Download] Target dir: ${versionDir.path}');
      if (!await versionDir.exists()) {
        await versionDir.create(recursive: true);
        debugPrint('[Download] Created directory: ${versionDir.path}');
      }

      debugPrint('[Download] Assets to fetch: $assetKeys');

      int completedSteps = 0;
      for (final assetKey in assetKeys) {
        debugPrint('[Download] Fetching asset: $assetKey');
        final bytes = await api.downloadAsset(version.id, assetKey);
        debugPrint('[Download] Received $assetKey: ${bytes.length} bytes');
        await File('${versionDir.path}/$assetKey').writeAsBytes(bytes);
        debugPrint('[Download] Wrote $assetKey to disk');
        inProgressNotifier.updateProgress(version.id, ++completedSteps);
      }

      if (version.pipeline_spec != null) {
        debugPrint('[Download] Writing pipeline_spec.json');
        await File(
          '${versionDir.path}/pipeline_spec.json',
        ).writeAsString(jsonEncode(version.pipeline_spec!.toJson()));
        inProgressNotifier.updateProgress(version.id, ++completedSteps);
      }

      debugPrint('[Download] Complete: $modelName → ${versionDir.path}');
      inProgressNotifier.finishDownload(version.id);
      notifier.addDownloadedModel(
        DownloadedModel(
          modelId: version.model_id,
          modelName: modelName,
          versionName: version.version_name,
          versionId: version.id,
          category:
              version.pipeline_spec?.metadata.firstOrNull?.model_task ??
              'unknown',
          modelCategory: widget.modelCategory,
          downloadedAt: DateTime.now(),
          localPath: versionDir.path,
        ),
      );

      messenger.showSnackBar(
        SnackBar(
          content: Text('$modelName downloaded!'),
          duration: const Duration(seconds: 3),
        ),
      );
    } on ApiConnectionException catch (_, stack) {
      debugPrint('[Download] Connection error\n$stack');
      inProgressNotifier.finishDownload(version.id);
      messenger.showSnackBar(
        const SnackBar(
          content: Text(
            "Couldn't reach the server. Check your connection and try again.",
          ),
          duration: Duration(seconds: 5),
        ),
      );
    } on ApiTimeoutException catch (_, stack) {
      debugPrint('[Download] Timeout\n$stack');
      inProgressNotifier.finishDownload(version.id);
      messenger.showSnackBar(
        const SnackBar(
          content: Text(
            'Download timed out. Check your connection and try again.',
          ),
          duration: Duration(seconds: 5),
        ),
      );
    } catch (e, stack) {
      debugPrint('[Download] Error: $e');
      debugPrint('[Download] Stack: $stack');
      inProgressNotifier.finishDownload(version.id);
      messenger.showSnackBar(
        const SnackBar(
          content: Text('Download failed. Please try again.'),
          duration: Duration(seconds: 4),
        ),
      );
    }
  }
}

// widget for the Range tab
class RangeTab extends ConsumerWidget {
  const RangeTab({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final selectedModel = ref.watch(selectedModelProvider);

    // check if no model has been selected
    if (selectedModel == null) {
      final downloaded = ref.watch(downloadedModelsProvider).models;
      final hasDownloads = downloaded.isNotEmpty;
      final theme = Theme.of(context);
      final cs = theme.colorScheme;

      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.radar, size: 56, color: cs.outline),
              const SizedBox(height: 16),
              Text(
                'Take a model out on the range',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: cs.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 24),
              FilledButton.tonal(
                onPressed: () {
                  ref.read(selectedIndexProvider.notifier).state =
                      hasDownloads ? 1 : 0;
                },
                child: Text(hasDownloads ? 'Go to My AI' : 'Browse Models'),
              ),
            ],
          ),
        ),
      );
    }

    // Fetch versions for the selected model
    final versionsAsync = ref.watch(modelVersionsProvider(selectedModel.id));

    return versionsAsync.when(
      data: (versions) {
        // Filter to only supported versions
        final supportedVersions = versions.where((v) => v.isUsable).toList();

        if (supportedVersions.isEmpty) {
          return const Center(
            child: Text('No supported versions available for this model'),
          );
        }

        return SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                ElevatedButton(
                  onPressed: () {
                    ref.read(selectedModelProvider.notifier).state = null;
                    ref.read(selectedIndexProvider.notifier).state = 0;
                  },
                  child: const Text('Back to Model List'),
                ),
                const SizedBox(height: 16),
                Text(
                  'Model: ${selectedModel.name}',
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                const SizedBox(height: 8),
                Text('Description: ${selectedModel.description}'),
                Text('Total Downloads: ${selectedModel.total_download_count}'),
                const SizedBox(height: 16),
                Text(
                  'Supported Versions (${supportedVersions.length})',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 12),
                ListView.builder(
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  itemCount: supportedVersions.length,
                  itemBuilder: (context, index) {
                    final version = supportedVersions[index];
                    return VersionCard(
                      version: version,
                      modelName: selectedModel.name,
                      modelCategory: selectedModel.category,
                    );
                  },
                ),
              ],
            ),
          ),
        );
      },
      loading: () => const Center(child: CircularProgressIndicator()),
      error:
          (err, stack) => Center(child: Text('Error loading versions: $err')),
    );
  }
}

// Widget to display individual version details
class VersionCard extends ConsumerWidget {
  final ModelVersion version;
  final String modelName;
  final String modelCategory;

  const VersionCard({
    required this.version,
    required this.modelName,
    required this.modelCategory,
    super.key,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8.0),
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              version.version_name,
              style: Theme.of(context).textTheme.titleSmall,
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                const Icon(Icons.download_outlined, size: 14),
                const SizedBox(width: 4),
                Text(
                  _formatCount(version.download_count),
                  style: Theme.of(context).textTheme.labelSmall,
                ),
                const SizedBox(width: 16),
                const Icon(Icons.star_outline, size: 14),
                const SizedBox(width: 4),
                Text(
                  version.num_ratings > 0
                      ? '${version.rating_avg.toStringAsFixed(1)} (${_formatCount(version.num_ratings)})'
                      : 'No ratings',
                  style: Theme.of(context).textTheme.labelSmall,
                ),
                const SizedBox(width: 16),
                Text(
                  '${(version.file_size_bytes / 1024 / 1024).toStringAsFixed(1)} MB',
                  style: Theme.of(context).textTheme.labelSmall,
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              version.license_type,
              style: Theme.of(context).textTheme.labelSmall?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 12),
            if (version.pipeline_spec != null)
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Pipeline Configuration:',
                    style: Theme.of(context).textTheme.labelMedium,
                  ),
                  const SizedBox(height: 8),
                  PipelineConfigDisplay(config: version.pipeline_spec!),
                  const SizedBox(height: 12),
                ],
              ),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                ElevatedButton.icon(
                  onPressed: () {
                    _showDownloadDialog(context, ref, version, modelName);
                  },
                  icon: const Icon(Icons.download),
                  label: const Text('Download Model'),
                ),
                ElevatedButton.icon(
                  onPressed: () {
                    ref
                        .read(selectedModelVersionProvider.notifier)
                        .setVersion(version);
                  },
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Use Model'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  void _showDownloadDialog(
    BuildContext context,
    WidgetRef ref,
    ModelVersion version,
    String modelName,
  ) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder:
          (_) => _StorageConfirmDialog(version: version, modelName: modelName),
    );
    if (confirmed != true || !context.mounted) return;

    final messenger = ScaffoldMessenger.of(context);
    final api = ref.read(apiServiceProvider);
    final notifier = ref.read(downloadedModelsProvider.notifier);

    // Show a persistent in-progress snackbar.
    messenger.showSnackBar(
      SnackBar(
        content: Text('Downloading $modelName\u2026'),
        duration: const Duration(minutes: 10),
      ),
    );

    try {
      final appDir = await getApplicationDocumentsDirectory();
      final versionDir = Directory(
        '${appDir.path}/downloaded_models/${version.id}',
      );
      debugPrint(
        '[Download] Starting: $modelName v${version.version_name} (${version.id})',
      );
      debugPrint('[Download] Target dir: ${versionDir.path}');
      if (!await versionDir.exists()) {
        await versionDir.create(recursive: true);
        debugPrint('[Download] Created directory: ${versionDir.path}');
      }

      final assetKeys = <String>[
        'tflite',
        if (version.assets.labels != null) 'labels',
        if (version.assets.anchors != null) 'anchors',
        if (version.assets.tokenizer != null) 'tokenizer',
        if (version.assets.vocab != null) 'vocab',
      ];
      debugPrint('[Download] Assets to fetch: $assetKeys');

      for (final assetKey in assetKeys) {
        debugPrint('[Download] Fetching asset: $assetKey');
        final bytes = await api.downloadAsset(version.id, assetKey);
        debugPrint('[Download] Received $assetKey: ${bytes.length} bytes');
        await File('${versionDir.path}/$assetKey').writeAsBytes(bytes);
        debugPrint('[Download] Wrote $assetKey to disk');
      }

      if (version.pipeline_spec != null) {
        debugPrint('[Download] Writing pipeline_spec.json');
        await File(
          '${versionDir.path}/pipeline_spec.json',
        ).writeAsString(jsonEncode(version.pipeline_spec!.toJson()));
      }

      debugPrint('[Download] Complete: $modelName → ${versionDir.path}');
      notifier.addDownloadedModel(
        DownloadedModel(
          modelId: version.model_id,
          modelName: modelName,
          versionName: version.version_name,
          versionId: version.id,
          category:
              version.pipeline_spec?.metadata.firstOrNull?.model_task ??
              'unknown',
          modelCategory: modelCategory,
          downloadedAt: DateTime.now(),
          localPath: versionDir.path,
        ),
      );

      messenger.hideCurrentSnackBar();
      messenger.showSnackBar(
        SnackBar(
          content: Text('$modelName downloaded! Check the "My AI" tab.'),
          duration: const Duration(seconds: 4),
        ),
      );
    } catch (e, stack) {
      debugPrint('[Download] Error: $e');
      debugPrint('[Download] Stack: $stack');
      messenger.hideCurrentSnackBar();
      messenger.showSnackBar(
        SnackBar(
          content: Text('Download failed: $e'),
          duration: const Duration(seconds: 4),
        ),
      );
    }
  }
}

// Confirmation dialog that shows model size vs available device storage
// before the user commits to a download.
class _StorageConfirmDialog extends StatefulWidget {
  final ModelVersion version;
  final String modelName;

  const _StorageConfirmDialog({required this.version, required this.modelName});

  @override
  State<_StorageConfirmDialog> createState() => _StorageConfirmDialogState();
}

class _StorageConfirmDialogState extends State<_StorageConfirmDialog> {
  int? _freeBytes;
  bool _loadingSpace = true;

  @override
  void initState() {
    super.initState();
    _loadFreeSpace();
  }

  Future<void> _loadFreeSpace() async {
    try {
      final appDir = await getApplicationDocumentsDirectory();
      final free = await _getFreeDiskBytes(appDir.path);
      if (mounted) {
        setState(() {
          _freeBytes = free;
          _loadingSpace = false;
        });
      }
    } catch (_) {
      if (mounted) {
        setState(() {
          _loadingSpace = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final modelBytes = widget.version.file_size_bytes;
    final tooLow = _freeBytes != null && _freeBytes! < modelBytes;

    return AlertDialog(
      title: const Text('Download Model?'),
      content: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Model: ${widget.modelName}'),
            Text('Version: ${widget.version.version_name}'),
            const SizedBox(height: 12),
            // Storage comparison
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Download size:'),
                Text(
                  _formatBytes(modelBytes),
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
              ],
            ),
            const SizedBox(height: 4),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Available storage:'),
                _loadingSpace
                    ? const SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                    : Text(
                      _freeBytes != null
                          ? _formatBytes(_freeBytes!)
                          : 'Unknown',
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: tooLow ? Colors.red : null,
                      ),
                    ),
              ],
            ),
            if (tooLow) ...[
              const SizedBox(height: 8),
              Row(
                children: [
                  const Icon(
                    Icons.warning_amber_rounded,
                    color: Colors.orange,
                    size: 16,
                  ),
                  const SizedBox(width: 6),
                  const Expanded(
                    child: Text(
                      'Not enough free space — download may fail.',
                      style: TextStyle(color: Colors.orange, fontSize: 12),
                    ),
                  ),
                ],
              ),
            ],
            const Divider(height: 24),
            const Text('Assets included:'),
            const SizedBox(height: 4),
            const Text('• TFLite model'),
            if (widget.version.assets.labels != null) const Text('• Labels'),
            if (widget.version.assets.anchors != null) const Text('• Anchors'),
            if (widget.version.assets.tokenizer != null)
              const Text('• Tokenizer'),
            if (widget.version.assets.vocab != null) const Text('• Vocab'),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context, false),
          child: const Text('Cancel'),
        ),
        ElevatedButton(
          onPressed: () => Navigator.pop(context, true),
          child: const Text('Download'),
        ),
      ],
    );
  }
}

// Widget to display pipeline configuration details
class PipelineConfigDisplay extends StatelessWidget {
  final PipelineConfig config;

  const PipelineConfigDisplay({required this.config, super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Inputs: ${config.inputs.length}',
          style: const TextStyle(fontSize: 12),
        ),
        for (var input in config.inputs)
          Padding(
            padding: const EdgeInsets.only(left: 16.0),
            child: Text(
              '• ${input.name} [${input.shape.join(',')}}] - ${input.dtype}',
              style: const TextStyle(fontSize: 10),
            ),
          ),
        const SizedBox(height: 8),
        Text(
          'Preprocessing: ${config.preprocessing.length} block(s)',
          style: const TextStyle(fontSize: 12),
        ),
        for (var block in config.preprocessing)
          Padding(
            padding: const EdgeInsets.only(left: 16.0),
            child: Text(
              '• ${block.input_name} (${block.steps.length} steps)',
              style: const TextStyle(fontSize: 10),
            ),
          ),
        const SizedBox(height: 8),
        Text(
          'Postprocessing: ${config.postprocessing.length} block(s)',
          style: const TextStyle(fontSize: 12),
        ),
        for (var block in config.postprocessing)
          Padding(
            padding: const EdgeInsets.only(left: 16.0),
            child: Text(
              '• ${block.output_name} (${block.interpretation})',
              style: const TextStyle(fontSize: 10),
            ),
          ),
      ],
    );
  }
}

class _VersionStatRow extends ConsumerWidget {
  final String versionId;
  const _VersionStatRow({required this.versionId});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final summaryAsync = ref.watch(versionStatSummaryProvider(versionId));
    return summaryAsync.when(
      data: (summary) {
        if (summary == null || summary.totalRuns == 0) {
          return const SizedBox.shrink();
        }
        return Padding(
          padding: const EdgeInsets.only(top: 4),
          child: Row(
            children: [
              const Icon(Icons.speed, size: 13),
              const SizedBox(width: 4),
              Flexible(
                child: Text(
                  '${summary.totalRuns} run${summary.totalRuns == 1 ? '' : 's'} · avg ${summary.avgLatencyMs.toStringAsFixed(0)}ms',
                  style: Theme.of(context).textTheme.labelSmall,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              if ((summary.avgConfidence ?? 0) > 0) ...[
                const SizedBox(width: 8),
                const Icon(Icons.check_circle_outline, size: 13),
                const SizedBox(width: 4),
                Flexible(
                  child: Text(
                    '${(summary.avgConfidence! * 100).toStringAsFixed(0)}% avg confidence',
                    style: Theme.of(context).textTheme.labelSmall,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ],
          ),
        );
      },
      loading: () => const SizedBox.shrink(),
      error: (_, __) => const SizedBox.shrink(),
    );
  }
}

// Widget to display downloaded models
class DownloadedModels extends ConsumerStatefulWidget {
  const DownloadedModels({super.key});

  @override
  ConsumerState<DownloadedModels> createState() => _DownloadedModelsState();
}

class _DownloadedModelsState extends ConsumerState<DownloadedModels> {
  final _searchController = TextEditingController();
  String _searchQuery = '';
  String? _selectedTask;
  String? _selectedCategory;
  int _viewIndex = 0; // 0 = models, 1 = stats

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  void _invalidateStats() {
    final models = ref.read(downloadedModelsProvider).models;
    for (final m in models) {
      ref.invalidate(versionStatSummaryProvider(m.versionId));
      ref.invalidate(versionStatsProvider(m.versionId));
    }
  }

  @override
  Widget build(BuildContext context) {
    final dmState = ref.watch(downloadedModelsProvider);
    final downloadedModels = dmState.models;
    final inProgress = ref.watch(inProgressDownloadsProvider);
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // ── View toggle ──────────────────────────────────────────────────────
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
          child: SegmentedButton<int>(
            segments: const [
              ButtonSegment(
                value: 0,
                label: Text('My Models'),
                icon: Icon(Icons.cloud_download_outlined),
              ),
              ButtonSegment(
                value: 1,
                label: Text('Stats'),
                icon: Icon(Icons.bar_chart_outlined),
              ),
            ],
            selected: {_viewIndex},
            onSelectionChanged: (s) {
              setState(() => _viewIndex = s.first);
              if (s.first == 1) _invalidateStats();
            },
          ),
        ),
        // ── View content ─────────────────────────────────────────────────────
        Expanded(
          child:
              _viewIndex == 0
                  ? _buildModelsView(
                    context,
                    downloadedModels,
                    inProgress,
                    cs,
                    isLoaded: dmState.isLoaded,
                  )
                  : _buildStatsView(context, downloadedModels, theme, cs),
        ),
      ],
    );
  }

  Widget _buildModelsView(
    BuildContext context,
    List<DownloadedModel> downloadedModels,
    Map<String, InProgressDownload> inProgress,
    ColorScheme cs, {
    required bool isLoaded,
  }) {
    if (!isLoaded && inProgress.isEmpty) {
      return const Center(child: CircularProgressIndicator());
    }
    if (downloadedModels.isEmpty && inProgress.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.cloud_download_outlined,
              size: 64,
              color: Colors.grey[400],
            ),
            const SizedBox(height: 16),
            Text(
              'No downloaded models yet',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            Text(
              'Download models from the Home page to use them offline',
              style: Theme.of(context).textTheme.bodySmall,
              textAlign: TextAlign.center,
            ),
          ],
        ),
      );
    }

    // Collect unique tasks and categories from downloaded models
    final tasks =
        downloadedModels
            .map((m) => m.category)
            .where((t) => t.isNotEmpty && t != 'unknown')
            .toSet()
            .toList()
          ..sort();
    final categories =
        downloadedModels
            .map((m) => m.modelCategory)
            .where((c) => c.isNotEmpty)
            .toSet()
            .toList()
          ..sort();

    // Apply search + filters
    final filtered =
        downloadedModels.where((m) {
          final q = _searchQuery.toLowerCase();
          final matchesSearch =
              q.isEmpty ||
              m.modelName.toLowerCase().contains(q) ||
              m.category.toLowerCase().contains(q);
          final matchesTask =
              _selectedTask == null || m.category == _selectedTask;
          final matchesCategory =
              _selectedCategory == null || m.modelCategory == _selectedCategory;
          return matchesSearch && matchesTask && matchesCategory;
        }).toList();

    final totalCount = downloadedModels.length + inProgress.length;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Search bar
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
          child: TextField(
            controller: _searchController,
            decoration: InputDecoration(
              hintText: 'Search my models\u2026',
              prefixIcon: const Icon(Icons.search),
              suffixIcon:
                  _searchQuery.isNotEmpty
                      ? IconButton(
                        icon: const Icon(Icons.clear),
                        onPressed: () {
                          _searchController.clear();
                          setState(() => _searchQuery = '');
                        },
                      )
                      : null,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              isDense: true,
              contentPadding: const EdgeInsets.symmetric(vertical: 10),
            ),
            onChanged: (v) => setState(() => _searchQuery = v),
          ),
        ),
        // Task filter chips
        if (tasks.isNotEmpty)
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 0, 12, 0),
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: [
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 4),
                    child: FilterChip(
                      label: const Text('All Tasks'),
                      selected: _selectedTask == null,
                      onSelected: (_) => setState(() => _selectedTask = null),
                    ),
                  ),
                  for (final task in tasks)
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 4),
                      child: FilterChip(
                        label: Text(_friendlyTask(task)),
                        selected: _selectedTask == task,
                        onSelected:
                            (selected) => setState(
                              () => _selectedTask = selected ? task : null,
                            ),
                      ),
                    ),
                ],
              ),
            ),
          ),
        // Count header
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 6, 16, 0),
          child: Text(
            'My AI ($totalCount)',
            style: Theme.of(context).textTheme.titleSmall?.copyWith(
              color: const Color(0xFF526870),
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
        // Scrollable content
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // In-progress download cards (always visible regardless of filter)
                for (final download in inProgress.values)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: Card(
                      child: Padding(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 10,
                        ),
                        child: Row(
                          children: [
                            Container(
                              width: 42,
                              height: 42,
                              decoration: BoxDecoration(
                                color: const Color(0xFFFAE5D8),
                                borderRadius: BorderRadius.circular(10),
                              ),
                              child: const Icon(
                                Icons.downloading,
                                color: Color(0xFF8C3A21),
                                size: 22,
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    download.modelName,
                                    style: Theme.of(context)
                                        .textTheme
                                        .bodyMedium
                                        ?.copyWith(fontWeight: FontWeight.w600),
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                  const SizedBox(height: 2),
                                  Text(
                                    'v${download.versionName} · ${(download.progress * 100).toInt()}%',
                                    style: Theme.of(
                                      context,
                                    ).textTheme.bodySmall?.copyWith(
                                      color: const Color(0xFF526870),
                                    ),
                                  ),
                                  const SizedBox(height: 6),
                                  LinearProgressIndicator(
                                    value: download.progress,
                                    borderRadius: BorderRadius.circular(4),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                // Filtered download cards
                if (filtered.isEmpty && downloadedModels.isNotEmpty)
                  const Padding(
                    padding: EdgeInsets.symmetric(vertical: 32),
                    child: Center(child: Text('No models match your filters.')),
                  )
                else
                  ...filtered.map(
                    (model) => Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Card(
                        clipBehavior: Clip.antiAlias,
                        child: InkWell(
                          onTap: () => _launchInference(context, model),
                          child: Padding(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 12,
                              vertical: 10,
                            ),
                            child: Row(
                              children: [
                                Container(
                                  width: 42,
                                  height: 42,
                                  decoration: BoxDecoration(
                                    color:
                                        Theme.of(
                                          context,
                                        ).colorScheme.primaryContainer,
                                    borderRadius: BorderRadius.circular(10),
                                  ),
                                  child: Icon(
                                    _taskIcon(model.category.toLowerCase()),
                                    color:
                                        Theme.of(context).colorScheme.primary,
                                    size: 22,
                                  ),
                                ),
                                const SizedBox(width: 12),
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      Row(
                                        children: [
                                          Expanded(
                                            child: Text(
                                              model.modelName,
                                              style: Theme.of(
                                                context,
                                              ).textTheme.bodyMedium?.copyWith(
                                                fontWeight: FontWeight.w600,
                                              ),
                                              maxLines: 1,
                                              overflow: TextOverflow.ellipsis,
                                            ),
                                          ),
                                          if (model.category != 'unknown' &&
                                              model.category.isNotEmpty)
                                            Container(
                                              padding:
                                                  const EdgeInsets.symmetric(
                                                    horizontal: 6,
                                                    vertical: 2,
                                                  ),
                                              decoration: BoxDecoration(
                                                color:
                                                    Theme.of(context)
                                                        .colorScheme
                                                        .secondaryContainer,
                                                borderRadius:
                                                    BorderRadius.circular(4),
                                              ),
                                              child: Text(
                                                _friendlyTask(model.category),
                                                style: TextStyle(
                                                  fontSize: 10,
                                                  fontWeight: FontWeight.w600,
                                                  color:
                                                      Theme.of(context)
                                                          .colorScheme
                                                          .onSecondaryContainer,
                                                ),
                                              ),
                                            ),
                                        ],
                                      ),
                                      const SizedBox(height: 2),
                                      Text(
                                        'v${model.versionName}',
                                        style: Theme.of(
                                          context,
                                        ).textTheme.bodySmall?.copyWith(
                                          color: const Color(0xFF526870),
                                        ),
                                      ),
                                      _VersionStatRow(
                                        versionId: model.versionId,
                                      ),
                                    ],
                                  ),
                                ),
                                const SizedBox(width: 4),
                                IconButton(
                                  icon: const Icon(
                                    Icons.delete_outline,
                                    size: 20,
                                  ),
                                  color: const Color(0xFF829CA5),
                                  onPressed: () {
                                    ref
                                        .read(downloadedModelsProvider.notifier)
                                        .removeDownloadedModel(model.versionId);
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      const SnackBar(
                                        content: Text(
                                          'Model removed successfully',
                                        ),
                                        duration: Duration(seconds: 2),
                                      ),
                                    );
                                  },
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildStatsView(
    BuildContext context,
    List<DownloadedModel> downloadedModels,
    ThemeData theme,
    ColorScheme cs,
  ) {
    return RefreshIndicator(
      onRefresh: () async => _invalidateStats(),
      child: ListView(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Inference Stats',
                style: theme.textTheme.titleSmall?.copyWith(
                  color: const Color(0xFF526870),
                  fontWeight: FontWeight.w600,
                ),
              ),
              IconButton(
                icon: const Icon(Icons.refresh, size: 20),
                tooltip: 'Refresh',
                onPressed: _invalidateStats,
              ),
            ],
          ),
          const SizedBox(height: 4),
          if (downloadedModels.isEmpty)
            Card(
              child: Padding(
                padding: const EdgeInsets.symmetric(
                  vertical: 40,
                  horizontal: 16,
                ),
                child: Center(
                  child: Column(
                    children: [
                      Icon(
                        Icons.bar_chart_outlined,
                        size: 48,
                        color: Colors.grey[400],
                      ),
                      const SizedBox(height: 12),
                      Text(
                        'No inference stats yet',
                        style: theme.textTheme.titleSmall,
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Download and run models to see stats here.',
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: cs.onSurfaceVariant,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              ),
            )
          else
            ...downloadedModels.map(
              (model) => Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: _ModelStatCard(model: model),
              ),
            ),
        ],
      ),
    );
  }
}

void _launchInference(BuildContext context, DownloadedModel model) {
  final tflitePath = '${model.localPath}/tflite';
  final pipelinePath = '${model.localPath}/pipeline_spec.json';

  Widget? inferenceWidget;
  switch (model.category) {
    case 'image_classification':
    case 'image-classification':
      inferenceWidget = ImageClassificationWidget(
        modelName: tflitePath,
        pipelinePath: pipelinePath,
        isLocalFile: true,
        localDir: model.localPath,
        modelVersionId: model.versionId,
        modelDisplayName: model.modelName,
      );
    case 'object_detection':
    case 'object-detection':
      inferenceWidget = ObjectDetectionWidget(
        modelName: tflitePath,
        pipelinePath: pipelinePath,
        isLocalFile: true,
        localDir: model.localPath,
        modelVersionId: model.versionId,
        modelDisplayName: model.modelName,
      );
    case 'text_generation':
    case 'text-generation':
      inferenceWidget = TextGenerationWidget(
        modelName: tflitePath,
        pipelinePath: pipelinePath,
        isLocalFile: true,
        localDir: model.localPath,
        modelVersionId: model.versionId,
        modelDisplayName: model.modelName,
      );
    case 'semantic_segmentation':
    case 'image_segmentation':
    case 'image-segmentation':
    case 'semantic-segmentation':
    case 'segmentation':
      inferenceWidget = SemanticSegmentationWidget(
        modelName: tflitePath,
        pipelinePath: pipelinePath,
        isLocalFile: true,
        localDir: model.localPath,
        modelVersionId: model.versionId,
        modelDisplayName: model.modelName,
      );
    case 'automatic_speech_recognition':
    case 'automatic-speech-recognition':
      inferenceWidget = AutomaticSpeechRecognitionWidget(
        modelName: tflitePath,
        pipelinePath: pipelinePath,
        isLocalFile: true,
        localDir: model.localPath,
        modelVersionId: model.versionId,
        modelDisplayName: model.modelName,
      );
    case 'image_to_text':
    case 'image-to-text':
      inferenceWidget = ImageToTextWidget(
        modelName: tflitePath,
        pipelinePath: pipelinePath,
        isLocalFile: true,
        localDir: model.localPath,
        modelVersionId: model.versionId,
        modelDisplayName: model.modelName,
      );
    case 'text_to_image':
    case 'text-to-image':
      inferenceWidget = TextToImageWidget(
        modelName: tflitePath,
        pipelinePath: pipelinePath,
        isLocalFile: true,
        localDir: model.localPath,
        modelVersionId: model.versionId,
        modelDisplayName: model.modelName,
      );
    default:
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Task type "${model.category}" is not yet supported for on-device inference.',
          ),
        ),
      );
      return;
  }

  Navigator.push(context, MaterialPageRoute(builder: (_) => inferenceWidget!));
}

// widget for the user
class Profile extends ConsumerWidget {
  const Profile({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final optedIn = ref.watch(telemetryOptInProvider);
    final user = ref.watch(currentUserProvider);
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        const SizedBox(height: 8),
        Text(
          'Profile',
          style: theme.textTheme.titleSmall?.copyWith(
            color: const Color(0xFF526870),
            fontWeight: FontWeight.w600,
          ),
        ),
        const SizedBox(height: 12),

        // ── Account ───────────────────────────────────────────────────────────
        if (user == null)
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.person_outline, size: 18, color: cs.primary),
                      const SizedBox(width: 8),
                      Text('Account', style: theme.textTheme.titleSmall),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Sign in to sync inference telemetry and access your account across devices.',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: cs.onSurfaceVariant,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Expanded(
                        child: FilledButton(
                          onPressed:
                              () => showDialog<void>(
                                context: context,
                                builder:
                                    (_) => const _AuthDialog(
                                      initialMode: _AuthMode.signIn,
                                    ),
                              ),
                          child: const Text('Sign in'),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: OutlinedButton(
                          onPressed:
                              () => showDialog<void>(
                                context: context,
                                builder:
                                    (_) => const _AuthDialog(
                                      initialMode: _AuthMode.createAccount,
                                    ),
                              ),
                          child: const Text('Create account'),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          )
        else
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                children: [
                  CircleAvatar(
                    backgroundColor: cs.primaryContainer,
                    radius: 20,
                    child: Text(
                      (user.email ?? '?')[0].toUpperCase(),
                      style: theme.textTheme.titleMedium?.copyWith(
                        color: cs.primary,
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          user.email ?? 'Jacana user',
                          style: theme.textTheme.bodyMedium?.copyWith(
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        Text(
                          'Signed in',
                          style: theme.textTheme.bodySmall?.copyWith(
                            color: cs.onSurfaceVariant,
                          ),
                        ),
                      ],
                    ),
                  ),
                  TextButton(
                    onPressed: () async {
                      await ref.read(authServiceProvider).signOut();
                    },
                    child: const Text('Sign out'),
                  ),
                ],
              ),
            ),
          ),

        const SizedBox(height: 16),

        // ── Privacy & Telemetry ───────────────────────────────────────────────
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.shield_outlined, size: 18, color: cs.primary),
                    const SizedBox(width: 8),
                    Text('Privacy', style: theme.textTheme.titleSmall),
                  ],
                ),
                const SizedBox(height: 8),
                Text(
                  'Inference stats are always stored locally on your device. '
                  'Enable below to share them anonymously — requires signing in '
                  'to your Jacana account.',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: cs.onSurfaceVariant,
                  ),
                ),
                const Divider(height: 24),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  title: const Text('Share inference telemetry'),
                  subtitle: const Text(
                    'Latency, accuracy metrics, and device model — no personal data.',
                  ),
                  value: optedIn,
                  onChanged:
                      (v) =>
                          ref.read(telemetryOptInProvider.notifier).setValue(v),
                ),
              ],
            ),
          ),
        ),

        const SizedBox(height: 16),

        // ── Stats shortcut hint ───────────────────────────────────────────────
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Container(
                  width: 36,
                  height: 36,
                  decoration: BoxDecoration(
                    color: cs.primaryContainer,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Icon(
                    Icons.bar_chart_outlined,
                    color: cs.primary,
                    size: 20,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Inference Stats',
                        style: theme.textTheme.bodyMedium?.copyWith(
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        'View per-model stats in the My AI tab.',
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: cs.onSurfaceVariant,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

// ── Auth dialog ─────────────────────────────────────────────────────────────

enum _AuthMode { signIn, createAccount }

class _AuthDialog extends ConsumerStatefulWidget {
  const _AuthDialog({required this.initialMode});

  final _AuthMode initialMode;

  @override
  ConsumerState<_AuthDialog> createState() => _AuthDialogState();
}

class _AuthDialogState extends ConsumerState<_AuthDialog> {
  late _AuthMode _mode;
  final _emailCtrl = TextEditingController();
  final _passwordCtrl = TextEditingController();
  bool _loading = false;
  String? _error;
  bool _obscurePassword = true;

  @override
  void initState() {
    super.initState();
    _mode = widget.initialMode;
  }

  @override
  void dispose() {
    _emailCtrl.dispose();
    _passwordCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    final email = _emailCtrl.text.trim();
    final password = _passwordCtrl.text;
    if (email.isEmpty || password.isEmpty) {
      setState(() => _error = 'Please enter your email and password.');
      return;
    }
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final svc = ref.read(authServiceProvider);
      if (_mode == _AuthMode.signIn) {
        await svc.signIn(email, password);
      } else {
        final res = await svc.signUp(email, password);
        // Supabase may require email confirmation before the session is active.
        if (res.session == null) {
          if (mounted) {
            Navigator.of(context).pop();
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text(
                  'Check your inbox to confirm your email, then sign in.',
                ),
              ),
            );
          }
          return;
        }
      }
      if (mounted) Navigator.of(context).pop();
    } on AuthException catch (e) {
      setState(() => _error = e.message);
    } on SocketException {
      setState(
        () => _error = "Couldn't reach the server. Check your connection.",
      );
    } on TimeoutException {
      setState(() => _error = 'The request timed out. Please try again.');
    } catch (e) {
      final msg = e.toString().toLowerCase();
      setState(
        () =>
            _error =
                (msg.contains('socket') ||
                        msg.contains('connection') ||
                        msg.contains('network'))
                    ? "Couldn't reach the server. Check your connection."
                    : 'Something went wrong. Please try again.',
      );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final isSignIn = _mode == _AuthMode.signIn;

    return AlertDialog(
      title: Text(isSignIn ? 'Sign in to Jacana' : 'Create account'),
      content: SizedBox(
        width: 340,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextField(
              controller: _emailCtrl,
              keyboardType: TextInputType.emailAddress,
              textInputAction: TextInputAction.next,
              autocorrect: false,
              decoration: const InputDecoration(
                labelText: 'Email',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _passwordCtrl,
              obscureText: _obscurePassword,
              textInputAction: TextInputAction.done,
              onSubmitted: (_) => _loading ? null : _submit(),
              decoration: InputDecoration(
                labelText: 'Password',
                border: const OutlineInputBorder(),
                suffixIcon: IconButton(
                  icon: Icon(
                    _obscurePassword
                        ? Icons.visibility_outlined
                        : Icons.visibility_off_outlined,
                  ),
                  onPressed:
                      () =>
                          setState(() => _obscurePassword = !_obscurePassword),
                ),
              ),
            ),
            if (_error != null) ...[
              const SizedBox(height: 12),
              Text(
                _error!,
                style: theme.textTheme.bodySmall?.copyWith(color: cs.error),
              ),
            ],
            const SizedBox(height: 16),
            FilledButton(
              onPressed: _loading ? null : _submit,
              child:
                  _loading
                      ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                      : Text(isSignIn ? 'Sign in' : 'Create account'),
            ),
            const SizedBox(height: 8),
            TextButton(
              onPressed:
                  _loading
                      ? null
                      : () => setState(() {
                        _mode =
                            isSignIn
                                ? _AuthMode.createAccount
                                : _AuthMode.signIn;
                        _error = null;
                      }),
              child: Text(
                isSignIn
                    ? "Don't have an account? Create one"
                    : 'Already have an account? Sign in',
              ),
            ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: _loading ? null : () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
      ],
    );
  }
}

// ── Per-model stats card ────────────────────────────────────────────────────

class _ModelStatCard extends ConsumerStatefulWidget {
  final DownloadedModel model;
  const _ModelStatCard({required this.model});

  @override
  ConsumerState<_ModelStatCard> createState() => _ModelStatCardState();
}

class _ModelStatCardState extends ConsumerState<_ModelStatCard> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final summaryAsync = ref.watch(
      versionStatSummaryProvider(widget.model.versionId),
    );
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: () => setState(() => _expanded = !_expanded),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const SizedBox(height: 4),
                        Text(
                          widget.model.modelName,
                          style: theme.textTheme.titleSmall,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                        Text(
                          'v${widget.model.versionName}',
                          style: theme.textTheme.bodySmall?.copyWith(
                            color: cs.onSurfaceVariant,
                          ),
                        ),
                      ],
                    ),
                  ),
                  Icon(
                    _expanded ? Icons.expand_less : Icons.expand_more,
                    color: cs.onSurfaceVariant,
                  ),
                ],
              ),
              const Divider(height: 20),

              // Summary + optional recent runs
              summaryAsync.when(
                data: (summary) {
                  if (summary == null || summary.totalRuns == 0) {
                    return Text(
                      'No inference runs recorded yet.',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: cs.onSurfaceVariant,
                      ),
                    );
                  }
                  return Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _StatsSummaryRow(summary: summary),
                      if (_expanded) ...[
                        const SizedBox(height: 16),
                        _RecentRunsList(versionId: widget.model.versionId),
                      ],
                    ],
                  );
                },
                loading: () => const LinearProgressIndicator(),
                error:
                    (_, __) => Text(
                      'Could not load stats.',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: cs.error,
                      ),
                    ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Summary stat row ────────────────────────────────────────────────────────

class _StatsSummaryRow extends StatelessWidget {
  final InferenceStatSummary summary;
  const _StatsSummaryRow({required this.summary});

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 16,
      runSpacing: 8,
      children: [
        _StatChip(
          icon: Icons.play_circle_outline,
          label: '${summary.totalRuns} run${summary.totalRuns == 1 ? '' : 's'}',
        ),
        _StatChip(
          icon: Icons.speed,
          label: '${summary.avgLatencyMs.toStringAsFixed(0)} ms avg',
        ),
        _StatChip(
          icon: Icons.check_circle_outline,
          label: '${(summary.successRate * 100).toStringAsFixed(0)}% success',
        ),
        if (summary.avgConfidence != null && summary.avgConfidence! > 0)
          _StatChip(
            icon: Icons.star_outline,
            label: '${(summary.avgConfidence! * 100).toStringAsFixed(0)}% conf',
          ),
        if (summary.lastRunAt != null)
          _StatChip(icon: Icons.history, label: _timeAgo(summary.lastRunAt!)),
      ],
    );
  }

  static String _timeAgo(DateTime dt) {
    final diff = DateTime.now().difference(dt);
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    if (diff.inDays < 7) return '${diff.inDays}d ago';
    return '${dt.month}/${dt.day}/${dt.year}';
  }
}

class _StatChip extends StatelessWidget {
  final IconData icon;
  final String label;
  const _StatChip({required this.icon, required this.label});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 14, color: cs.onSurfaceVariant),
        const SizedBox(width: 4),
        Text(label, style: Theme.of(context).textTheme.labelSmall),
      ],
    );
  }
}

// ── Recent runs list (expanded view) ───────────────────────────────────────

class _RecentRunsList extends ConsumerWidget {
  final String versionId;
  const _RecentRunsList({required this.versionId});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final statsAsync = ref.watch(versionStatsProvider(versionId));
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return statsAsync.when(
      data: (stats) {
        if (stats.isEmpty) return const SizedBox.shrink();
        final recent = stats.take(10).toList();
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Recent runs',
              style: theme.textTheme.labelMedium?.copyWith(
                color: cs.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 8),
            ...recent.map((s) => _RunRow(stat: s)),
          ],
        );
      },
      loading: () => const SizedBox.shrink(),
      error: (_, __) => const SizedBox.shrink(),
    );
  }
}

class _RunRow extends StatelessWidget {
  final InferenceStat stat;
  const _RunRow({required this.stat});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final dt = stat.timestamp;
    final dateLabel =
        '${dt.month}/${dt.day} ${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Icon(
            stat.success ? Icons.check_circle : Icons.error_outline,
            size: 14,
            color: stat.success ? Colors.green : cs.error,
          ),
          const SizedBox(width: 6),
          Text(
            dateLabel,
            style: theme.textTheme.labelSmall?.copyWith(
              color: cs.onSurfaceVariant,
            ),
          ),
          const SizedBox(width: 12),
          Text(
            '${stat.totalInferenceMs} ms',
            style: theme.textTheme.labelSmall,
          ),
          if (stat.topConfidence != null) ...[
            const SizedBox(width: 12),
            Text(
              '${(stat.topConfidence! * 100).toStringAsFixed(0)}% conf',
              style: theme.textTheme.labelSmall,
            ),
          ],
          const Spacer(),
          if (stat.synced)
            Icon(
              Icons.cloud_done_outlined,
              size: 12,
              color: cs.onSurfaceVariant,
            ),
        ],
      ),
    );
  }
}

// ── Jacana bird logo ─────────────────────────────────────────────────────────
// Cartoony Jacana bird drawn on a 100×100 logical canvas, scaled to widget size.
// Colours match the African Jacana: chestnut body, dark-blue head, white cheek,
// yellow frontal shield, grey legs with long splayed toes.
class JacanaLogoPainter extends CustomPainter {
  const JacanaLogoPainter();

  @override
  void paint(Canvas canvas, Size size) {
    final sx = size.width / 100;
    final sy = size.height / 100;
    Offset p(double x, double y) => Offset(x * sx, y * sy);

    Paint fill(Color c) =>
        Paint()
          ..color = c
          ..style = PaintingStyle.fill;
    Paint stroke(Color c, double w) =>
        Paint()
          ..color = c
          ..style = PaintingStyle.stroke
          ..strokeWidth = w * sx
          ..strokeCap = StrokeCap.round;

    // ── Legs ─────────────────────────────────────────────────────────────────
    const legColor = Color(0xFF7B9490);
    canvas.drawLine(p(49, 74), p(43, 93), stroke(legColor, 3));
    canvas.drawLine(p(60, 72), p(67, 91), stroke(legColor, 3));

    // Front foot toes (very long — the jacana's signature)
    for (final toe in [
      [43.0, 93.0, 33.0, 96.0],
      [43.0, 93.0, 44.0, 98.0],
      [43.0, 93.0, 53.0, 96.0],
    ]) {
      canvas.drawLine(
        p(toe[0], toe[1]),
        p(toe[2], toe[3]),
        stroke(legColor, 2),
      );
    }
    canvas.drawLine(p(43, 93), p(36, 99), stroke(legColor, 1.8));

    // Back foot toes
    for (final toe in [
      [67.0, 91.0, 57.0, 94.0],
      [67.0, 91.0, 68.0, 97.0],
      [67.0, 91.0, 77.0, 94.0],
    ]) {
      canvas.drawLine(
        p(toe[0], toe[1]),
        p(toe[2], toe[3]),
        stroke(legColor, 2),
      );
    }

    // ── Tail ─────────────────────────────────────────────────────────────────
    final tail =
        Path()
          ..moveTo(p(76, 60).dx, p(76, 60).dy)
          ..cubicTo(
            p(87, 52).dx,
            p(87, 52).dy,
            p(90, 63).dx,
            p(90, 63).dy,
            p(83, 68).dx,
            p(83, 68).dy,
          )
          ..cubicTo(
            p(79, 72).dx,
            p(79, 72).dy,
            p(73, 66).dx,
            p(73, 66).dy,
            p(74, 62).dx,
            p(74, 62).dy,
          )
          ..close();
    canvas.drawPath(tail, fill(const Color(0xFF9B3512)));

    // ── Body ─────────────────────────────────────────────────────────────────
    canvas.drawOval(
      Rect.fromCenter(center: p(54, 60), width: 50 * sx, height: 38 * sy),
      fill(const Color(0xFFC84A18)),
    );

    // Wing shading
    final wing =
        Path()
          ..moveTo(p(35, 63).dx, p(35, 63).dy)
          ..cubicTo(
            p(41, 50).dx,
            p(41, 50).dy,
            p(70, 53).dx,
            p(70, 53).dy,
            p(76, 61).dx,
            p(76, 61).dy,
          )
          ..cubicTo(
            p(72, 67).dx,
            p(72, 67).dy,
            p(56, 70).dx,
            p(56, 70).dy,
            p(40, 67).dx,
            p(40, 67).dy,
          )
          ..close();
    canvas.drawPath(wing, fill(const Color(0xFFA83C14).withValues(alpha: 0.5)));

    // ── Chest / throat (yellow) ───────────────────────────────────────────────
    final chest =
        Path()
          ..moveTo(p(34, 68).dx, p(34, 68).dy)
          ..cubicTo(
            p(26, 60).dx,
            p(26, 60).dy,
            p(27, 48).dx,
            p(27, 48).dy,
            p(34, 43).dx,
            p(34, 43).dy,
          )
          ..cubicTo(
            p(39, 48).dx,
            p(39, 48).dy,
            p(39, 60).dx,
            p(39, 60).dy,
            p(37, 68).dx,
            p(37, 68).dy,
          )
          ..close();
    canvas.drawPath(chest, fill(const Color(0xFFE09820)));

    // ── Head (dark slate blue) ───────────────────────────────────────────────
    canvas.drawCircle(p(33, 43), 16 * sx, fill(const Color(0xFF243C6A)));

    // White cheek patch (rotated ellipse)
    canvas.save();
    canvas.translate(p(30, 49).dx, p(30, 49).dy);
    canvas.rotate(-8 * 3.14159265 / 180);
    canvas.drawOval(
      Rect.fromCenter(center: Offset.zero, width: 19 * sx, height: 13 * sy),
      fill(const Color(0xFFECEAE8)),
    );
    canvas.restore();

    // Yellow frontal shield (casque)
    final shield =
        Path()
          ..moveTo(p(26, 35).dx, p(26, 35).dy)
          ..cubicTo(
            p(28, 27).dx,
            p(28, 27).dy,
            p(37, 29).dx,
            p(37, 29).dy,
            p(37, 35).dx,
            p(37, 35).dy,
          )
          ..cubicTo(
            p(34, 40).dx,
            p(34, 40).dy,
            p(28, 39).dx,
            p(28, 39).dy,
            p(26, 35).dx,
            p(26, 35).dy,
          )
          ..close();
    canvas.drawPath(shield, fill(const Color(0xFFE29420)));

    // Dark blue crown (top of head)
    final crown =
        Path()
          ..moveTo(p(22, 41).dx, p(22, 41).dy)
          ..cubicTo(
            p(22, 33).dx,
            p(22, 33).dy,
            p(27, 27).dx,
            p(27, 27).dy,
            p(34, 27).dx,
            p(34, 27).dy,
          )
          ..cubicTo(
            p(41, 27).dx,
            p(41, 27).dy,
            p(47, 33).dx,
            p(47, 33).dy,
            p(47, 41).dx,
            p(47, 41).dy,
          )
          ..cubicTo(
            p(43, 38).dx,
            p(43, 38).dy,
            p(38, 36).dx,
            p(38, 36).dy,
            p(34, 36).dx,
            p(34, 36).dy,
          )
          ..cubicTo(
            p(30, 36).dx,
            p(30, 36).dy,
            p(25, 38).dx,
            p(25, 38).dy,
            p(22, 41).dx,
            p(22, 41).dy,
          )
          ..close();
    canvas.drawPath(crown, fill(const Color(0xFF1A2D50)));

    // ── Eye ──────────────────────────────────────────────────────────────────
    canvas.drawCircle(p(28, 44), 4.5 * sx, fill(const Color(0xFF0D0D0D)));
    canvas.drawCircle(p(27, 43), 1.8 * sx, fill(Colors.white));

    // ── Beak ─────────────────────────────────────────────────────────────────
    final beak =
        Path()
          ..moveTo(p(20, 45).dx, p(20, 45).dy)
          ..lineTo(p(6, 42).dx, p(6, 42).dy)
          ..lineTo(p(20, 50).dx, p(20, 50).dy)
          ..close();
    canvas.drawPath(beak, fill(const Color(0xFF5E7872)));
    canvas.drawLine(p(7, 46), p(20, 47), stroke(const Color(0xFF4A6260), 0.8));
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

// provider for selected page
class _SelectedIndexNotifier extends Notifier<int> {
  @override
  int build() => 0;
}

final selectedIndexProvider = NotifierProvider<_SelectedIndexNotifier, int>(
  _SelectedIndexNotifier.new,
);

class ModelList extends ConsumerStatefulWidget {
  const ModelList({super.key, required this.title});

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".

  final String title;

  @override
  ConsumerState<ModelList> createState() => _ModelList();
}

class _ModelList extends ConsumerState<ModelList> {
  late List<Widget> pages;

  @override
  void initState() {
    super.initState();
    pages = [Models(), DownloadedModels(), RangeTab(), Profile()];
  }

  @override
  Widget build(BuildContext context) {
    var selectedIndex = ref.watch(selectedIndexProvider);
    // This method is rerun every time setState is called, for instance as done
    // by the _incrementCounter method above.
    //
    // The Flutter framework has been optimized to make rerunning build methods
    // fast, so that you can just rebuild anything that needs updating rather
    // than having to individually change instances of widgets.
    return Scaffold(
      appBar: AppBar(
        title: Image.asset(
          'assets/images/logo_with_name.png',
          height: 60,
          color: Colors.white,
          colorBlendMode: BlendMode.srcIn,
        ),
        centerTitle: true,
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(1),
          child: Container(height: 1, color: const Color(0xFFE2E8F0)),
        ),
      ),
      body: Column(
        children: [
          Expanded(child: pages[selectedIndex]),
          Container(height: 1, color: const Color(0xFFE2E8F0)),
          NavigationBar(
            destinations: const [
              NavigationDestination(
                icon: Icon(Icons.home_outlined),
                selectedIcon: Icon(Icons.home),
                label: 'Home',
              ),
              NavigationDestination(
                icon: Icon(Icons.cloud_download_outlined),
                selectedIcon: Icon(Icons.cloud_download),
                label: 'My AI',
              ),
              NavigationDestination(
                icon: Icon(Icons.science_outlined),
                selectedIcon: Icon(Icons.science),
                label: 'Range',
              ),
              NavigationDestination(
                icon: Icon(Icons.person_outline),
                selectedIcon: Icon(Icons.person),
                label: 'Profile',
              ),
            ],
            selectedIndex: selectedIndex,
            onDestinationSelected: (int index) {
              ref.read(selectedIndexProvider.notifier).state = index;
            },
          ),
        ],
      ),
    );
  }
}
