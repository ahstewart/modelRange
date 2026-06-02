import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../data_models/inference_stat.dart';
import '../services/stats_service.dart';
import '../services/telemetry_service.dart';
import '../services/api_service.dart';
import 'auth_providers.dart';

// ---------------------------------------------------------------------------
// StatsService — singleton for the app lifetime
// ---------------------------------------------------------------------------
final statsServiceProvider = Provider<StatsService>((ref) {
  final svc = StatsService();
  ref.onDispose(svc.close);
  return svc;
});

// ---------------------------------------------------------------------------
// Telemetry opt-in setting (SharedPreferences, default = false)
// ---------------------------------------------------------------------------
class TelemetryOptInNotifier extends Notifier<bool> {
  static const _key = 'telemetry_opt_in';

  @override
  bool build() {
    Future.microtask(_load);
    return false;
  }

  Future<void> _load() async {
    final prefs = await SharedPreferences.getInstance();
    state = prefs.getBool(_key) ?? false;
  }

  Future<void> setValue(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_key, value);
    state = value;
  }
}

final telemetryOptInProvider = NotifierProvider<TelemetryOptInNotifier, bool>(
  TelemetryOptInNotifier.new,
);

// ---------------------------------------------------------------------------
// Per-version stat summary (aggregate)
// ---------------------------------------------------------------------------
final versionStatSummaryProvider =
    FutureProvider.family<InferenceStatSummary?, String>((ref, versionId) {
  final service = ref.watch(statsServiceProvider);
  return service.getSummaryForVersion(versionId);
});

// ---------------------------------------------------------------------------
// Per-version recent runs (individual records, newest first)
// ---------------------------------------------------------------------------
final versionStatsProvider =
    FutureProvider.family<List<InferenceStat>, String>((ref, versionId) {
  final service = ref.watch(statsServiceProvider);
  return service.getStatsForVersion(versionId);
});

// ---------------------------------------------------------------------------
// TelemetryService
// ---------------------------------------------------------------------------
final telemetryServiceProvider = Provider<TelemetryService>((ref) {
  return TelemetryService(
    statsService: ref.watch(statsServiceProvider),
    apiService: ApiService(),
  );
});

// ---------------------------------------------------------------------------
// Convenience: sync telemetry using current opt-in state and auth token
// ---------------------------------------------------------------------------
Future<void> syncTelemetry(Ref ref) {
  final svc = ref.read(telemetryServiceProvider);
  final optedIn = ref.read(telemetryOptInProvider);
  final token = ref.read(authTokenProvider);
  return svc.syncIfEligible(optedIn: optedIn, authToken: token);
}
