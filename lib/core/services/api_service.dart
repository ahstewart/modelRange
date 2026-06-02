import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import '../config/app_config.dart';
import '../data_models/api_models.dart';
import '../data_models/inference_stat.dart';

class ApiService {
  static const String baseUrl = AppConfig.apiBaseUrl;

  /// Timeout applied to all external calls — metadata and file downloads.
  static const Duration _apiTimeout = Duration(seconds: 30);

  final http.Client _httpClient;

  ApiService({http.Client? httpClient})
    : _httpClient = httpClient ?? http.Client();

  // ==========================================
  // MODELS API
  // ==========================================

  /// Fetch all models, optionally filtered by task or supported status
  Future<List<MLModel>> getModels({
    int skip = 0,
    int limit = 1000,
    String? task,
    bool supportedOnly = false,
  }) async {
    try {
      final uri = Uri.parse('$baseUrl/models').replace(
        queryParameters: {
          'skip': skip.toString(),
          'limit': limit.toString(),
          if (task != null) 'task': task,
          if (supportedOnly) 'supported_only': 'true',
        },
      );

      final response = await _httpClient.get(uri).timeout(_apiTimeout);

      if (response.statusCode == 200) {
        final List<dynamic> jsonList = jsonDecode(response.body);
        return jsonList.map((json) => MLModel.fromJson(json)).toList();
      } else {
        throw Exception('Failed to load models: ${response.statusCode}');
      }
    } on SocketException {
      throw ApiConnectionException();
    } on TimeoutException {
      throw ApiTimeoutException();
    } catch (e) {
      debugPrint('Error in getModels: $e');
      rethrow;
    }
  }

  /// Fetch a specific model by ID
  Future<MLModel> getModel(String modelId) async {
    try {
      final response = await _httpClient
          .get(Uri.parse('$baseUrl/models/$modelId'))
          .timeout(_apiTimeout);

      if (response.statusCode == 200) {
        return MLModel.fromJson(jsonDecode(response.body));
      } else {
        throw Exception('Failed to load model: ${response.statusCode}');
      }
    } on SocketException {
      throw ApiConnectionException();
    } on TimeoutException {
      throw ApiTimeoutException();
    } catch (e) {
      debugPrint('Error in getModel: $e');
      rethrow;
    }
  }

  // ==========================================
  // MODEL VERSIONS API
  // ==========================================

  /// Fetch all versions for a specific model
  Future<List<ModelVersion>> getModelVersions(String modelId) async {
    try {
      final response = await _httpClient
          .get(Uri.parse('$baseUrl/models/$modelId/versions'))
          .timeout(_apiTimeout);

      if (response.statusCode == 200) {
        final List<dynamic> jsonList = jsonDecode(response.body);
        return jsonList.map((json) => ModelVersion.fromJson(json)).toList();
      } else {
        throw Exception('Failed to load versions: ${response.statusCode}');
      }
    } on SocketException {
      throw ApiConnectionException();
    } on TimeoutException {
      throw ApiTimeoutException();
    } catch (e) {
      debugPrint('Error in getModelVersions: $e');
      rethrow;
    }
  }

  /// Fetch a specific model version by ID
  Future<ModelVersion> getModelVersion(String versionId) async {
    try {
      final response = await _httpClient
          .get(Uri.parse('$baseUrl/versions/$versionId'))
          .timeout(_apiTimeout);

      if (response.statusCode == 200) {
        return ModelVersion.fromJson(jsonDecode(response.body));
      } else {
        throw Exception('Failed to load version: ${response.statusCode}');
      }
    } on SocketException {
      throw ApiConnectionException();
    } on TimeoutException {
      throw ApiTimeoutException();
    } catch (e) {
      debugPrint('Error in getModelVersion: $e');
      rethrow;
    }
  }

  // ==========================================
  // UTILITY METHODS
  // ==========================================

  /// Download file from URL and return bytes.
  Future<List<int>> downloadFile(String url) async {
    try {
      final response =
          await _httpClient.get(Uri.parse(url)).timeout(_apiTimeout);

      if (response.statusCode == 200) {
        return response.bodyBytes;
      } else {
        throw Exception('Failed to download file: ${response.statusCode}');
      }
    } on SocketException {
      throw ApiConnectionException();
    } on TimeoutException {
      throw ApiTimeoutException();
    } catch (e) {
      debugPrint('Error downloading file: $e');
      rethrow;
    }
  }

  /// Download a model asset through the backend's telemetry endpoint.
  /// The backend increments download counts then issues a 307 redirect to the
  /// storage provider (Hugging Face / S3). The http client follows the redirect
  /// transparently and returns the final binary bytes.
  Future<List<int>> downloadAsset(String versionId, String assetKey) async {
    try {
      final uri = Uri.parse('$baseUrl/versions/$versionId/download/$assetKey');
      final response = await _httpClient.get(uri);

      if (response.statusCode == 200) {
        return response.bodyBytes;
      } else {
        throw Exception(
          'Failed to download asset "$assetKey": ${response.statusCode}',
        );
      }
    } on SocketException {
      throw ApiConnectionException();
    } on TimeoutException {
      throw ApiTimeoutException();
    } catch (e) {
      debugPrint('Error downloading asset "$assetKey" for version $versionId: $e');
      rethrow;
    }
  }

  // ==========================================
  // TELEMETRY API
  // ==========================================

  /// Upload a batch of inference stats to the backend.
  /// Only called when the user is authenticated and has opted in.
  Future<void> uploadTelemetryBatch({
    required List<InferenceStat> stats,
    required String authToken,
  }) async {
    try {
      final response = await _httpClient
          .post(
            Uri.parse('$baseUrl/telemetry/batch'),
            headers: {
              'Content-Type': 'application/json',
              'Authorization': 'Bearer $authToken',
            },
            body: jsonEncode({'stats': stats.map((s) => s.toApiJson()).toList()}),
          )
          .timeout(_apiTimeout);
      if (response.statusCode != 200 && response.statusCode != 201) {
        throw Exception('Telemetry upload failed: ${response.statusCode}');
      }
    } on SocketException {
      throw ApiConnectionException();
    } on TimeoutException {
      throw ApiTimeoutException();
    } catch (e) {
      debugPrint('Error uploading telemetry: $e');
      rethrow;
    }
  }

  void dispose() {
    _httpClient.close();
  }
}

// ==========================================
// TYPED CONNECTION EXCEPTIONS
// ==========================================

/// Thrown when the device cannot reach the backend at all (no route to host,
/// connection refused, etc.).
class ApiConnectionException implements Exception {
  const ApiConnectionException();
}

/// Thrown when the backend is reachable but takes too long to respond.
class ApiTimeoutException implements Exception {
  const ApiTimeoutException();
}
