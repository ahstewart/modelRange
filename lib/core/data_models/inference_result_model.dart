import 'dart:convert'; // For jsonEncode/Decode if used for YamlMap conversion
import 'dart:typed_data';
import 'package:flutter/material.dart'; // For kDebugMode, Color
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/foundation.dart'; // For kDebugMode
import 'package:yaml/yaml.dart'; // For YAML parsing
import 'dart:developer' as developer; // For inspect

sealed class InferenceResult {}

class ClassificationResult extends InferenceResult {
  final List<Map<String, dynamic>> results; // e.g., [{'label': ..., 'confidence': ...}]
  ClassificationResult(this.results);
}

class DetectionResult extends InferenceResult {
  final List<Map<String, dynamic>> results; // e.g., [{'rect': ..., 'label': ..., 'confidence': ...}]
  DetectionResult(this.results);
}

class TextResult extends InferenceResult {
  final String text;
  TextResult(this.text);
}

class SegmentationMaskResult extends InferenceResult {
  // Could be List<List<int>>, Uint8List, img.Image, etc.
  final dynamic maskData;
  final int height;
  final int width;
  SegmentationMaskResult(this.maskData, this.height, this.width);
}