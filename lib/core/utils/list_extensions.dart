// Helper Extension for Reshaping Lists (Basic)
// Note: This is a basic reshape and assumes row-major order.
// For robust tensor operations, consider dedicated Dart math libraries if needed.
import 'dart:typed_data';

extension ReshapeList on List {
  List reshape(List<int> newShape) {
    if (newShape.isEmpty) {
      if (length == 1 && this[0] is List) return this[0]; // Handle scalar tensor case
      if (length == 0) return [];
      throw ArgumentError('New shape cannot be empty unless list contains a single element list (scalar tensor).');
    }

    final totalElements = newShape.fold<int>(1, (prev, el) => prev * el);

    if (length != totalElements) {
      throw ArgumentError(
          'Cannot reshape list of length $length into shape $newShape (requires $totalElements elements)');
    }

    if (newShape.length == 1) return List.from(this); // Return a copy for 1D

    if (newShape.length == 2) {
      final rows = newShape[0];
      final cols = newShape[1];
      List<List<dynamic>> reshaped = List.generate(
          rows, (_) => List.filled(cols, null, growable: false),
          growable: false);
      for (int i = 0; i < length; i++) {
        reshaped[i ~/ cols][i % cols] = this[i];
      }
      return reshaped;
    }
    if (newShape.length == 3) {
      final d1 = newShape[0];
      final d2 = newShape[1];
      final d3 = newShape[2];
      final stride2 = d3;
      final stride1 = d2 * d3;
      List<List<List<dynamic>>> reshaped = List.generate(
          d1,
          (_) => List.generate(
              d2, (_) => List.filled(d3, null, growable: false),
              growable: false),
          growable: false);
      for (int i = 0; i < length; i++) {
        reshaped[i ~/ stride1][(i % stride1) ~/ stride2][i % stride2] = this[i];
      }
      return reshaped;
    }
    if (newShape.length == 4) {
      final d1 = newShape[0]; // Batch
      final d2 = newShape[1]; // Height
      final d3 = newShape[2]; // Width
      final d4 = newShape[3]; // Channels
      final stride3 = d4;
      final stride2 = d3 * d4;
      final stride1 = d2 * d3 * d4;
      List<List<List<List<dynamic>>>> reshaped = List.generate(
          d1,
          (_) => List.generate(
              d2,
              (_) => List.generate(
                  d3, (_) => List.filled(d4, null, growable: false),
                  growable: false),
              growable: false),
          growable: false);
      for (int i = 0; i < length; i++) {
        reshaped[i ~/ stride1][(i % stride1) ~/ stride2]
            [(i % stride2) ~/ stride3][i % stride3] = this[i];
      }
      return reshaped;
    }
    // Add more dimensions if needed or throw error
    throw UnimplementedError(
        'Reshaping to ${newShape.length} dimensions not implemented in this basic helper.');
  }
}

extension ReshapeFloat32List on Float32List {
  List reshape(List<int> newShape) => this.toList().reshape(newShape);
}