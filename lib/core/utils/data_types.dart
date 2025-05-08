import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;


// helper function to check if list is flattened
bool isNestedList(List list) {
  return (list.isNotEmpty && list.first is List);
}

// helper method to check if a shape is NHWC
bool isNHWC(List<int> shape) {
  // NHWC format should have 4 dimensions [batch, height, width, channels]
  if (shape.length != 4) {
    return false;
  }
  
  // For typical image data:
  // - batch size is usually 1
  // - channels should be 1 (grayscale), 3 (RGB), or 4 (RGBA)
  // - height and width should be positive numbers
  final batch = shape[0];
  final height = shape[1];
  final width = shape[2];
  final channels = shape[3];
  
  if (channels != 1 && channels != 3 && channels != 4) {
    return false;
  }
  
  if (height <= 0 || width <= 0) {
    return false;
  }
  
  return true;
}

// helper method to check if a shape is NHWC
bool isNCHW(List<int> shape) {
  // NCHW format should have 4 dimensions [batch, channels, height, width]
  if (shape.length != 4) {
    return false;
  }
  
  // For typical image data:
  // - batch size is usually 1
  // - height and width should be positive numbers
  // - channels should be 1 (grayscale), 3 (RGB), or 4 (RGBA)
  final batch = shape[0];
  final channels = shape[1];
  final height = shape[2];
  final width = shape[3];

  
  if (channels != 1 && channels != 3 && channels != 4) {
    return false;
  }
  
  if (height <= 0 || width <= 0) {
    return false;
  }
  
  return true;
}

// helper method to convert image into bytes, defaults to rgb if colorSpace config is not supported
Uint8List imgToBytes(img.Image inputImage, [String colorSpace = 'RGB']) {
  switch (colorSpace.toLowerCase()) {
    case 'rgb':
      return inputImage.getBytes(order: img.ChannelOrder.rgb);
    case 'bgr':
      return inputImage.getBytes(order: img.ChannelOrder.bgr);
    case 'rgba':
      return inputImage.getBytes(order: img.ChannelOrder.rgba);
    case 'argb':
      return inputImage.getBytes(order: img.ChannelOrder.argb);
    default:
      return inputImage.getBytes(order: img.ChannelOrder.rgb);
  }
}

// helper method to convert NHWC to NCHW
dynamic nhwcToNchw(dynamic input) {
  if (kDebugMode) {
    debugPrint("Converting image layout from NHWC to NCHW");
  }
  final int height = input.shape[1];
  final int width = input.shape[2];
  final int channels = input.shape[3];
  final int batchSize = input.length ~/ (height * width * channels);
  dynamic output;
  if (input is Float32List) {
    output = Float32List(input.length);
  }
  else if (input is Uint8List) {
    output = Uint8List(input.length);
  }
  else {
    if (kDebugMode) {
      debugPrint("input type: $input.runtimeType()");
    }
    throw ArgumentError("Cannot convert NHWC to NCHW, input must be Float32List or Uint8List.");
  }
  for (int b = 0; b < batchSize; b++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int c = 0; c < channels; c++) {
          // Convert from NHWC [b, h, w, c] to NCHW [b, c, h, w]
          final nhwcIndex = b * height * width * channels + 
                          h * width * channels + 
                          w * channels + 
                          c;
          
          final nchwIndex = b * channels * height * width + 
                          c * height * width + 
                          h * width + 
                          w;
                          
          output[nchwIndex] = input[nhwcIndex];
        }
      }
    }
  }
  return output;
}

// helper method to convert NCHW to NHWC
dynamic nchwToNhwc(dynamic input) {
  if (kDebugMode) {
    debugPrint("Converting image layout from NCHW to NHWC");
  }
  
  final int channels = input.shape[1];
  final int height = input.shape[2];
  final int width = input.shape[3];
  final int batchSize = input.length ~/ (height * width * channels);
  
  dynamic output;
  if (input is Float32List) {
    output = Float32List(input.length);
  }
  else if (input is Uint8List) {
    output = Uint8List(input.length);
  }
  else {
    if (kDebugMode) {
      debugPrint("input type: ${input.runtimeType}");
    }
    throw ArgumentError("Cannot convert NCHW to NHWC, input must be Float32List or Uint8List.");
  }

  for (int b = 0; b < batchSize; b++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int c = 0; c < channels; c++) {
          // Convert from NCHW [b, c, h, w] to NHWC [b, h, w, c]
          final nchwIndex = b * channels * height * width + 
                          c * height * width + 
                          h * width + 
                          w;
          
          final nhwcIndex = b * height * width * channels + 
                          h * width * channels + 
                          w * channels + 
                          c;
                          
          output[nhwcIndex] = input[nchwIndex];
        }
      }
    }
  }
  return output;
}