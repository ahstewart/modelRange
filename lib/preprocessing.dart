import 'dart:ffi';
import 'dart:io';
import 'dart:math' as Math;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';
import 'data_models.dart';
import 'package:yaml/yaml.dart';


// inference object that contains methods for loading models, pre and post processing, and running the actual inference
class InferenceObject {
  final String modelPath;
  final String pipelinePath;

  // initialize interpreter, pipeline (metadata), etc
  Interpreter? _interpreter;
  Pipeline? modelPipeline;
  List<String>? _labels;
  bool _isLoading = false;
  File? _selectedImage;
  List<dynamic>? _recognitions;

  const InferenceObject {
    loadModel(modelPath);
    loadPipeline(pipelinePath);
  }

  // load model from model_path (AKA create an interpreter)
  Future<void> loadModel(String modelPath, {String modelFramework="tflite"}) async {
    if (modelFramework == "tflite") {
      final options = InterpreterOptions();

      try {
        _interpreter = await Interpreter.fromAsset(modelPath, options:options);

        if (kDebugMode) {
          debugPrint(_interpreter?.getInputTensors().toString());
          debugPrint(_interpreter?.getOutputTensors().toString());
        }
      }
      catch (e) {
        if (kDebugMode) {
          debugPrint("Failed to load model: $e");
        }
      }
    }
  }

  // load pipeline from a pipeline_path, using the pipeline data model
  Future<void> loadPipeline(String pipelinePath) async {
    // get string from pipeline file
    String pipelineContents = await File(pipelinePath).readAsString();
    // parse the string using the yaml package and return the parsed map
    YamlMap pipelineYamlMap = loadYaml(pipelineContents);
    // convert YAML map to Map<String, 
    Map<String, dynamic> pipelineMap = _convertYamlToJson(pipelineYamlMap);

    // create pipeline object from pipeline_map
    modelPipeline = Pipeline.fromJson(pipelineMap);
  }

  // helper method for converting YAML map to JSON
  Map<String, dynamic> _convertYamlToJson(YamlMap yaml) {
      return yaml.map((key, value) => MapEntry(
        key.toString(), 
        value is YamlMap ? _convertYamlToJson(value) : value,
      ));
  }
  


}


class imagePreprocessing {
  final String step;
  const imagePreprocessing({required this.step,});

  preprocessImage(img.Image image, Map metadata) {
    var preprocessingPhases = metadata['preprocessing'];
    int phase_count = 0;
    for (var phase in preprocessingPhases) {
      int step_count = 0;
      for (var step in phase['steps']) {
        if (step['step'] == 'resize_image') {
            image = resizeImage(image, step['params']['width'], step['params']['height']);
        }
        else if (step['step'] == 'normalize') {
          var imageBytes = image.getBytes(order: img.ChannelOrder.rgb);
          imageBytes = normalizeImage(imageBytes, step['params']['method']);
        }
        else if (step['step'] == 'format') {
          var imageBytes = image.getBytes(order: img.ChannelOrder.rgb);
          normalizeImage(imageBytes, step['params']['method']);
        }
      }
    }

  }


  // method to resize an image
  // takes a decoded image as input
  img.Image resizeImage(img.Image inputImage, int inputWidth, int inputHeight) {
    try {
      // resize image to sizes defined in the model metadata schema (MMS)
      img.Image resizedImage = img.copyResize(inputImage, width: inputWidth, height: inputHeight);
      return resizedImage;
    }
    catch (e) {
      if (kDebugMode) {
        debugPrint("Error resizing image: $e");
        debugPrint("Returning input image with size unchanged.");
      }
      return inputImage;
    }
  }

  // method to normalize an image
  // takes a decoded image represented as bytes as input
  // valid normalization methods are "mean_stddev", "normalize_uniform", "scale_uniform"
  // outputs a normalized float 32 list
  List<dynamic> normalizeImage(List<dynamic> inputImageBytes, String method, 
                                {List<double>? mean = const [0.485, 0.456, 0.406], 
                                List<double>? stddev = const [0.229, 0.224, 0.225], 
                                double? scaleParam = 255.0}) {
    try {
      // var imageBytes = inputImage.getBytes(order: img.ChannelOrder.rgb);
      var normBytes = Float32List(inputImageBytes.length);
      // first try the 'mean_stddev' method, which first normalizes the image pixel between 0 and 1 by dividing by 255,
      // then applies the mean and stddev normalization for each channel. This method requires the mean and stddev parameters 
      // to be lists with a length equal to the number of channels in the inputImage
      if (method == "mean_stddev") {
        for (var i = 0; i < inputImageBytes.length; i += 3) {
          normBytes[i] = ((inputImageBytes[i] / 255) - mean?[0]) / stddev?[0];
          normBytes[i++] = ((inputImageBytes[i++] / 255) - mean?[1]) / stddev?[1];
          normBytes[i++] = ((inputImageBytes[i++] / 255) - mean?[2]) / stddev?[2];
        }
        // final normImage = normBytes.reshape([1, inputImage.width, inputImage.height, inputImage.numChannels]);
        return normBytes;
      }
      // uniform normalization, instead of applying a per-channel norm, apply a singular mean and stddev value to all
      // pixel normalizations
      else if (method == "normalize_uniform") {
        for (var i = 0; i < inputImageBytes.length; i += 3) {
          normBytes[i] = (inputImageBytes[i] - mean?[0]) / stddev?[0];
          normBytes[i++] = (inputImageBytes[i++] - mean?[0]) / stddev?[0];
          normBytes[i++] = (inputImageBytes[i++] - mean?[0]) / stddev?[0];
        }
        // final normImage = normBytes.reshape([1, inputImage.width, inputImage.height, inputImage.numChannels]);
        return normBytes;
      }
      // uniform scaling - simply normalize all pixels to be between 0 and 1, based on a given scale_param (usually 255)
      else if (method == "normalize_uniform") {
        for (var i = 0; i < inputImageBytes.length; i += 3) {
          normBytes[i] = inputImageBytes[i] / scaleParam;
          normBytes[i++] = inputImageBytes[i++] / scaleParam;
          normBytes[i++] = inputImageBytes[i++] / scaleParam;
        }
        // final normImage = normBytes.reshape([1, inputImage.width, inputImage.height, inputImage.numChannels]);
        return normBytes;
      }
      else {
        if (kDebugMode) {
          debugPrint("Error normalizing image: Invalid 'method' parameter");
        }
      }
      return inputImageBytes; // return input image bytes if normalization didn't take place
    }
    catch (e) {
      if (kDebugMode) {
        debugPrint("Error resizing image: $e");
        debugPrint("Returning input image with size unchanged.");
      }
      return inputImageBytes;
    }
  }


  /// Converts a Float32List from NHWC format to NCHW format
  /// Parameters:
  ///   input: Float32List in NHWC format
  ///   height: height of the image
  ///   width: width of the image
  ///   channels: number of channels (typically 3 for RGB)
  Float32List nhwcToNchw(Float32List input, int height, int width, int channels) {
    final int batchSize = input.length ~/ (height * width * channels);
    var output = Float32List(input.length);
    
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


  // format image to 

}