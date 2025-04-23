import 'dart:ffi';
import 'dart:io';
import 'dart:math' as Math;
import 'dart:typed_data';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:model_range/inference/pipeline_schema.txt';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';
import '../data_models/data_models.dart';
import 'package:yaml/yaml.dart';


// inference object that contains methods for loading models, pre and post processing, and running the actual inference
class InferenceObject {
  // initialize interpreter, pipeline (metadata), etc
  Interpreter? _interpreter;
  bool get isReady => _interpreter != null && modelPipeline != null;
  Pipeline? modelPipeline;

  List<String>? _labels;
  bool _isLoading = false;
  File? _selectedImage;
  List<dynamic>? _recognitions;

  final String modelPath;
  final String pipelinePath;

  InferenceObject({
    required this.modelPath,
    required this.pipelinePath,
  }) {
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
  
  // the inference object needs to handle the preprocessing, inference, and postprocessing phase
  // preprocess will take a raw input, and based on the pipeline YAML, perform the preprocessing, and return a
  // tensor ready for inference
  Future<dynamic> preprocess(dynamic rawInput, String _inputName) async {
    // first make sure model and pipeline are loaded
    if (!isReady) {
      if (kDebugMode) {
            debugPrint("Cannot preprocess input, model or pipeline have not been successfully loaded.");
          }
      return rawInput;
    }
    // then make sure pipeline includes preprocessing steps? if it doesn't, skip all of this and return rawInput
    if (modelPipeline!.preprocessing.isEmpty) {
      if (kDebugMode) {
            debugPrint("Pipeline is missing preprocessing block, returning raw input unchanged.");
          }
      return rawInput;
    }
    // if preprocessing steps are included, then match the preprocessing step with an input using input_name
    int? _inputIndex;
    int? _preprocessBlockIndex;

    // match _inputName to an input block
      for (var i; i < modelPipeline!.inputs.length; i++) {
        if (modelPipeline!.inputs[i].name == _inputName) {
          _inputIndex = i;
          break;
      }
    }

    // match _inputName to a preprocessing block
    for (var i; i < modelPipeline!.preprocessing.length; i++) {
      if (modelPipeline!.preprocessing[i].input_name == _inputName) {
        _preprocessBlockIndex = i;
        break;
      }
    }

    // check if the input could not be matched to an input block or preprocessing block
    if (_inputIndex == null || _preprocessBlockIndex == null) {
      if (kDebugMode) {
            debugPrint("Provided input name does not match input or preprocessing block in pipeline. Aborting preprocessing.");
          }
      return rawInput;
    }

    // once the preprocessing step and input are matched, use the expects_type to validate the rawInput
    final String _expectedType = modelPipeline!.preprocessing[_preprocessBlockIndex].expects_type;

    switch (_expectedType) {
      case 'image':
        if (rawInput is! img.Image) {
          if (kDebugMode) {
            debugPrint("Raw input type: $rawInput.rawInput.runtimeType.toString()");
            throw ArgumentError("This preprocessing block expects an image, but raw input is not an img.Image type. Aborting preprocessing.");
          }
          return rawInput;
        }
        if (kDebugMode) {
            debugPrint("Raw input type match expected type. Proceeding with preprocessing.");
            }
        break;
      case 'text':
        if (rawInput is! String) {
          if (kDebugMode) {
            debugPrint("Raw input type: $rawInput.rawInput.runtimeType.toString()");
            throw ArgumentError("This preprocessing block expects text, but raw input is not a String. Aborting preprocessing.");
          }
          return rawInput;
        }
        if (kDebugMode) {
            debugPrint("Raw input type match expected type. Proceeding with preprocessing.");
        }
        break;
      case 'audio':
        if (rawInput is! Uint8List) {
          if (kDebugMode) {
            debugPrint("Raw input type: $rawInput.rawInput.runtimeType.toString()");
            throw ArgumentError("This preprocessing block expects a audio, but raw input is not a Uint8List type. Aborting preprocessing.");
          }
          return rawInput;
        }
        if (kDebugMode) {
            debugPrint("Raw input type match expected type. Proceeding with preprocessing.");
        }
        break;
      // Add cases for other expected raw input types ('tensor', 'generic_list', etc.)
      default:
        throw UnimplementedError("Unsupported 'expects_type' in pipeline: $_expectedType");
      }

      // now that input is validated, start tracking the input with a variable
      dynamic currentInput;

      // start looping through the preprocessing steps
      for (var preStep in modelPipeline!.preprocessing[_preprocessBlockIndex].steps) {
        currentInput = await _performPreprocessingStep(currentInput, preStep, _preprocessBlockIndex);
      }

      // return final input tensor
      return currentInput;
  }


  // postprocess will take a map of all the outputted tensors and their associated output_names, and iterate through every
  Future<dynamic> postprocess(Map<String, dynamic> outputMap) async {
    // make sure pipeline includes postprocessing steps? if it doesn't, skip all of this and return rawInput
    if (modelPipeline!.postprocessing.isEmpty) {
      if (kDebugMode) {
            debugPrint("Pipeline is missing postprocessing blocks, returning raw output unchanged.");
          }
      return rawOutput;
    }
    // if postprocessing blocks are included, then match the step with an output using output_name
    int? _outputIndex;
    int? _postprocessBlockIndex;

    // match _inputName to an input block
      for (var i; i < modelPipeline!.outputs.length; i++) {
        if (modelPipeline!.outputs[i].name == _outputName) {
          _outputIndex = i;
          break;
      }
    }

    // match _inputName to a preprocessing block
    for (var i; i < modelPipeline!.postprocessing.length; i++) {
      if (modelPipeline!.postprocessing[i].input_name == _outputName) {
        _postprocessBlockIndex = i;
        break;
      }
    }

    // check if the input could not be matched to an input block or preprocessing block
    if (_outputIndex == null || _postprocessBlockIndex == null) {
      if (kDebugMode) {
            debugPrint("Provided input name does not match input or preprocessing block in pipeline. Aborting preprocessing.");
          }
      return rawInput;
    }

    // once the preprocessing step and input are matched, use the expects_type to validate the rawInput
    final String _expectedType = modelPipeline!.preprocessing[_preprocessBlockIndex].expects_type;

    switch (_expectedType) {
      case 'image':
        if (rawInput is! img.Image) {
          if (kDebugMode) {
            debugPrint("Raw input type: $rawInput.rawInput.runtimeType.toString()");
            throw ArgumentError("This preprocessing block expects an image, but raw input is not an img.Image type. Aborting preprocessing.");
          }
          return rawInput;
        }
        if (kDebugMode) {
            debugPrint("Raw input type match expected type. Proceeding with preprocessing.");
            }
        break;
      case 'text':
        if (rawInput is! String) {
          if (kDebugMode) {
            debugPrint("Raw input type: $rawInput.rawInput.runtimeType.toString()");
            throw ArgumentError("This preprocessing block expects text, but raw input is not a String. Aborting preprocessing.");
          }
          return rawInput;
        }
        if (kDebugMode) {
            debugPrint("Raw input type match expected type. Proceeding with preprocessing.");
        }
        break;
      case 'audio':
        if (rawInput is! Uint8List) {
          if (kDebugMode) {
            debugPrint("Raw input type: $rawInput.rawInput.runtimeType.toString()");
            throw ArgumentError("This preprocessing block expects a audio, but raw input is not a Uint8List type. Aborting preprocessing.");
          }
          return rawInput;
        }
        if (kDebugMode) {
            debugPrint("Raw input type match expected type. Proceeding with preprocessing.");
        }
        break;
      // Add cases for other expected raw input types ('tensor', 'generic_list', etc.)
      default:
        throw UnimplementedError("Unsupported 'expects_type' in pipeline: $_expectedType");
      }

      // now that input is validated, start tracking the input with a variable
      dynamic currentInput;

      // start looping through the preprocessing steps
      for (var preStep in modelPipeline!.preprocessing[_preprocessBlockIndex].steps) {
        currentInput = await _performPreprocessingStep(currentInput, preStep, _preprocessBlockIndex);
      }

      // return final input tensor
      return currentInput;
  }



  // execute a preprocessing step, given the step and the input data
  Future<dynamic> _performPreprocessingStep(dynamic inputData, ProcessingStep step, int preprocessingBlockIndex) async {
    if (kDebugMode) {
          debugPrint("Executing preprocessing step: $step.step");
    }

    switch (step.step) {
      // resizing an image, input is img.Image
      case 'resize_image':
        try {
          // resize image to sizes defined in the model metadata schema (MMS)
          img.Image resizedImage = img.copyResize(inputData, width: step.params['width'], height: step.params['height']);
          return resizedImage;
        }
        catch (e) {
          if (kDebugMode) {
            debugPrint("Error resizing image: $e");
            debugPrint("Returning input image with size unchanged.");
          }
          return inputData;
        }

      // normalize image, input is either img.Image or U8intList
      case 'normalize':
        String? method = step.params['method'];
        dynamic mean;
        dynamic stddev;
        dynamic value;
        String normalizeColorSpace = step.params['color_space'] ?? "RGB";

        try {
          // check the input type. if it's img.Image, convert to U8intList
          if (inputData == img.Image) {
            try {
                // Convert to RGB bytes
                inputData = imgToBytes(inputData, normalizeColorSpace);
                inputData = Uint8List.fromList(inputData);
              } 
            catch (e) {
                if (kDebugMode) {
                  debugPrint('Error converting image to bytes: $e');
                }
                throw Exception('Normalization error: Failed to convert image to bytes');
            }
          }
          var normBytes = Float32List(inputData.length);
          // first try the 'mean_stddev' method, which first normalizes the image pixel between 0 and 1 by dividing by 255,
          // then applies the mean and stddev normalization for each channel. This method requires the mean and stddev parameters 
          // to be lists with a length equal to the number of channels in the inputImage
          if (method == "mean_stddev") {
            for (var i = 0; i < inputData.length; i += 3) {
              normBytes[i] = ((inputData[i] / 255) - mean?[0]) / stddev?[0];
              normBytes[i++] = ((inputData[i++] / 255) - mean?[1]) / stddev?[1];
              normBytes[i++] = ((inputData[i++] / 255) - mean?[2]) / stddev?[2];
            }
            // final normImage = normBytes.reshape([1, inputImage.width, inputImage.height, inputImage.numChannels]);
            return normBytes;
          }
          // uniform normalization, instead of applying a per-channel norm, apply a singular mean and stddev value to all
          // pixel normalizations
          else if (method == "normalize_uniform") {
            for (var i = 0; i < inputData.length; i += 3) {
              normBytes[i] = (inputData[i] - mean?[0]) / stddev?[0];
              normBytes[i++] = (inputData[i++] - mean?[0]) / stddev?[0];
              normBytes[i++] = (inputData[i++] - mean?[0]) / stddev?[0];
            }
            // final normImage = normBytes.reshape([1, inputImage.width, inputImage.height, inputImage.numChannels]);
            return normBytes;
          }
          // uniform scaling - simply normalize all pixels to be between 0 and 1, based on a given scale_param (usually 255)
          else if (method == "normalize_uniform") {
            for (var i = 0; i < inputData.length; i += 3) {
              normBytes[i] = inputData[i] / value;
              normBytes[i++] = inputData[i++] / value;
              normBytes[i++] = inputData[i++] / value;
            }
            // final normImage = normBytes.reshape([1, inputImage.width, inputImage.height, inputImage.numChannels]);
            return normBytes;
          }
          else {
            if (kDebugMode) {
              debugPrint("Error normalizing image: Invalid 'method' parameter");
            }
          }
          return inputData; // return input image bytes if normalization didn't take place
        }
        catch (e) {
          if (kDebugMode) {
            debugPrint("Error normalizing image: $e");
            debugPrint("Returning input unchanged.");
          }
          return inputData;
        }
      
      // reformat preprocessed data to match input requirements
      case 'format':
        dynamic finalData;
        String targetDtype = step.params['target_dtype'];
        String inputColorSpace = step.params['color_space'];
        String dataLayout = step.params['data_layout'].toLowerCase();
        // first format data type and color space. supports 2 types of conversions: float32 and uint8
        // color space formatting is done with the imgToBytes helper method
        if (targetDtype == 'float32' && inputData is! Float32List) { 
          if (inputData is img.Image) { 
            var imageBytes = imgToBytes(inputData, inputColorSpace);
            finalData = Float32List.fromList(imageBytes.map((e) => e / 255.0).toList());
          } 
          else { 
            throw Exception("Cannot format unsupported type to float32"); 
          } 
        } 
        else if (targetDtype == 'uint8' && inputData is! Uint8List) { 
          if (inputData is img.Image) { 
            finalData = imgToBytes(inputData, inputColorSpace);
          } 
          else { 
            throw Exception("Cannot format unsupported type to uint8"); 
          } 
        }
        // then check data layout (ex: NHWC)
        // if target layout is NHWC, convert to NHWC if not already there, can only take NCHW as input
        if (dataLayout == 'nhwc' && !isNHWC(inputData.shape)) {
          if (isNCHW(inputData.shape)) {
            finalData = nchwToNhwc(inputData);
          }
          else {
            throw Exception("Only NCHW layouts can be converted to NHWC.");
          }
        }
        // if target layout is NCHW, convert to NCHW if not already there, can only take NHWC as input
        if (dataLayout == 'nchw' && !isNCHW(inputData.shape)) {
          if (isNHWC(inputData.shape)) {
            finalData = nhwcToNchw(inputData);
          }
          else {
            throw Exception("Only NHWC layouts can be converted to NCHW.");
          }
        }

        // lastly, check that the shape is identical to the shape parameter of the input object
        List<int>? targetInputShape;
        String preprocessingBlockInputName = modelPipeline!.preprocessing[preprocessingBlockIndex].input_name;
        for (var inputs in modelPipeline!.inputs) {
          if (inputs.name == preprocessingBlockInputName) {
            targetInputShape = inputs.shape;
            break;
          }
        }

        if (finalData.shape != targetInputShape) {
          if (kDebugMode) {
            debugPrint("Final formatted data shape $finalData.shape does not match target input shape $targetInputShape");
            debugPrint("Attempting to convert final data to target input shape");
          }

          try { 
            return finalData.reshape(targetInputShape!); 
          } 
          catch (e) { 
            if (kDebugMode) {
              debugPrint("Reshape error: $e. Input shape: ${finalData.shape}, Target shape: $targetInputShape");
            }
            return finalData;
          }
        }
      
        return finalData;

      default:
        if (kDebugMode) {
          debugPrint("Warning: Unsupported preprocessing step: ${step.step}"); 
        }
        return inputData;
    }
  }


  // perform inference given inputs and target output buffers
  // order of inferenceInputs will be mapped directly to the order of the pipeline inputs, so they must match
  // on the Flutter screen implementation
  // inferenceInputs is map keyed by the input name, so that the given input
  Future<void> _performInference(Map<String, dynamic> inferenceInputs, Pipeline pipeline) async {
    // check that the inputs provided match the inputs expected based on the pipeline file
    if (inferenceInputs.length != pipeline.inputs.length) {
      throw ArgumentError("Provided number of inputs (${inferenceInputs.length}) and expected number of inputs ($pipeline.inputs.length}) do not match, cannot proceed with inference.")
    }

    // preprocess inputs and construct the final input list for inference
    List<Object> processedInputs = [];
    for (var input in inferenceInputs.entries) {
      var tempInput = input.value;
      // check if a preprocessing step exists for the given input
      for (var preprocessBlock in pipeline.preprocessing) {
        if (input.key == preprocessBlock.input_name) {
          tempInput = await preprocess(input.value, input.key);
        }
        break;
      }
      // add input to the final processInputs list
      processedInputs.add(tempInput);
    }

    // create ouput buffers
    Map<int, Object> outputBuffers = {};
    List<IO> outputs = pipeline.outputs;
    for (int i = 0; i < outputs.length; i++) {
      outputBuffers[i] = _createOutputBuffer(outputs[i].shape, outputs[i].dtype);
    }

    if (kDebugMode) {
      debugPrint("Running inference on model.");
    }
    _interpreter?.runForMultipleInputs(processedInputs, outputBuffers);
  }


  // helper method to create an output buffer, given an output shape and data type
dynamic _createOutputBuffer(List<int> shape, String dtype) {
  int totalElements = shape.reduce((a, b) => a* b);
  switch (dtype.toLowerCase()) {
    case 'float32':
      return List.filled(totalElements, 0.0).reshape(shape);
    case 'uint8':
      return List.filled(totalElements, 0).reshape(shape);
    default:
      throw Exception("Unsupported output dtype: $dtype");
  }
}


  // perform a postprocessing step, given a postprocessing block step object
  Future<dynamic> _performPostprocessingStep(dynamic outputData, ProcessingStep step, )




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

}


// class to store a pipeline data object (an input or an output)
class PipelineIO {
  dynamic data;
  String dataType;
  List<int> shape;
  String? colorSpace;
  String? datalayout;

  PipelineIO({
    required this.data,
    required this.dataType,
    required this.shape,
    required this.colorSpace,
    required this.datalayout,
  });
}




/* 
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

} */