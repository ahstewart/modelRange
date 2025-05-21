import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';
import '../data_models/pipeline.dart';
import 'package:yaml/yaml.dart';
import '../utils/list_extensions.dart';
import '../utils/data_types.dart';
import '../utils/math.dart';
import 'dart:developer' as developer;


// inference object that contains methods for loading models, pre and post processing, and running the actual inference
class InferenceObject {
  // initialize interpreter, pipeline (metadata), etc
  Interpreter? _interpreter;
  Pipeline? modelPipeline;

  List<String>? _labels;
  bool get isReady => _interpreter != null && modelPipeline != null;

  dynamic scoreTensor;

  final String modelPath;
  final String pipelinePath;

  InferenceObject({
    required this.modelPath,
    required this.pipelinePath,
  }) 
  
  {
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
    if (kDebugMode) {
      debugPrint("Loading model pipeline file...");
    }

    // get string from pipeline file
    String pipelineContents = await rootBundle.loadString(pipelinePath);
    // parse the string using the yaml package and return the parsed map
    YamlMap pipelineYamlMap = loadYaml(pipelineContents);
    // convert YAML map to Map<String, 
    Map<String, dynamic> pipelineMap = _convertYamlToJson(pipelineYamlMap);

    // create pipeline object from pipeline_map
    modelPipeline = Pipeline.fromJson(pipelineMap);

    if (kDebugMode) {
      debugPrint("Model pipeline file loaded successfully.");
    }
  }

  // helper method for converting YAML map to JSON
  dynamic _convertYamlToJson(dynamic yaml) {
    if (yaml is YamlMap) {
      return Map<String, dynamic>.fromEntries(
        yaml.entries.map(
          (e) => MapEntry(e.key.toString(), _convertYamlToJson(e.value)),
        ),
      );
    }
    if (yaml is YamlList) {
      return yaml.map(_convertYamlToJson).toList();
    }
    return yaml;
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

    debugPrint("Starting preprocessing...");
    // if preprocessing steps are included, then match the preprocessing step with an input using input_name
    int? inputIndex;
    int? preprocessBlockIndex;

    debugPrint("Matching given input name to input name in pipeline file.");
    // match _inputName to an input block
      for (var i = 0; i < modelPipeline!.inputs.length; i++) {
        if (modelPipeline!.inputs[i].name == _inputName) {
          inputIndex = i;
          break;
      }
    }

    debugPrint("Matching input name to a preprocessing block in pipeline file.");
    // match _inputName to a preprocessing block
    for (var i = 0; i < modelPipeline!.preprocessing.length; i++) {
      if (modelPipeline!.preprocessing[i].input_name == _inputName) {
        preprocessBlockIndex = i;
        break;
      }
    }

    // check if the input could not be matched to an input block or preprocessing block
    if (inputIndex == null || preprocessBlockIndex == null) {
      if (kDebugMode) {
            debugPrint("Provided input name does not match input or preprocessing block in pipeline. Aborting preprocessing.");
          }
      return rawInput;
    }

    // once the preprocessing step and input are matched, use the expects_type to validate the rawInput
    final String expectedType = modelPipeline!.preprocessing[preprocessBlockIndex].expects_type;

    debugPrint("Validating that the raw input matches the 'expects_type' parameter in the pipeline file...");
    switch (expectedType) {
      case 'image':
        if (rawInput is! img.Image) {
          if (kDebugMode) {
            debugPrint("Raw input type: ${rawInput.runtimeType.toString()}");
            throw ArgumentError("This preprocessing block expects an image, but raw input is not an img.Image type. Aborting preprocessing.");
          }
          return rawInput;
        }
        if (kDebugMode) {
            debugPrint("Raw input type matches expected type. Proceeding with preprocessing.");
            }
        break;
      case 'text':
        if (rawInput is! String) {
          if (kDebugMode) {
            debugPrint("Raw input type: ${rawInput.runtimeType.toString()}");
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
            debugPrint("Raw input type: ${rawInput.runtimeType.toString()}");
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
        throw UnimplementedError("Unsupported 'expects_type' in pipeline: $expectedType");
      }
      debugPrint("Raw input type validated.");

      // now that input is validated, start tracking the input with a variable
      // initialize it with the rawInput
      dynamic currentInput = rawInput;

      debugPrint("Executing steps for preprocessing block ${modelPipeline!.preprocessing[preprocessBlockIndex].input_name}...");
      // start looping through the preprocessing steps
      for (var preStep in modelPipeline!.preprocessing[preprocessBlockIndex].steps) {
        currentInput = await _performPreprocessingStep(currentInput, preStep, preprocessBlockIndex);
        debugPrint("Preprocessing step ${preStep.step} completed successfully...");
      }

      debugPrint("Preprocessing complete.");

      // return final input tensor
      return currentInput;
  }

  // postprocess complete a postprocessing block and return a Map which will be the final inference result
  // the input to postprocess is the postprocessing block, the raw outputs map (source tensors), and the final result map
  Future<dynamic> postprocess(Map<String, dynamic> rawOutputs, int postprocessBlockIndex) async {
    debugPrint("Starting postprocessing...");

    // declare some variables to make things easier
    ProcessingBlock block = modelPipeline!.postprocessing[postprocessBlockIndex];
    String outputName = block.output_name;
    // declare a variable to track the current output
    dynamic currentResult = rawOutputs[block.source_tensors[0]];
    
    debugPrint("Checking if the postprocessing block contains steps...");
    // make sure pipeline includes postprocessing steps? if it doesn't, skip all of this and return rawInput
    if (modelPipeline!.postprocessing.isEmpty) {
      if (kDebugMode) {
            debugPrint("Pipeline is missing postprocessing blocks, returning raw output unchanged.");
          }
      return currentResult;
    }

    debugPrint("Postprocessing block contained steps.");

    debugPrint("Checking if source tensors are present in all postprocessing blocks...");
    // check that all source tensors in the postprocessing block are present in the output map
    List<String> sourceTensors = block.source_tensors;
    for (var tensor in sourceTensors) {
      if (!rawOutputs.containsKey(tensor)) {
        if (kDebugMode) {
            debugPrint("Output map does not contain source tensor: $tensor. Aborting postprocessing and returning raw output unchanged.");
          }
        return currentResult;
      }
    }

    debugPrint("Source tensors present in all postprocessing blocks.");

    debugPrint("Executing steps for ${block.output_name}...");
    // start looping through the postprocessing steps
    for (var postStep in block.steps) {
      currentResult = await _performPostprocessingStep(currentResult, rawOutputs, postStep);
      if (currentResult == null && postStep != block.steps.last) {
        throw Exception("Postprocessing block '$outputName', step '${postStep.step}' failed or returned null unexpectedly.");
      }
      debugPrint("Postprocessing step ${postStep.step} completed successfully...");
    }
    
    if (kDebugMode) {
      debugPrint("Postprocessing complete.");
      developer.log("Inspecting postprocessing result for block ${modelPipeline!.postprocessing[postprocessBlockIndex].output_name}");
      developer.inspect(currentResult);
    }


    debugPrint("Adding result from postprocess block ${modelPipeline!.postprocessing[postprocessBlockIndex].output_name} to the final results map.");

    // return final output map
    return currentResult;
  }


  // execute a preprocessing step, given the step and the input data
  Future<dynamic> _performPreprocessingStep(dynamic inputData, ProcessingStep step, int preprocessingBlockIndex) async {
    if (kDebugMode) {
          debugPrint("Executing preprocessing step: ${step.step}");
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
        debugPrint("Normalizing image...");
        String? method = step.params['method'];
        String normalizeColorSpace = step.params['color_space'] ?? "RGB";

        try {
          debugPrint("Checking input type. Normalization requires U8intList.");
          // check the input type. if it's img.Image, convert to U8intList
          if (inputData.runtimeType == img.Image) {
            try {
              debugPrint("Input data is img.Image, converting to Uint8List...");
              // Convert to RGB bytes
              inputData = imgToBytes(inputData, normalizeColorSpace);
              inputData = Uint8List.fromList(inputData);
              debugPrint("Converted image to Uint8List.");
            } 
            catch (e) {
                if (kDebugMode) {
                  debugPrint('Error converting image to bytes: $e');
                }
                throw Exception('Normalization error: Failed to convert image to bytes');
            }
          }
          else {
            debugPrint("Input data type is ${inputData.runtimeType}");
          }
          var normBytes = Float32List(inputData.length);
          // first try the 'mean_stddev' method, which first normalizes the image pixel between 0 and 1 by dividing by 255,
          // then applies the mean and stddev normalization for each channel. This method requires the mean and stddev parameters 
          // to be lists with a length equal to the number of channels in the inputImage
          debugPrint("Executing normalization using the $method method...");
          if (method == "mean_stddev") {
            List mean = step.params['mean'] ?? List.filled(normalizeColorSpace.length, 0.456);
            List stddev = step.params['stddev'] ?? List.filled(normalizeColorSpace.length, 0.224);
            for (var i = 0; i < inputData.length; i += 3) {
              normBytes[i] = ((inputData[i] / 255) - mean[0]) / stddev[0];
              normBytes[i++] = ((inputData[i++] / 255) - mean[1]) / stddev[1];
              normBytes[i++] = ((inputData[i++] / 255) - mean[2]) / stddev[2];
            }
            // final normImage = normBytes.reshape([1, inputImage.width, inputImage.height, inputImage.numChannels]);
            return normBytes;
          }
          // uniform normalization, instead of applying a per-channel norm, apply a singular mean and stddev value to all
          // pixel normalizations
          else if (method == "normalize_uniform") {
            var mean = step.params['mean'] ?? 127.5;
            var stddev = step.params['stddev'] ?? 127.5;
            for (var i = 0; i < inputData.length; i += 3) {
              normBytes[i] = (inputData[i] - mean) / stddev;
              normBytes[i++] = (inputData[i++] - mean) / stddev;
              normBytes[i++] = (inputData[i++] - mean) / stddev;
            }
            // final normImage = normBytes.reshape([1, inputImage.width, inputImage.height, inputImage.numChannels]);
            return normBytes;
          }
          // uniform scaling - simply normalize all pixels to be between 0 and 1, based on a given scale_param (usually 255)
          else if (method == "scale_div") {
            var value = step.params['value'] ?? 255.0;
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
          debugPrint("Normalized image successfully.");
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
        dynamic finalData = inputData;
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

        // get finalData shape for further processing
        List<int> finalData_shape;
        debugPrint("finalData runtime type = ${finalData.runtimeType}");
        if (finalData is Float32List || finalData is Uint8List) {
          finalData_shape = [1, finalData.length];
        }
        else {
          finalData_shape = finalData.shape;
        }

        /*
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
        */

        // lastly, check that the shape is identical to the shape parameter of the input object
        List<int> targetInputShape = [];
        String preprocessingBlockInputName = modelPipeline!.preprocessing[preprocessingBlockIndex].input_name;
        for (var inputs in modelPipeline!.inputs) {
          if (inputs.name == preprocessingBlockInputName) {
            targetInputShape = inputs.shape;
            break;
          }
        }

        if (finalData_shape != targetInputShape) {
          if (kDebugMode) {
            debugPrint("Final formatted data shape $finalData_shape does not match target input shape $targetInputShape");
            debugPrint("Attempting to convert final data to target input shape");
          }

          try { 
            switch (targetDtype.toLowerCase()) {
              case 'float32':
                return (finalData as Float32List).reshape(targetInputShape);
              //case 'uint8':
              //  return (finalData as Uint8List).reshape(targetInputShape);
            }
          }
          catch (e) { 
            if (kDebugMode) {
              debugPrint("Reshape error: $e. Input shape: $finalData_shape, Target shape: $targetInputShape");
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
  Future<dynamic> performInference(Map<String, dynamic> inferenceInputs) async {
    // check that the inputs provided match the inputs expected based on the pipeline file
    if (inferenceInputs.length != modelPipeline!.inputs.length) {
      throw ArgumentError("Provided number of inputs (${inferenceInputs.length}) and expected number of inputs ($modelPipeline!.inputs.length}) do not match, cannot proceed with inference.");
    }

    // preprocess inputs and construct the final input list for inference
    List<Object> processedInputs = [];
    for (var input in inferenceInputs.entries) {
      var tempInput = input.value;
      // check if a preprocessing step exists for the given input
      for (var preprocessBlock in modelPipeline!.preprocessing) {
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
    List<IO> outputs = modelPipeline!.outputs;
    for (int i = 0; i < outputs.length; i++) {
      outputBuffers[i] = _createOutputBuffer(outputs[i].shape, outputs[i].dtype);
    }

    if (kDebugMode) {
      debugPrint("Running inference on model.");
    }
    _interpreter?.runForMultipleInputs(processedInputs, outputBuffers);
    debugPrint("Inference completed.");
    if (kDebugMode) {
      developer.log("Inspecting outputBuffers variable from the TFLite inference.");
      developer.inspect(outputBuffers);
    }

    // convert outputBuffers map to a String-keyed map using the output tensor names
    Map<String, dynamic> inferenceOutputs = {};
    for (int i=0; i < outputBuffers.length; i++) {
      inferenceOutputs[modelPipeline!.outputs[i].name] = outputBuffers[i];
      }

    // define the final results map, which will contain the final output from the model inference
    // the map is keyed by each postprocessing block's name and final output
    Map<String, dynamic> finalResults = {};

    // check if any postprocessing blocks exist
    if (modelPipeline!.postprocessing.isNotEmpty) {
      // loop through the postprocessing blocks and run the postprocess method
      for (int i = 0; i < modelPipeline!.postprocessing.length; i++) {
        String blockName = modelPipeline!.postprocessing[i].output_name;
        finalResults[blockName] = await postprocess(inferenceOutputs, i);
        if (kDebugMode) {
          debugPrint("Postprocessing block ${modelPipeline!.postprocessing[i].output_name} completed.");
        }
      }
    }
    else {
      if (kDebugMode) {
        debugPrint("No postprocessing blocks found in pipeline, returning raw output.");
      }
      finalResults = inferenceOutputs;
    }

    if (kDebugMode) {
      developer.log("Postprocessing complete, inspecting the finalResults output...");
      developer.inspect(finalResults);
    }
    
    // return the final results map
    return finalResults;
  }


  // method to dispose of the inference objects from memory
  void dispose() {
    _interpreter?.close(); 
    _interpreter = null; 
    modelPipeline = null;
    if (kDebugMode) {
      debugPrint("Inference Object disposed.");
    }
  }


  // helper method to create an output buffer, given an output shape and data type
  dynamic _createOutputBuffer(List<int> shape, String dtype) {
    int totalElements = shape.reduce((a, b) => a* b);
    switch (dtype.toLowerCase()) {
      case 'float32':
        return ListShape(List.filled(totalElements, 0.0)).reshape(shape);
      case 'uint8':
        return ListShape(List.filled(totalElements, 0)).reshape(shape);
      default:
        throw Exception("Unsupported output dtype: $dtype");
    }
  }


  // perform a postprocessing step, given a postprocessing block step object
  Future<dynamic> _performPostprocessingStep(dynamic processedOutput, Map<String, dynamic> outputTensors, ProcessingStep step) async {
    if (kDebugMode) {
      debugPrint("Executing postprocessing step: ${step.step}");
    }

    switch (step.step) {
      // applies an activation function to a list of values
      case 'apply_activation':
        // check that the current processed data is a List
        if (processedOutput is! List) {
          throw FormatException("Processed output is not a List, cannot apply activation function.");
        }
        // run activation function on input data based on function name
        String function = step.params["function"];
        switch (function) {
          case 'softmax':
            processedOutput = applySoftmax(processedOutput);
          //case 'sigmoid':
          //  return processedOutput.map((x) => 1 / (1 + Math.exp(-x))).toList();
          //case 'relu':
          //  return processedOutput.map((x) => x < 0 ? 0 : x).toList();
          default:
            if (kDebugMode) {
              debugPrint("Warning: Unsupported activation function: $function, returning input data unchanged."); 
            }
        }

        if (kDebugMode) {
          debugPrint("Finished ${step.step} postprocessing step.");
          developer.log("Inspecting ${step.step} postprocessing output:n processedOutput");
          developer.inspect(processedOutput);
        }

      // map raw or activated outputs to a set of labels for classification
      case 'map_labels':
        // load labels into memory
        try {
          final classificationLabels = await rootBundle.loadString(step.params['labels_url']);
          _labels = classificationLabels.split('\n').map((label) => label.trim()).where((label) => label.isNotEmpty).toList();
          if (kDebugMode) {
            debugPrint("Successfully loaded ${_labels?.length} labels.");
          }
        }
        catch (e) {
          throw Exception("Failed fetching classification labels from $step.params['labels_url']: $e");
        }

        // check whether task is classification or object detection
        // image classification map_labels takes a List of floats as input
        // object detection map_labels takes a Map<int, List<Map<String, dynamic>>>
        
        if (processedOutput is Map) {
          debugPrint("Mapping labels assuming object detection task.");
          
          // grabbing detection class index tensor for label mapping
          String detectionClassTensorName = step.params['class_tensor'];
          dynamic detectionClassTensor = outputTensors[detectionClassTensorName];
          if (kDebugMode) {
            developer.log("Inspecting detection class tensor...");
            developer.inspect(detectionClassTensor);
          }

          // loop through detection batches
          for (int i = 0; i < processedOutput.length; i++) {
            int detectionCount = 1;
            // loop through detections
            for (var detectionMap in processedOutput[i]!) {
              debugPrint("Mapping label ${_labels?[detectionMap['original_index']]} to detection $detectionCount");
              detectionMap['label'] = _labels?[detectionClassTensor[0][detectionMap['original_index']].toInt()];
              detectionCount++;
            }
          }
        }
        else if (processedOutput is List) {
          debugPrint("Mapping labels assuming image classification task");
          // create recognitions, which is a list of labels mapped to a value in the raw output tensor
          List<Map<String, dynamic>> recognitions = [];
          // declare tempOutput
          List<dynamic> tempOutput = [];
          // grabbing classification class index tensor for label mapping
          String classTensorName = step.params['class_tensor'];
          dynamic classTensor = outputTensors[classTensorName];
          if (kDebugMode) {
            developer.log("Inspecting classification class tensor...");
            developer.inspect(classTensor);
          }
          // check if processedOutput is a nested list
          debugPrint("Checking if processedOutput type = ${processedOutput.runtimeType} is a nested list.");
          if (isNestedList(processedOutput)) {
            debugPrint("Flattening processedOutput nested List.");
            List<dynamic> flattenedProcessedOutput = processedOutput.expand((x) => x).toList();
            tempOutput = flattenedProcessedOutput;
          }
          else {
            tempOutput = processedOutput;
          }
          debugPrint("Map label debug message: tempOutput type = ${tempOutput.runtimeType}");
          for (int i=0; i<tempOutput.length; i++) {
            recognitions.add({
              "index": i,
              "label": _labels![i],
              "confidence": tempOutput[i],
            });
          }
          // set the processed output to the recognition list
          processedOutput = recognitions;
        // filters object detections by some threshold
        // expects a tensor, or a List<dynamic> in Dart
        }
        if (kDebugMode) {
          debugPrint("Finished ${step.step} postprocessing step.");
          developer.log("Inspecting ${step.step} postprocessing output: processedOutput");
          developer.inspect(processedOutput);
        }
       
      // filters object detections by some threshold
      // expects a tensor, or a List<dynamic> in Dart
      case 'filter_by_score':
/*         // check that the processed output is a list of floats
        debugPrint("Validating that 'filter_by_score' input is a List");
        if (processedOutput is! List) {
          try {
            debugPrint("Input to 'map_label' step isn't a List, trying to convert to List");
            processedOutput = processedOutput.toList();
          }
          catch (e) {
            throw FormatException("Processed output is not a List and cannot be converted to a List. Cannot map to classification labels.");
          }
        }
        // declare tempOutput
        List<dynamic> tempOutput = [];
        // check if processedOutput is a nested list
        debugPrint("Checking if processedOutput type = ${processedOutput.runtimeType} is a nested list.");
        if (isNestedList(processedOutput)) {
          debugPrint("Flattening processedOutput nested List.");
          List<dynamic> flattenedProcessedOutput = processedOutput.expand((x) => x).toList();
          tempOutput = flattenedProcessedOutput;
        }
        else {
          tempOutput = processedOutput;
        } */
        // find raw output using score_tensor param
        String scoreTensorName = step.params['score_tensor'];
        scoreTensor = outputTensors[scoreTensorName];
        String numDetectionsTensorName = step.params['num_detections_tensor'];
        dynamic numDetectionsTensor = outputTensors[numDetectionsTensorName];
        final int numDetections = (numDetectionsTensor is List ? numDetectionsTensor[0] as num : numDetectionsTensor as num).toInt();
        
        // set threshold according to YAML file, default to 0.5 if none found
        double threshold = (step.params['threshold'] as num?)?.toDouble() ?? 0.5;
        Map<int, List<int>> filteredDetectionIndices = {};
        debugPrint("Filtering ${scoreTensor[0].length} detections with threshold $threshold...");

        // loop through each batch of score tensors
        for (int i = 0; i < scoreTensor.length; i++) {
          filteredDetectionIndices[i] = <int>[]; // Initialize the list for each batch
          for (int j = 0; j < numDetections; j++) {
            debugPrint("scoreTensor[$i][$j] = ${scoreTensor[i][j]}");
            debugPrint("threshold = $threshold");
            debugPrint("scoreTensor[$i][$j] > threshold = ${scoreTensor[i][j] > threshold}");
            if (scoreTensor[i][j] > threshold) {
              filteredDetectionIndices[i]!.add(j);
            }
          }
        }
        
        debugPrint("InferenceService: Filtered ${filteredDetectionIndices[0]!.length} detections above threshold $threshold");
        if (kDebugMode) {
          developer.log("Inspecting ${step.step} postprocessing variable: filteredDetectionIndices");
          developer.inspect(filteredDetectionIndices);
        }
        // This step's output (filteredIndices) becomes processedData for the next step.
        processedOutput = filteredDetectionIndices;

        if (kDebugMode) {
            debugPrint("Finished ${step.step} postprocessing step.");
            developer.log("Inspecting ${step.step} postprocessing output: processedOutput");
            developer.inspect(processedOutput);
        }

      // uses filtered indices and later on the coordinate format config to construct the final detection tensors
      // expected a Map of filtered indices as input, from 'filter_by_score'
      case 'decode_boxes':
        // Expects processedOutput (from previous step) to be the list of filtered indices
        if (processedOutput is! Map<int, List<int>>) {
          throw FormatException(
              "Step 'decode_boxes' expects Map<int, List<int>> (filtered indices) input. Got ${processedOutput.runtimeType}");
        }
        final params = step.params;
        final boxTensorName = params['box_tensor'] as String?;
        if (boxTensorName == null || !outputTensors.containsKey(boxTensorName)) {
          throw Exception("decode_boxes requires a valid 'box_tensor' param pointing to a raw output.");
        }
        // define raw box Map, generalizing for multiple batch detection box tensors
        final Map<int, List<List<dynamic>>> boxesRaw = {};
        debugPrint("Building boxesRaw map...");
        for (int i = 0; i < outputTensors[boxTensorName].length; i++) {
          boxesRaw[i] = outputTensors[boxTensorName][i];
        }
        debugPrint("BoxesRaw map built successfully. \n $boxesRaw");
        debugPrint("Starting box decoding...");
        Map<int, List<Map<String, dynamic>>> decodedData = {};
        for (int i = 0; i < processedOutput.length; i++) {
          debugPrint("Decoding boxes for batch $i...");
          decodedData[i] = []; // Initialize the list for each batch
          for (int index in processedOutput[i]!) {
            debugPrint("Decoding box $index for batch $i...");
            if (index < boxesRaw[i]!.length) {
              // initialize the map for each detection
              // Convert box coordinates to double
              final box = boxesRaw[i]?[index].map((val) => (val as num).toDouble()).toList();
              if (box?.length == 4) { // Basic validation
                debugPrint("Adding box coordinates: $box");
                decodedData[i]!.add({
                  "original_index": index, // Preserve original index for later mapping
                  "score": scoreTensor[i]?[index],
                  "raw_box": box, // Pass raw normalized box [ymin, xmin, ymax, xmax] or other format
                });
              } else {
                debugPrint("InferenceService: Warning: Box at index $index does not have 4 coordinates in decode_boxes.");
              }
            } else {
              debugPrint("InferenceService: Warning: Index $index out of bounds for boxesRaw (length ${boxesRaw.length}) in decode_boxes.");
            }
          }
        }
        
        if (kDebugMode) {
          developer.log("Inspecting ${step.step} postprocessing variable: decodedData");
          developer.inspect(decodedData);
        }
        // This map of a list of maps becomes processedData for the next step
        processedOutput = decodedData;

        if (kDebugMode) {
          debugPrint("Finished ${step.step} postprocessing step.");
          developer.log("Inspecting ${step.step} postprocessing output: processedOutput");
          developer.inspect(processedOutput);
        }


      default:
        if (kDebugMode) {
          debugPrint("Warning: Unsupported postprocessing step: ${step.step}"); 
        }
    }

    return processedOutput;
  }

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