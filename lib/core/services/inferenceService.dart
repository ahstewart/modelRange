import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';
import 'dart:developer' as developer;
import 'package:yaml/yaml.dart';
import '../data_models/pipeline.dart';
import '../data_models/inference_result_model.dart';
import '../data_models/inference_stat.dart';
import '../services/stats_service.dart';
import '../utils/list_extensions.dart';
import '../utils/data_types.dart';
import '../utils/math.dart';
import 'tokenizer_service.dart';
import 'package:flutter_gemma/flutter_gemma.dart';

// ---------------------------------------------------------------------------
// Top-level helpers required by compute() — must live outside InferenceService
// ---------------------------------------------------------------------------

/// Parameters for [_tfliteInferenceCompute].
class _TfliteComputeParams {
  final String modelPath;
  final List<Object> inputs;
  final List<List<int>> inputShapes;
  final int declaredOutputCount;

  _TfliteComputeParams({
    required this.modelPath,
    required this.inputs,
    required this.inputShapes,
    required this.declaredOutputCount,
  });
}

/// Infer the shape of a nested List by walking the first element at each depth.
/// Mirrors [InferenceService._inferListShape] but usable in a compute() isolate.
List<int> _inferShapeStatic(dynamic data) {
  final dims = <int>[];
  dynamic current = data;
  while (current is List) {
    dims.add((current).length);
    if ((current).isEmpty) break;
    current = (current).first;
  }
  return dims;
}

/// Runs TFLite inference in a background Dart isolate without pre-allocating
/// output buffers. Uses [Interpreter.invoke] directly, then reads post-invoke
/// tensor data for declared outputs only — completely avoids shape-mismatch
/// errors caused by undeclared extra output tensors.
Map<int, dynamic> _tfliteInferenceCompute(_TfliteComputeParams params) {
  final options = InterpreterOptions()..threads = 4;
  final interpreter = Interpreter.fromFile(
    File(params.modelPath),
    options: options,
  );

  // Resize input tensors to actual data shapes, then re-allocate.
  for (int i = 0; i < params.inputShapes.length; i++) {
    if (params.inputShapes[i].isNotEmpty) {
      interpreter.resizeInputTensor(i, params.inputShapes[i]);
    }
  }
  interpreter.allocateTensors();

  // Copy input data into native input tensors.
  final inputTensors = interpreter.getInputTensors();
  for (int i = 0; i < params.inputs.length && i < inputTensors.length; i++) {
    inputTensors[i].setTo(params.inputs[i]);
  }

  // Run inference — no output buffer map needed.
  interpreter.invoke();

  // Read only the declared output tensors using their correct post-invoke shapes.
  final result = <int, dynamic>{};
  final outputTensors = interpreter.getOutputTensors();
  for (
    int i = 0;
    i < params.declaredOutputCount && i < outputTensors.length;
    i++
  ) {
    final tensor = outputTensors[i];
    final shape = List<int>.from(tensor.shape);
    final numElements = shape.fold(1, (int a, int b) => a * b);
    if (numElements <= 0) continue;

    // We must reshape the buffers BEFORE copying, so the FFI shape check passes.
    switch (tensor.type) {
      case TensorType.float32:
        final buf = ListShape(
          List<double>.filled(numElements, 0.0),
        ).reshape(shape);
        tensor.copyTo(buf);
        result[i] = buf;
      case TensorType.uint8:
      case TensorType.int8:
        final buf = ListShape(List<int>.filled(numElements, 0)).reshape(shape);
        tensor.copyTo(buf);
        result[i] = buf;
      case TensorType.int32:
        final buf = ListShape(List<int>.filled(numElements, 0)).reshape(shape);
        tensor.copyTo(buf);
        result[i] = buf;
      default:
        final buf = ListShape(
          List<double>.filled(numElements, 0.0),
        ).reshape(shape);
        tensor.copyTo(buf);
        result[i] = buf;
    }
  }

  interpreter.close();
  return result;
}

// inference class that contains methods for loading models, pre and post processing, and running the actual inference
class InferenceService {
  // initialize interpreter, pipeline (metadata), etc
  Interpreter? _interpreter;
  IsolateInterpreter? _isolateInterpreter;
  InferenceModel? _mediaPipeLlm;
  Pipeline? modelPipeline;

  List<String>? _labels;
  TokenizerService? _tokenizer;
  // Stores the preprocessed inputs from the last performInference call.
  // Used by the autoregressive 'generate' postprocessing step.
  Map<String, dynamic> _lastPreprocessedInputs = {};
  bool get isReady =>
      (_interpreter != null || _mediaPipeLlm != null) && modelPipeline != null;

  bool get _isMediaPipeLlm =>
      modelPipeline?.metadata.firstOrNull?.framework == 'mediapipe_litert';

  dynamic scoreTensor;

  final String modelPath;
  final String pipelinePath;
  // When true, modelPath and pipelinePath are filesystem paths (downloaded models).
  // When false (default), they are Flutter asset paths.
  final bool isLocalFile;
  // Directory containing the downloaded model assets (tflite, labels, etc.).
  // Required when isLocalFile is true.
  final String? localDir;

  // Optional stat recording — set when the model has a known version ID.
  final String? modelVersionId;
  final String? modelDisplayName;
  final StatsService? _statsService;

  InferenceService({
    required this.modelPath,
    required this.pipelinePath,
    this.isLocalFile = false,
    this.localDir,
    this.modelVersionId,
    this.modelDisplayName,
    StatsService? statsService,
  }) : _statsService = statsService;

  // Call this to initialize
  Future<void> initialize() async {
    await loadPipeline(
      pipelinePath,
    ); // Load pipeline first to get model info if needed
    if (_isMediaPipeLlm) {
      await _initializeMediaPipeLlm();
      return;
    }
    await loadModel(modelPath); // Load model
    await loadLabelsIfNeeded(); // Load labels if needed by any postprocessing step
    await loadTokenizerIfNeeded(); // Load tokenizer if any preprocessing block uses text input
  }

  Future<void> _initializeMediaPipeLlm() async {
    // Read generation params from the mediapipe_generate postprocessing step.
    int maxTokens = 512;
    double temperature = 0.8;
    int topK = 40;
    int? randomSeed;
    for (final block in modelPipeline!.postprocessing) {
      for (final step in block.steps) {
        if (step.step == 'mediapipe_generate') {
          maxTokens = (step.params['max_tokens'] as num?)?.toInt() ?? maxTokens;
          temperature =
              (step.params['temperature'] as num?)?.toDouble() ?? temperature;
          topK = (step.params['top_k'] as num?)?.toInt() ?? topK;
          randomSeed = (step.params['random_seed'] as num?)?.toInt();
          break;
        }
      }
    }
    // Resolve ModelType from the pipeline config's model_type string.
    String modelTypeStr = 'gemmaIt';
    for (final block in modelPipeline!.postprocessing) {
      for (final step in block.steps) {
        if (step.step == 'mediapipe_generate') {
          modelTypeStr = (step.params['model_type'] as String?) ?? modelTypeStr;
          break;
        }
      }
    }
    final modelTypeMap = {
      'gemmaIt': ModelType.gemmaIt,
      'general': ModelType.general,
      'deepSeek': ModelType.deepSeek,
      'qwen': ModelType.qwen,
      'llama': ModelType.llama,
      'hammer': ModelType.hammer,
    };
    final resolvedModelType = modelTypeMap[modelTypeStr] ?? ModelType.gemmaIt;

    await FlutterGemma.initialize();
    await FlutterGemma.installModel(
      modelType: resolvedModelType,
      fileType: ModelFileType.task,
    ).fromFile(modelPath).install();
    _mediaPipeLlm = await FlutterGemma.getActiveModel(maxTokens: maxTokens);
    debugPrint('[InferenceService] MediaPipe LLM initialized: $modelPath');
  }

  // load model from model_path (AKA create an interpreter)
  Future<void> loadModel(
    String modelPath, {
    String modelFramework = "tflite",
  }) async {
    if (modelFramework == "tflite") {
      debugPrint("Loading TFLite model from $modelPath...");
      final options = InterpreterOptions()..threads = 4;

      try {
        _interpreter =
            isLocalFile
                ? Interpreter.fromFile(File(modelPath), options: options)
                : await Interpreter.fromAsset(modelPath, options: options);

        _isolateInterpreter = await IsolateInterpreter.create(
          address: _interpreter!.address,
        );

        if (kDebugMode) {
          debugPrint(_interpreter?.getInputTensors().toString());
          debugPrint(_interpreter?.getOutputTensors().toString());
        }
      } catch (e) {
        debugPrint("Failed to load model: $e");
        rethrow;
      }
    }
  }

  // load pipeline from a pipeline_path, using the pipeline data model
  Future<void> loadPipeline(String pipelinePath) async {
    if (kDebugMode) {
      debugPrint("Loading model pipeline file...");
    }

    final Map<String, dynamic> pipelineMap;
    if (isLocalFile) {
      // Local files are stored as JSON (PipelineConfig schema, compatible with Pipeline.fromJson).
      final contents = await File(pipelinePath).readAsString();
      pipelineMap = jsonDecode(contents) as Map<String, dynamic>;
    } else {
      // Asset files are YAML.
      final contents = await rootBundle.loadString(pipelinePath);
      pipelineMap =
          _convertYamlToJson(loadYaml(contents)) as Map<String, dynamic>;
    }

    // create pipeline object from pipeline_map
    modelPipeline = Pipeline.fromJson(pipelineMap);

    if (kDebugMode) {
      debugPrint("Model pipeline file loaded successfully.");
    }
  }

  Future<void> loadLabelsIfNeeded() async {
    debugPrint("Loading labels if needed...");
    if (modelPipeline == null) return;
    String? labelsUrl;
    for (var block in modelPipeline!.postprocessing) {
      for (var step in block.steps) {
        if (step.step == 'map_labels') {
          labelsUrl = step.params['labels_url'] as String?;
          break;
        }
        if (step.step == 'ctc_decode') {
          // Vocabulary for CTC decode is stored as the labels asset (one token per line).
          // Fall through to load from localDir/labels or labels_url below.
          labelsUrl = step.params['vocabulary_url'] as String?;
          break;
        }
      }
      if (labelsUrl != null) break;
    }

    // For local models, load labels from the downloaded file regardless of the pipeline's labels_url.
    if (isLocalFile && localDir != null) {
      final labelsFile = File('$localDir/labels');
      if (await labelsFile.exists()) {
        final labelsData = await labelsFile.readAsString();
        _labels =
            labelsData
                .split('\n')
                .map((l) => l.trim())
                .where((l) => l.isNotEmpty)
                .toList();
        debugPrint(
          'InferenceService: Labels loaded from local file: ${_labels?.length} labels',
        );
      } else {
        _labels = [];
        debugPrint('InferenceService: No local labels file found.');
      }
      return;
    }

    if (labelsUrl != null) {
      try {
        final labelsData = await rootBundle.loadString(labelsUrl);
        _labels =
            labelsData
                .split('\n')
                .map((label) => label.trim())
                .where((label) => label.isNotEmpty)
                .toList();
        if (kDebugMode) {
          debugPrint(
            "InferenceService: Labels loaded from $labelsUrl: ${_labels?.length} labels",
          );
        }
      } catch (e) {
        if (kDebugMode) {
          debugPrint(
            "InferenceService: Failed to load labels from $labelsUrl: $e",
          );
        }
        _labels = []; // Default to empty on error
      }
    } else {
      _labels = [];
      if (kDebugMode) {
        debugPrint(
          "InferenceService: No 'labels_url' found in any 'map_labels' postprocessing step.",
        );
      }
    }
  }

  Future<void> loadTokenizerIfNeeded() async {
    if (modelPipeline == null) return;

    // Only load if at least one preprocessing block handles text input
    final hasTextInput = modelPipeline!.preprocessing.any(
      (b) => b.expects_type == 'text',
    );
    if (!hasTextInput) return;

    // Find optional explicit file references from the tokenize step params
    String? vocabFile;
    String? tokenizerFile;
    for (final block in modelPipeline!.preprocessing) {
      for (final step in block.steps) {
        if (step.step == 'tokenize') {
          vocabFile = step.params['vocab_file'] as String?;
          tokenizerFile = step.params['tokenizer_file'] as String?;
          break;
        }
      }
    }

    _tokenizer = await TokenizerService.load(
      vocabPath: vocabFile,
      tokenizerJsonPath: tokenizerFile,
      isLocalFile: isLocalFile,
      localDir: localDir,
    );

    if (_tokenizer == null) {
      debugPrint(
        '[InferenceService] Warning: text input detected but no tokenizer could be loaded.',
      );
    } else {
      debugPrint('[InferenceService] Tokenizer loaded successfully.');
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
  Future<dynamic> preprocess(dynamic rawInput, String inputName) async {
    // first make sure model and pipeline are loaded
    if (!isReady) {
      if (kDebugMode) {
        debugPrint(
          "Cannot preprocess input, model or pipeline have not been successfully loaded.",
        );
      }
      return rawInput;
    }
    // then make sure pipeline includes preprocessing steps? if it doesn't, skip all of this and return rawInput
    if (modelPipeline!.preprocessing.isEmpty) {
      if (kDebugMode) {
        debugPrint(
          "Pipeline is missing preprocessing block, returning raw input unchanged.",
        );
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
      if (modelPipeline!.inputs[i].name == inputName) {
        inputIndex = i;
        break;
      }
    }

    debugPrint(
      "Matching input name to a preprocessing block in pipeline file.",
    );
    // match _inputName to a preprocessing block
    for (var i = 0; i < modelPipeline!.preprocessing.length; i++) {
      if (modelPipeline!.preprocessing[i].input_name == inputName) {
        preprocessBlockIndex = i;
        break;
      }
    }

    // check if the input could not be matched to an input block or preprocessing block
    if (inputIndex == null || preprocessBlockIndex == null) {
      if (kDebugMode) {
        debugPrint(
          "Provided input name does not match input or preprocessing block in pipeline. Aborting preprocessing.",
        );
      }
      return rawInput;
    }

    // once the preprocessing step and input are matched, use the expects_type to validate the rawInput
    final String expectedType =
        modelPipeline!.preprocessing[preprocessBlockIndex].expects_type;

    debugPrint(
      "Validating that the raw input matches the 'expects_type' parameter in the pipeline file...",
    );
    switch (expectedType) {
      case 'image':
        if (rawInput is! img.Image) {
          if (kDebugMode) {
            debugPrint("Raw input type: ${rawInput.runtimeType.toString()}");
            throw ArgumentError(
              "This preprocessing block expects an image, but raw input is not an img.Image type. Aborting preprocessing.",
            );
          }
          return rawInput;
        }
        if (kDebugMode) {
          debugPrint(
            "Raw input type matches expected type. Proceeding with preprocessing.",
          );
        }
        break;
      case 'text':
        if (rawInput is! String) {
          if (kDebugMode) {
            debugPrint("Raw input type: ${rawInput.runtimeType.toString()}");
            throw ArgumentError(
              "This preprocessing block expects text, but raw input is not a String. Aborting preprocessing.",
            );
          }
          return rawInput;
        }
        if (kDebugMode) {
          debugPrint(
            "Raw input type match expected type. Proceeding with preprocessing.",
          );
        }
        break;
      case 'audio':
        if (rawInput is! Uint8List) {
          if (kDebugMode) {
            debugPrint("Raw input type: ${rawInput.runtimeType.toString()}");
            throw ArgumentError(
              "This preprocessing block expects a audio, but raw input is not a Uint8List type. Aborting preprocessing.",
            );
          }
          return rawInput;
        }
        if (kDebugMode) {
          debugPrint(
            "Raw input type match expected type. Proceeding with preprocessing.",
          );
        }
        break;
      // Add cases for other expected raw input types ('tensor', 'generic_list', etc.)
      default:
        throw UnimplementedError(
          "Unsupported 'expects_type' in pipeline: $expectedType",
        );
    }
    debugPrint("Raw input type validated.");

    // now that input is validated, start tracking the input with a variable
    // initialize it with the rawInput
    dynamic currentInput = rawInput;

    debugPrint(
      "Executing steps for preprocessing block ${modelPipeline!.preprocessing[preprocessBlockIndex].input_name}...",
    );
    // start looping through the preprocessing steps
    for (var preStep
        in modelPipeline!.preprocessing[preprocessBlockIndex].steps) {
      currentInput = await _performPreprocessingStep(
        currentInput,
        preStep,
        preprocessBlockIndex,
      );
      debugPrint(
        "Preprocessing step ${preStep.step} completed successfully...",
      );
    }

    debugPrint("Preprocessing complete.");

    // return final input tensor
    return currentInput;
  }

  // postprocess complete a postprocessing block and return a Map which will be the final inference result
  // the input to postprocess is the postprocessing block, the raw outputs map (source tensors), and the final result map
  Future<dynamic> postprocess(
    Map<String, dynamic> rawOutputs,
    int postprocessBlockIndex,
  ) async {
    debugPrint("Starting postprocessing...");

    // declare some variables to make things easier
    ProcessingBlock block =
        modelPipeline!.postprocessing[postprocessBlockIndex];
    String outputName = block.output_name;
    // declare a variable to track the current output
    dynamic currentResult = rawOutputs[block.source_tensors[0]];
    String interpretation = block.interpretation.toString().toLowerCase();

    debugPrint("Checking if the postprocessing block contains steps...");
    // make sure pipeline includes postprocessing steps? if it doesn't, skip all of this and return rawInput
    if (modelPipeline!.postprocessing.isEmpty) {
      if (kDebugMode) {
        debugPrint(
          "Pipeline is missing postprocessing blocks, returning raw output unchanged.",
        );
      }
      return currentResult;
    }

    debugPrint("Postprocessing block contained steps.");

    debugPrint(
      "Checking if source tensors are present in all postprocessing blocks...",
    );
    // Autoregressive generate blocks produce their own output internally —
    // they don't read from rawOutputs, so skip the source tensor check.
    final bool isAutoregressiveBlock =
        block.steps.isNotEmpty &&
        block.steps.first.step == 'generate' &&
        (block.steps.first.params['mode'] as String?) == 'autoregressive';

    if (!isAutoregressiveBlock) {
      // check that all source tensors in the postprocessing block are present in the output map
      List<String> sourceTensors = block.source_tensors;
      for (var tensor in sourceTensors) {
        if (!rawOutputs.containsKey(tensor)) {
          if (kDebugMode) {
            debugPrint(
              "Output map does not contain source tensor: $tensor. Aborting postprocessing and returning raw output unchanged.",
            );
          }
          return currentResult;
        }
      }
    }

    debugPrint("Source tensors present in all postprocessing blocks.");

    debugPrint("Executing steps for ${block.output_name}...");
    // start looping through the postprocessing steps
    for (var postStep in block.steps) {
      currentResult = await _performPostprocessingStep(
        currentResult,
        rawOutputs,
        postStep,
      );
      if (currentResult == null && postStep != block.steps.last) {
        throw Exception(
          "Postprocessing block '$outputName', step '${postStep.step}' failed or returned null unexpectedly.",
        );
      }
      debugPrint(
        "Postprocessing step ${postStep.step} completed successfully...",
      );
    }

    if (kDebugMode) {
      debugPrint("Postprocessing complete.");
      developer.log(
        "Inspecting postprocessing result for block ${modelPipeline!.postprocessing[postprocessBlockIndex].output_name}",
      );
      developer.inspect(currentResult);
    }

    debugPrint(
      "Adding result from postprocess block ${modelPipeline!.postprocessing[postprocessBlockIndex].output_name} to the final results map.",
    );
    debugPrint(
      "Mapping postprocessed result to a sealed inference result class for interpretation $interpretation",
    );
    switch (interpretation) {
      case 'classification_logits':
      case 'classification_scores':
      case 'classification_probabilities':
        if (currentResult is List) {
          try {
            return ClassificationResult(
              currentResult.cast<Map<String, dynamic>>(),
            );
          } catch (e) {
            return ErrorResult(
              "Postprocessing for '$outputName' (classification) produced a List, but elements were not Map<String, dynamic>.",
            );
          }
        }
        return ErrorResult(
          "Postprocessing for '$outputName' (classification) did not produce a List. Got ${currentResult.runtimeType}",
        );
      case 'detection_boxes_scores_classes':
        if (currentResult is Map<int, List<Map<String, dynamic>>>) {
          try {
            return ErrorResult("test");
          } catch (e) {
            return ErrorResult(
              "Postprocessing for '$outputName' (detection) produced a Map<int, List<Map<String, dynamic>>> but failed to cast.",
            );
          }
        } else if (currentResult is List &&
            currentResult.isNotEmpty &&
            currentResult.first is Map<int, List<Map<String, dynamic>>>) {
          try {
            return DetectionResult(
              currentResult.first as Map<int, List<Map<String, dynamic>>>,
            );
          } catch (e) {
            return ErrorResult(
              "Postprocessing for '$outputName' (detection) produced a List, but elements were not Map<int, List<Map<String, dynamic>>>.",
            );
          }
        }
        return ErrorResult(
          "Postprocessing for '$outputName' (detection) did not produce a Map<int, List<Map<String, dynamic>>>. Got ${currentResult.runtimeType}",
        );
      case 'text_generation':
        if (currentResult is String) {
          return TextResult(currentResult);
        }
        return ErrorResult(
          "Postprocessing for '$outputName' (text) did not produce a String. Got ${currentResult.runtimeType}",
        );
      case 'image_to_text':
        if (currentResult is String) return TextResult(currentResult);
        return ErrorResult(
          "image_to_text postprocessing did not produce a String for '$outputName'.",
        );
      case 'generated_image':
        if (currentResult is img.Image)
          return GeneratedImageResult(currentResult);
        return ErrorResult(
          "generated_image postprocessing did not produce an img.Image for '$outputName'.",
        );
      case 'speech_recognition':
        if (currentResult is String) {
          return SpeechResult(currentResult);
        }
        return ErrorResult(
          "Postprocessing for '$outputName' (speech) did not produce a String. Got ${currentResult.runtimeType}",
        );
      case 'segmentation_mask':
        if (currentResult is img.Image) {
          int numClasses = 1;
          List<List<int>> palette = [];
          String? colorMap;
          for (final s in block.steps) {
            if (s.step == 'decode_segmentation_mask') {
              numClasses = (s.params['num_classes'] as num).toInt();
              colorMap = s.params['color_map'] as String?;
              palette = _buildSegmentationPalette(numClasses, colorMap);
              break;
            }
          }
          return SegmentationResult(
            mask: currentResult,
            numClasses: numClasses,
            palette: palette,
            labels: _labels?.isNotEmpty == true ? _labels : null,
          );
        }
        return ErrorResult(
          "Postprocessing for '$outputName' (segmentation) did not produce an img.Image. Got ${currentResult.runtimeType}",
        );
      case 'image_segmentation_masks':
        // Single-channel class-index mask: raw tensor shape [1, H, W, 1] uint8.
        // map_labels may have mangled currentResult, so read from rawOutputs directly.
        final rawMask =
            block.source_tensors.isNotEmpty
                ? rawOutputs[block.source_tensors[0]]
                : null;
        if (rawMask == null) {
          return ErrorResult(
            "image_segmentation_masks: source tensor '${block.source_tensors.firstOrNull}' not found.",
          );
        }
        int numClasses = 0;
        String? colorMap;
        for (final s in block.steps) {
          if (numClasses == 0 && s.params.containsKey('num_classes')) {
            numClasses = (s.params['num_classes'] as num).toInt();
          }
          if (colorMap == null && s.params.containsKey('color_map')) {
            colorMap = s.params['color_map'] as String?;
          }
        }
        try {
          final List batch0 = (rawMask as List)[0] as List;
          final int H = batch0.length;
          final int W = (batch0[0] as List).length;
          if (numClasses == 0) {
            for (int h = 0; h < H; h++) {
              for (int w = 0; w < W; w++) {
                final cell = (batch0[h] as List)[w];
                final idx =
                    cell is List
                        ? (cell[0] as num).toInt()
                        : (cell as num).toInt();
                if (idx + 1 > numClasses) numClasses = idx + 1;
              }
            }
            numClasses = numClasses.clamp(1, 256);
          }
          final palette = _buildSegmentationPalette(numClasses, colorMap);
          final maskImg = img.Image(width: W, height: H);
          for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
              final cell = (batch0[h] as List)[w];
              final classIdx = (cell is List
                      ? (cell[0] as num).toInt()
                      : (cell as num).toInt())
                  .clamp(0, palette.length - 1);
              final rgb = palette[classIdx];
              maskImg.setPixel(w, h, img.ColorRgb8(rgb[0], rgb[1], rgb[2]));
            }
          }
          return SegmentationResult(
            mask: maskImg,
            numClasses: numClasses,
            palette: palette,
            labels: _labels?.isNotEmpty == true ? _labels : null,
          );
        } catch (e) {
          return ErrorResult(
            "image_segmentation_masks colorization failed: $e",
          );
        }
      default:
        if (kDebugMode) {
          debugPrint(
            "InferenceService: Warning: Unknown postprocessing interpretation '$interpretation' for output '$outputName'. Wrapping as GenericDataResult.",
          );
        }
        return GenericDataResult(currentResult);
    }
  }

  // execute a preprocessing step, given the step and the input data
  Future<dynamic> _performPreprocessingStep(
    dynamic inputData,
    ProcessingStep step,
    int preprocessingBlockIndex,
  ) async {
    if (kDebugMode) {
      debugPrint("Executing preprocessing step: ${step.step}");
    }

    switch (step.step) {
      // resizing an image, input is img.Image
      case 'resize_image':
        try {
          int targetWidth = (step.params['width'] as num?)?.toInt() ?? 224;
          int targetHeight = (step.params['height'] as num?)?.toInt() ?? 224;
          // Use actual interpreter input tensor shape when available — overrides
          // the pipeline-declared size, which the LLM may have got wrong.
          final actualShape = _getActualInputShape(preprocessingBlockIndex, []);
          if (actualShape.length == 4 &&
              actualShape[1] > 0 &&
              actualShape[2] > 0) {
            if (kDebugMode &&
                (actualShape[1] != targetHeight ||
                    actualShape[2] != targetWidth)) {
              debugPrint(
                '[InferenceService] resize_image: pipeline size '
                '${targetWidth}x$targetHeight overridden by actual model '
                'input shape ${actualShape[2]}x${actualShape[1]}.',
              );
            }
            targetHeight = actualShape[1];
            targetWidth = actualShape[2];
          }
          img.Image resizedImage = img.copyResize(
            inputData,
            width: targetWidth,
            height: targetHeight,
          );
          return resizedImage;
        } catch (e) {
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

        // Quantized (uint8) models embed normalization in the model weights.
        // Passing float-normalized values would corrupt inference — skip to
        // raw byte conversion only.
        final actualInputDtype = _getActualInputDtype(preprocessingBlockIndex);
        if (actualInputDtype == 'uint8' || actualInputDtype == 'int8') {
          if (kDebugMode) {
            debugPrint(
              '[InferenceService] normalize: skipping float normalization '
              'because actual model input dtype is $actualInputDtype.',
            );
          }
          if (inputData is img.Image) {
            return Uint8List.fromList(
              imgToBytes(inputData, normalizeColorSpace),
            );
          }
          return inputData;
        }

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
            } catch (e) {
              if (kDebugMode) {
                debugPrint('Error converting image to bytes: $e');
              }
              throw Exception(
                'Normalization error: Failed to convert image to bytes',
              );
            }
          } else {
            debugPrint("Input data type is ${inputData.runtimeType}");
          }
          var normBytes = Float32List(inputData.length);
          // first try the 'mean_stddev' method, which first normalizes the image pixel between 0 and 1 by dividing by 255,
          // then applies the mean and stddev normalization for each channel. This method requires the mean and stddev parameters
          // to be lists with a length equal to the number of channels in the inputImage
          debugPrint("Executing normalization using the $method method...");
          if (method == "mean_stddev") {
            List mean =
                step.params['mean'] ??
                List.filled(normalizeColorSpace.length, 0.456);
            List stddev =
                step.params['stddev'] ??
                List.filled(normalizeColorSpace.length, 0.224);
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
            for (var i = 0; i < inputData.length; i++) {
              normBytes[i] = (inputData[i] - mean) / stddev;
            }
            // final normImage = normBytes.reshape([1, inputImage.width, inputImage.height, inputImage.numChannels]);
            return normBytes;
          }
          // uniform scaling - simply normalize all pixels to be between 0 and 1, based on a given scale_param (usually 255)
          else if (method == "scale_div") {
            var value = step.params['value'] ?? 255.0;
            for (var i = 0; i < inputData.length; i++) {
              normBytes[i] = inputData[i] / value;
            }
            // final normImage = normBytes.reshape([1, inputImage.width, inputImage.height, inputImage.numChannels]);
            return normBytes;
          } else {
            if (kDebugMode) {
              debugPrint("Error normalizing image: Invalid 'method' parameter");
            }
          }
          debugPrint("Normalized image successfully.");
          return inputData; // return input image bytes if normalization didn't take place
        } catch (e) {
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

        // Override pipeline-declared dtype with the actual interpreter input dtype.
        final actualFormatDtype = _getActualInputDtype(preprocessingBlockIndex);
        if (actualFormatDtype.isNotEmpty && actualFormatDtype != targetDtype) {
          if (kDebugMode) {
            debugPrint(
              '[InferenceService] format: pipeline dtype $targetDtype '
              'overridden by actual model input dtype $actualFormatDtype.',
            );
          }
          targetDtype = actualFormatDtype;
        }

        // first format data type and color space. supports 2 types of conversions: float32 and uint8
        // color space formatting is done with the imgToBytes helper method
        if (targetDtype == 'float32' && inputData is! Float32List) {
          if (inputData is img.Image) {
            var imageBytes = imgToBytes(inputData, inputColorSpace);
            finalData = Float32List.fromList(
              imageBytes.map((e) => e / 255.0).toList(),
            );
          } else {
            throw Exception("Cannot format unsupported type to float32");
          }
        } else if (targetDtype == 'uint8' && inputData is! Uint8List) {
          if (inputData is img.Image) {
            finalData = imgToBytes(inputData, inputColorSpace);
          } else {
            throw Exception("Cannot format unsupported type to uint8");
          }
        }

        debugPrint("finalData runtime type = ${finalData.runtimeType}");

        // Compute the proper spatial shape and, if needed, reorder channels.
        // imgToBytes always produces pixels in H*W*C (NHWC) row-major order.
        List<int> finaldataShape;
        if ((finalData is Float32List || finalData is Uint8List) &&
            inputData is img.Image) {
          final int H = inputData.height;
          final int W = inputData.width;
          const int C = 3; // imgToBytes outputs 3-channel RGB/BGR
          if (dataLayout == 'nchw') {
            // Reorder bytes from NHWC → NCHW in-place.
            if (finalData is Float32List) {
              final out = Float32List(H * W * C);
              for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++) {
                  for (int c = 0; c < C; c++) {
                    out[c * H * W + h * W + w] =
                        (finalData)[h * W * C + w * C + c];
                  }
                }
              finalData = out;
            } else {
              final out = Uint8List(H * W * C);
              for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++) {
                  for (int c = 0; c < C; c++) {
                    out[c * H * W + h * W + w] =
                        (finalData as Uint8List)[h * W * C + w * C + c];
                  }
                }
              finalData = out;
            }
            finaldataShape = [1, C, H, W];
          } else {
            // NHWC — already the correct byte order.
            finaldataShape = [1, H, W, C];
          }
        } else if (finalData is Float32List || finalData is Uint8List) {
          finaldataShape = [1, finalData.length];
        } else {
          finaldataShape = finalData.shape;
        }

        // Use actual interpreter tensor shape as authoritative target.
        List<int> targetInputShape = _getActualInputShape(
          preprocessingBlockIndex,
          [],
        );
        if (targetInputShape.isEmpty) {
          final inputName =
              modelPipeline!.preprocessing[preprocessingBlockIndex].input_name;
          for (final inp in modelPipeline!.inputs) {
            if (inp.name == inputName) {
              targetInputShape = inp.shape;
              break;
            }
          }
        }

        // Decide the reshape target: prefer interpreter shape when element counts match.
        final int dataLen =
            (finalData is Float32List)
                ? (finalData).length
                : (finalData is Uint8List ? (finalData).length : 0);
        final int targetLen = targetInputShape.fold<int>(1, (a, b) => a * b);
        final List<int> reshapeTarget =
            (targetInputShape.isNotEmpty && dataLen == targetLen)
                ? targetInputShape
                : finaldataShape;

        if (kDebugMode &&
            finaldataShape.toString() != targetInputShape.toString()) {
          debugPrint(
            "format: spatial shape $finaldataShape, model expects $targetInputShape"
            "${dataLen == targetLen ? ' — reshaping to model shape' : ' — incompatible, model may not accept direct image input'}",
          );
        }

        try {
          return (finalData is Float32List)
              ? (finalData).reshape(reshapeTarget)
              : (finalData as Uint8List).reshape(reshapeTarget);
        } catch (e) {
          if (kDebugMode) debugPrint("format step reshape failed: $e");
          return finalData;
        }

      // tokenize text input to a tensor of token IDs
      case 'tokenize':
        if (_tokenizer == null) {
          throw StateError(
            "Preprocessing step 'tokenize' requires a tokenizer, but none was loaded. "
            "Ensure the model has a tokenizer.json or vocab.txt asset.",
          );
        }
        if (inputData is! String) {
          throw ArgumentError(
            "Step 'tokenize' expects a String input, got ${inputData.runtimeType}.",
          );
        }
        final int paramMaxLength =
            (step.params['max_length'] as num?)?.toInt() ?? 512;
        // Use the interpreter's actual input tensor shape — the pipeline-declared
        // max_length may have been incorrectly generated (e.g. from n_ctx in config.json
        // rather than the TFLite model's baked-in positional embedding size).
        final int maxLength = _getModelInputSeqLen(fallback: paramMaxLength);
        if (kDebugMode && maxLength != paramMaxLength) {
          debugPrint(
            '[InferenceService] tokenize: pipeline max_length=$paramMaxLength overridden by actual model input shape=$maxLength.',
          );
        }
        final bool padding = step.params['padding'] as bool? ?? true;
        final bool truncation = step.params['truncation'] as bool? ?? true;
        final bool addSpecialTokens =
            step.params['add_special_tokens'] as bool? ?? true;

        final List<int> tokenIds = _tokenizer!.encode(
          inputData,
          maxLength: maxLength,
          padding: padding,
          truncation: truncation,
          addSpecialTokens: addSpecialTokens,
        );
        if (kDebugMode) {
          debugPrint(
            '[InferenceService] Tokenized to ${tokenIds.length} token IDs.',
          );
        }
        // Wrap in an outer list to match shape [1, sequenceLength]
        return [tokenIds];

      case 'resample_audio':
        if (inputData is! Uint8List) {
          throw ArgumentError(
            "Step 'resample_audio' expects Uint8List (WAV bytes), got ${inputData.runtimeType}.",
          );
        }
        return _resampleAudio(inputData, step.params, preprocessingBlockIndex);

      default:
        if (kDebugMode) {
          debugPrint("Warning: Unsupported preprocessing step: ${step.step}");
        }
        return inputData;
    }
  }

  /// Decodes WAV bytes, resamples to target sample rate, normalizes, and reshapes
  /// to the model's expected input shape [1, N].
  dynamic _resampleAudio(
    Uint8List wavBytes,
    Map<String, dynamic> params,
    int preprocessingBlockIndex,
  ) {
    final int targetSampleRate =
        (params['target_sample_rate'] as num?)?.toInt() ?? 16000;
    final double maxDurationS =
        (params['max_duration_s'] as num?)?.toDouble() ?? 10.0;
    final bool normalize = params['normalize'] as bool? ?? true;

    // --- Parse WAV header ---
    // WAV format: RIFF chunk (12 bytes) + fmt sub-chunk + data sub-chunk
    // We scan for the "data" marker to handle variable-length fmt chunks.
    if (wavBytes.length < 44) {
      throw FormatException("WAV file too short (${wavBytes.length} bytes).");
    }
    final ByteData header = ByteData.sublistView(wavBytes);

    // Read fmt fields at fixed offsets (standard PCM WAV)
    final int numChannels = header.getUint16(22, Endian.little);
    final int sourceSampleRate = header.getInt32(24, Endian.little);
    final int bitDepth = header.getUint16(34, Endian.little);

    if (bitDepth != 16) {
      throw FormatException(
        "resample_audio only supports 16-bit PCM WAV, got $bitDepth-bit.",
      );
    }

    // Scan for "data" sub-chunk marker
    int dataOffset = -1;
    int dataLength = 0;
    for (int i = 12; i < wavBytes.length - 8; i++) {
      if (wavBytes[i] == 0x64 && // 'd'
          wavBytes[i + 1] == 0x61 && // 'a'
          wavBytes[i + 2] == 0x74 && // 't'
          wavBytes[i + 3] == 0x61) {
        // 'a'
        dataOffset = i + 8;
        dataLength = header.getInt32(i + 4, Endian.little);
        break;
      }
    }
    if (dataOffset < 0) {
      throw FormatException("WAV 'data' chunk not found.");
    }

    final int totalSamples = dataLength ~/ (bitDepth ~/ 8);
    final int framesToRead = totalSamples ~/ numChannels;

    // Read Int16 PCM samples
    final Int16List pcm16 = wavBytes.buffer.asInt16List(
      wavBytes.offsetInBytes + dataOffset,
      totalSamples,
    );

    // Convert to float32 and mix down to mono
    Float32List mono;
    if (numChannels == 1) {
      mono = Float32List(framesToRead);
      for (int i = 0; i < framesToRead; i++) {
        mono[i] = pcm16[i] / 32768.0;
      }
    } else {
      mono = Float32List(framesToRead);
      for (int i = 0; i < framesToRead; i++) {
        double sum = 0.0;
        for (int c = 0; c < numChannels; c++) {
          sum += pcm16[i * numChannels + c];
        }
        mono[i] = sum / numChannels / 32768.0;
      }
    }

    // Resample via linear interpolation if needed
    Float32List resampled;
    if (sourceSampleRate == targetSampleRate) {
      resampled = mono;
    } else {
      final double ratio = sourceSampleRate / targetSampleRate;
      final int outLen = (mono.length / ratio).ceil();
      resampled = Float32List(outLen);
      for (int i = 0; i < outLen; i++) {
        final double srcPos = i * ratio;
        final int srcIdx = srcPos.floor();
        final double frac = srcPos - srcIdx;
        if (srcIdx + 1 < mono.length) {
          resampled[i] = mono[srcIdx] * (1.0 - frac) + mono[srcIdx + 1] * frac;
        } else if (srcIdx < mono.length) {
          resampled[i] = mono[srcIdx];
        }
      }
    }

    // Peak-normalize to [-1.0, 1.0]
    if (normalize && resampled.isNotEmpty) {
      double peak = 0.0;
      for (final s in resampled) {
        final abs = s.abs();
        if (abs > peak) peak = abs;
      }
      if (peak > 0.0) {
        for (int i = 0; i < resampled.length; i++) {
          resampled[i] /= peak;
        }
      }
    }

    // Trim or zero-pad to target length
    final int targetLen = (targetSampleRate * maxDurationS).round();
    // Prefer actual interpreter shape if available
    final actualShape = _getActualInputShape(preprocessingBlockIndex, []);
    final int N =
        (actualShape.length == 2 && actualShape[1] > 0)
            ? actualShape[1]
            : targetLen;

    final Float32List output = Float32List(N);
    final int copyLen = math.min(resampled.length, N);
    for (int i = 0; i < copyLen; i++) {
      output[i] = resampled[i];
    }

    // Reshape to [1, N] as nested list (tflite_flutter expects List for multi-dim inputs)
    return output.reshape([1, N]);
  }

  // perform inference given inputs and target output buffers
  // order of inferenceInputs will be mapped directly to the order of the pipeline inputs, so they must match
  // ── Public entry point (times the run and records stats) ─────────────────

  Future<Map<String, InferenceResult>> performInference(
    Map<String, dynamic> inferenceInputs,
  ) async {
    final sw = Stopwatch()..start();
    Map<String, InferenceResult> results = {};
    bool success = true;
    try {
      results = await _doInference(inferenceInputs);
      success = !results.values.any((r) => r is ErrorResult);
      return results;
    } catch (e) {
      success = false;
      rethrow;
    } finally {
      sw.stop();
      if (modelVersionId != null && _statsService != null) {
        _recordStat(sw.elapsedMilliseconds, success, results);
      }
    }
  }

  void _recordStat(int ms, bool success, Map<String, InferenceResult> results) {
    double? topConfidence;
    int? numResults;
    for (final result in results.values) {
      if (result is ClassificationResult && result.results.isNotEmpty) {
        topConfidence =
            (result.results.first['confidence'] as num?)?.toDouble();
        numResults = result.results.length;
        break;
      } else if (result is DetectionResult) {
        final all = result.results.values.expand((v) => v).toList();
        numResults = all.length;
        if (all.isNotEmpty) {
          final confs = all.map(
            (d) => (d['confidence'] as num? ?? 0).toDouble(),
          );
          topConfidence = confs.reduce((a, b) => a + b) / confs.length;
        }
        break;
      }
    }

    final stat = InferenceStat(
      modelVersionId: modelVersionId!,
      modelName: modelDisplayName ?? '',
      taskType: modelPipeline?.metadata.firstOrNull?.model_task,
      timestamp: DateTime.now(),
      totalInferenceMs: ms,
      success: success,
      topConfidence: topConfidence,
      numResults: numResults,
      deviceModel: DeviceInfoHelper.cachedDeviceModel,
      platform: Platform.isAndroid ? 'android' : 'ios',
    );

    // Fire-and-forget — stat recording must never delay inference results.
    _statsService!.recordStat(stat).catchError((_) {});
  }

  // ── Internal inference implementation ────────────────────────────────────

  // inferenceInputs is map keyed by the input name, so that the given input
  Future<Map<String, InferenceResult>> _doInference(
    Map<String, dynamic> inferenceInputs,
  ) async {
    if (_isMediaPipeLlm) {
      return _runMediaPipeLlmInference(inferenceInputs);
    }
    // check that the inputs provided match the inputs expected based on the pipeline file
    if (inferenceInputs.length != modelPipeline!.inputs.length) {
      throw ArgumentError(
        "Provided number of inputs (${inferenceInputs.length}) and expected number of inputs ($modelPipeline!.inputs.length}) do not match, cannot proceed with inference.",
      );
    }

    // preprocess inputs and construct the final input list for inference
    _lastPreprocessedInputs = {};
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
      // store for use by the autoregressive generate step
      _lastPreprocessedInputs[input.key] = tempInput;
      // add input to the final processInputs list
      processedInputs.add(tempInput);
    }

    List<IO> outputs = modelPipeline!.outputs;

    // Skip the initial inference pass for autoregressive pipelines — the
    // 'generate' postprocessing step runs its own loop from _lastPreprocessedInputs
    // and ignores this output entirely, so running it here is wasted work.
    final bool isAutoregressive = modelPipeline!.postprocessing.any(
      (block) => block.steps.any(
        (s) =>
            s.step == 'generate' &&
            (s.params['mode'] as String?) == 'autoregressive',
      ),
    );

    // Resize input tensors to actual data shapes so that dynamic output shapes
    // are propagated correctly by allocateTensors().
    if (_interpreter != null) {
      bool resized = false;
      for (int i = 0; i < processedInputs.length; i++) {
        final shape = _inferListShape(processedInputs[i]);
        if (shape.isNotEmpty) {
          _interpreter!.resizeInputTensor(i, shape);
          resized = true;
        }
      }
      if (resized) _interpreter!.allocateTensors();
    }

    Map<String, dynamic> inferenceOutputs = {};
    if (!isAutoregressive) {
      // Models that expose more output tensors than the pipeline declares (e.g.
      // CTC ASR wav2vec2 with 13 tensors, 12 undeclared) cannot use
      // IsolateInterpreter.runForMultipleInputs: that API null-checks EVERY
      // tensor index and its pre-allocated buffer must match the post-invoke
      // shape — impossible to guarantee for undeclared dynamic tensors.
      // Solution: compute() + invoke() reads native tensor data after inference
      // with correct post-invoke shapes; undeclared extras are simply ignored.
      final actualTensorCount =
          _interpreter?.getOutputTensors().length ?? outputs.length;
      final hasExtraOutputs = isLocalFile && actualTensorCount > outputs.length;

      if (hasExtraOutputs) {
        if (kDebugMode) {
          debugPrint(
            '[performInference] $actualTensorCount output tensors, '
            '${outputs.length} declared → using compute() + invoke() path.',
          );
        }
        final inputShapes = processedInputs.map(_inferListShape).toList();
        final computeResult = await compute(
          _tfliteInferenceCompute,
          _TfliteComputeParams(
            modelPath: modelPath,
            inputs: processedInputs,
            inputShapes: inputShapes,
            declaredOutputCount: outputs.length,
          ),
        );
        if (kDebugMode) debugPrint('Inference completed (compute path).');
        for (int i = 0; i < outputs.length; i++) {
          if (computeResult.containsKey(i)) {
            inferenceOutputs[outputs[i].name] = computeResult[i];
          }
        }
      } else {
        // Standard path: IsolateInterpreter with pre-allocated output buffers.
        // Safe when the model exposes only the pipeline-declared output tensors.
        Map<int, Object> outputBuffers = _buildOutputBuffers(outputs);
        if (kDebugMode) debugPrint("Running inference on model.");
        await _isolateInterpreter!.runForMultipleInputs(
          processedInputs,
          outputBuffers,
        );
        debugPrint("Inference completed.");
        if (kDebugMode) {
          developer.log(
            "Inspecting outputBuffers variable from the TFLite inference.",
          );
          developer.inspect(outputBuffers);
        }
        final actualOutputTensors = _interpreter?.getOutputTensors();
        for (int i = 0; i < outputs.length; i++) {
          final flat = outputBuffers[i];
          final List<int> shape =
              (actualOutputTensors != null && i < actualOutputTensors.length)
                  ? List<int>.from(actualOutputTensors[i].shape)
                  : outputs[i].shape;
          inferenceOutputs[outputs[i].name] = switch (flat) {
            Float32List f => f.reshape(shape),
            Uint8List u => u.reshape(shape),
            Int32List il => ListShape(il.toList()).reshape(shape),
            _ => flat,
          };
        }
      }
    } else {
      if (kDebugMode) {
        debugPrint(
          "Autoregressive pipeline — skipping initial inference pass.",
        );
      }
    }

    // define the final results map, which will contain the final output from the model inference
    // the map is keyed by each postprocessing block's name and final output
    Map<String, InferenceResult> finalResults = {};

    // check if any postprocessing blocks exist
    if (modelPipeline!.postprocessing.isNotEmpty) {
      // loop through the postprocessing blocks and run the postprocess method
      for (int i = 0; i < modelPipeline!.postprocessing.length; i++) {
        String blockName = modelPipeline!.postprocessing[i].output_name;
        finalResults[blockName] = await postprocess(inferenceOutputs, i);
        if (kDebugMode) {
          debugPrint(
            "Postprocessing block ${modelPipeline!.postprocessing[i].output_name} completed.",
          );
        }
      }
    } else {
      if (kDebugMode) {
        debugPrint(
          "No postprocessing blocks found in pipeline, returning raw output.",
        );
      }
      finalResults = inferenceOutputs.map(
        (key, value) => MapEntry(key, GenericDataResult(value)),
      );
    }

    if (kDebugMode) {
      developer.log(
        "Postprocessing complete, inspecting the finalResults output...",
      );
      developer.inspect(finalResults);
    }

    // return the final results map
    return finalResults;
  }

  Future<Map<String, InferenceResult>> _runMediaPipeLlmInference(
    Map<String, dynamic> inputs,
  ) async {
    final prompt = inputs.values.first.toString();

    // Read per-session generation params from the pipeline config.
    double temperature = 0.8;
    int topK = 40;
    int randomSeed = 0;
    for (final block in modelPipeline!.postprocessing) {
      for (final step in block.steps) {
        if (step.step == 'mediapipe_generate') {
          temperature =
              (step.params['temperature'] as num?)?.toDouble() ?? temperature;
          topK = (step.params['top_k'] as num?)?.toInt() ?? topK;
          randomSeed =
              (step.params['random_seed'] as num?)?.toInt() ?? randomSeed;
          break;
        }
      }
    }

    final session = await _mediaPipeLlm!.createSession(
      temperature: temperature,
      topK: topK,
      randomSeed: randomSeed,
    );
    await session.addQueryChunk(Message.text(text: prompt, isUser: true));
    final response = await session.getResponse();
    await session.close();
    return {'generated_text': TextResult(response)};
  }

  /// Returns an N×3 RGB palette for segmentation mask colorization.
  /// Supports "pascal_voc" (21 classes), "cityscapes" (19 classes),
  /// and auto-generated HSV hues for everything else.
  List<List<int>> _buildSegmentationPalette(int numClasses, String? colorMap) {
    if (colorMap == 'pascal_voc') {
      return [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
      ];
    }
    if (colorMap == 'cityscapes') {
      return [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
      ];
    }
    // Auto-generate evenly-spaced HSV hues.
    final List<List<int>> palette = [];
    for (int i = 0; i < numClasses; i++) {
      final double hue = (i / numClasses) * 360.0;
      final color = img.ColorRgb8(0, 0, 0);
      // HSV→RGB: S=1, V=1
      final double s = 1.0, v = 1.0;
      final double c = v * s;
      final double x = c * (1 - ((hue / 60) % 2 - 1).abs());
      final double m = v - c;
      double r1, g1, b1;
      if (hue < 60) {
        r1 = c;
        g1 = x;
        b1 = 0;
      } else if (hue < 120) {
        r1 = x;
        g1 = c;
        b1 = 0;
      } else if (hue < 180) {
        r1 = 0;
        g1 = c;
        b1 = x;
      } else if (hue < 240) {
        r1 = 0;
        g1 = x;
        b1 = c;
      } else if (hue < 300) {
        r1 = x;
        g1 = 0;
        b1 = c;
      } else {
        r1 = c;
        g1 = 0;
        b1 = x;
      }
      palette.add([
        ((r1 + m) * 255).round(),
        ((g1 + m) * 255).round(),
        ((b1 + m) * 255).round(),
      ]);
    }
    return palette;
  }

  // method to dispose of the inference objects from memory
  void dispose() {
    unawaited(_mediaPipeLlm?.close());
    _mediaPipeLlm = null;
    _isolateInterpreter?.close();
    _isolateInterpreter = null;
    _interpreter?.close();
    _interpreter = null;
    modelPipeline = null;
    if (kDebugMode) {
      debugPrint("Inference Object disposed.");
    }
  }

  /// Converts a tflite_flutter [TensorType] to the dtype string used by the pipeline schema.
  String _tensorTypeToDtype(TensorType type) {
    switch (type) {
      case TensorType.float32:
        return 'float32';
      case TensorType.int32:
        return 'int32';
      case TensorType.uint8:
        return 'uint8';
      case TensorType.int8:
        return 'int8';
      default:
        return 'float32';
    }
  }

  /// Builds the output buffer map for [runForMultipleInputs].
  ///
  /// Only called for models whose actual output tensor count equals the number
  /// of pipeline-declared outputs (no extra undeclared tensors). Models with
  /// extra undeclared outputs are handled by the compute() + invoke() path.
  Map<int, Object> _buildOutputBuffers(List<IO> pipelineOutputs) {
    final buffers = <int, Object>{};
    for (int i = 0; i < pipelineOutputs.length; i++) {
      final shape = _getActualOutputShape(i, pipelineOutputs[i].shape);
      final dtype = pipelineOutputs[i].dtype;
      buffers[i] = _createOutputBuffer(shape, dtype);
    }
    return buffers;
  }

  /// Infers the shape of a nested List by walking the first element at each depth.
  /// Returns an empty list if [data] is not a List (e.g. a flat typed buffer).
  List<int> _inferListShape(dynamic data) {
    final dims = <int>[];
    dynamic current = data;
    while (current is List) {
      dims.add((current).length);
      if ((current).isEmpty) break;
      current = (current).first;
    }
    return dims;
  }

  // helper method to create an output buffer, given an output shape and data type
  dynamic _createOutputBuffer(List<int> shape, String dtype) {
    int totalElements = shape.reduce((a, b) => a * b);
    switch (dtype.toLowerCase()) {
      case 'float32':
        return ListShape(List.filled(totalElements, 0.0)).reshape(shape);
      case 'uint8':
      case 'int8':
        return ListShape(List.filled(totalElements, 0)).reshape(shape);
      case 'int32':
        // TFLite Flutter represents int32 tensors as List<int>
        return ListShape(List.filled(totalElements, 0)).reshape(shape);
      default:
        throw Exception("Unsupported output dtype: $dtype");
    }
  }

  // perform a postprocessing step, given a postprocessing block step object
  Future<dynamic> _performPostprocessingStep(
    dynamic processedOutput,
    Map<String, dynamic> outputTensors,
    ProcessingStep step,
  ) async {
    if (kDebugMode) {
      debugPrint("Executing postprocessing step: ${step.step}");
    }

    switch (step.step.toLowerCase()) {
      // applies an activation function to a list of values
      case 'apply_activation':
        // check that the current processed data is a List
        if (processedOutput is! List) {
          throw FormatException(
            "Processed output is not a List, cannot apply activation function.",
          );
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
              debugPrint(
                "Warning: Unsupported activation function: $function, returning input data unchanged.",
              );
            }
        }

        if (kDebugMode) {
          debugPrint("Finished ${step.step} postprocessing step.");
          developer.log(
            "Inspecting ${step.step} postprocessing output:n processedOutput",
          );
          developer.inspect(processedOutput);
        }

      // map raw or activated outputs to a set of labels for classification
      case 'map_labels':
        // check that labels are loaded
        if (_labels == null) {
          debugPrint("Labels not loaded, cannot map labels to output.");
          return processedOutput;
        }

        // check whether task is classification or object detection
        // image classification map_labels takes a List of floats as input
        // object detection map_labels takes a Map<int, List<Map<String, dynamic>>>

        if (processedOutput is Map) {
          debugPrint("Mapping labels assuming object detection task.");

          // grabbing detection class index tensor for label mapping
          String detectionClassTensorName = step.params['class_tensor'];
          dynamic detectionClassTensor =
              outputTensors[detectionClassTensorName];
          if (kDebugMode) {
            developer.log("Inspecting detection class tensor...");
            developer.inspect(detectionClassTensor);
          }

          // loop through detection batches
          for (int i = 0; i < processedOutput.length; i++) {
            int detectionCount = 1;
            // loop through detections
            for (var detectionMap in processedOutput[i]!) {
              debugPrint(
                "Mapping label ${_labels?[detectionMap['original_index']]} to detection $detectionCount",
              );
              detectionMap['label'] =
                  _labels?[detectionClassTensor[0][detectionMap['original_index']]
                      .toInt()];
              detectionCount++;
            }
          }
        } else if (processedOutput is List) {
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
          debugPrint(
            "Checking if processedOutput type = ${processedOutput.runtimeType} is a nested list.",
          );
          if (isNestedList(processedOutput)) {
            debugPrint("Flattening processedOutput nested List.");
            List<dynamic> flattenedProcessedOutput =
                processedOutput.expand((x) => x).toList();
            tempOutput = flattenedProcessedOutput;
          } else {
            tempOutput = processedOutput;
          }
          debugPrint(
            "Map label debug message: tempOutput type = ${tempOutput.runtimeType}",
          );
          for (int i = 0; i < tempOutput.length; i++) {
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
          developer.log(
            "Inspecting ${step.step} postprocessing output: processedOutput",
          );
          developer.inspect(processedOutput);
        }

      // filters object detections by some threshold
      // expects a tensor, or a List<dynamic> in Dart
      case 'filter_by_score':
        // find raw output using score_tensor param
        String scoreTensorName = step.params['score_tensor'];
        scoreTensor = outputTensors[scoreTensorName];
        String numDetectionsTensorName = step.params['num_detections_tensor'];
        dynamic numDetectionsTensor = outputTensors[numDetectionsTensorName];
        final int numDetections =
            (numDetectionsTensor is List
                    ? numDetectionsTensor[0] as num
                    : numDetectionsTensor as num)
                .toInt();

        // set threshold according to YAML file, default to 0.5 if none found
        double threshold =
            (step.params['threshold'] as num?)?.toDouble() ?? 0.5;
        Map<int, List<int>> filteredDetectionIndices = {};
        debugPrint(
          "Filtering ${scoreTensor[0].length} detections with threshold $threshold...",
        );

        // loop through each batch of score tensors
        for (int i = 0; i < scoreTensor.length; i++) {
          filteredDetectionIndices[i] =
              <int>[]; // Initialize the list for each batch
          for (int j = 0; j < numDetections; j++) {
            debugPrint("scoreTensor[$i][$j] = ${scoreTensor[i][j]}");
            debugPrint("threshold = $threshold");
            debugPrint(
              "scoreTensor[$i][$j] > threshold = ${scoreTensor[i][j] > threshold}",
            );
            if (scoreTensor[i][j] > threshold) {
              filteredDetectionIndices[i]!.add(j);
            }
          }
        }

        debugPrint(
          "InferenceService: Filtered ${filteredDetectionIndices[0]!.length} detections above threshold $threshold",
        );
        if (kDebugMode) {
          developer.log(
            "Inspecting ${step.step} postprocessing variable: filteredDetectionIndices",
          );
          developer.inspect(filteredDetectionIndices);
        }
        // This step's output (filteredIndices) becomes processedData for the next step.
        processedOutput = filteredDetectionIndices;

        if (kDebugMode) {
          debugPrint("Finished ${step.step} postprocessing step.");
          developer.log(
            "Inspecting ${step.step} postprocessing output: processedOutput",
          );
          developer.inspect(processedOutput);
        }

      // uses filtered indices and later on the coordinate format config to construct the final detection tensors
      // expected a Map of filtered indices as input, from 'filter_by_score'
      case 'decode_boxes':
        // Expects processedOutput (from previous step) to be the list of filtered indices
        if (processedOutput is! Map<int, List<int>>) {
          throw FormatException(
            "Step 'decode_boxes' expects Map<int, List<int>> (filtered indices) input. Got ${processedOutput.runtimeType}",
          );
        }
        final params = step.params;
        final boxTensorName = params['box_tensor'] as String?;
        if (boxTensorName == null ||
            !outputTensors.containsKey(boxTensorName)) {
          throw Exception(
            "decode_boxes requires a valid 'box_tensor' param pointing to a raw output.",
          );
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
              final box =
                  boxesRaw[i]?[index]
                      .map((val) => (val as num).toDouble())
                      .toList();
              if (box?.length == 4) {
                // Basic validation
                debugPrint("Adding box coordinates: $box");
                decodedData[i]!.add({
                  "original_index":
                      index, // Preserve original index for later mapping
                  "score": scoreTensor[i]?[index],
                  "raw_box":
                      box, // Pass raw normalized box [ymin, xmin, ymax, xmax] or other format
                });
              } else {
                debugPrint(
                  "InferenceService: Warning: Box at index $index does not have 4 coordinates in decode_boxes.",
                );
              }
            } else {
              debugPrint(
                "InferenceService: Warning: Index $index out of bounds for boxesRaw (length ${boxesRaw.length}) in decode_boxes.",
              );
            }
          }
        }

        if (kDebugMode) {
          developer.log(
            "Inspecting ${step.step} postprocessing variable: decodedData",
          );
          developer.inspect(decodedData);
        }
        // This map of a list of maps becomes processedData for the next step
        processedOutput = decodedData;

        if (kDebugMode) {
          debugPrint("Finished ${step.step} postprocessing step.");
          developer.log(
            "Inspecting ${step.step} postprocessing output: processedOutput",
          );
          developer.inspect(processedOutput);
        }

      // autoregressive decode loop or single-pass pass-through
      case 'generate':
        final String mode = step.params['mode'] as String? ?? 'single_pass';
        if (mode == 'single_pass') {
          // Single-pass: model already ran, output tensor is the token ID sequence.
          // Nothing to do here — decode_tokens handles the rest.
          break;
        }

        // Autoregressive: run the model in a loop, appending one token at a time.
        if (_interpreter == null || modelPipeline == null) {
          throw StateError(
            "Cannot run autoregressive generation: interpreter not ready.",
          );
        }
        if (_tokenizer == null) {
          throw StateError(
            "Cannot run autoregressive generation: tokenizer not loaded.",
          );
        }

        final int maxNewTokens =
            (step.params['max_new_tokens'] as num?)?.toInt() ?? 128;
        final double temperature =
            (step.params['temperature'] as num?)?.toDouble() ?? 0.8;
        final bool doSample = step.params['do_sample'] as bool? ?? true;
        final int? eosTokenId = (step.params['eos_token_id'] as num?)?.toInt();
        final double repetitionPenalty =
            (step.params['repetition_penalty'] as num?)?.toDouble() ?? 1.3;

        // Get initial token IDs from the preprocessed input (first text input).
        // _lastPreprocessedInputs is keyed by input tensor name; value is [List<int>].
        List<int> currentIds = [];
        for (final entry in _lastPreprocessedInputs.entries) {
          final val = entry.value;
          if (val is List && val.isNotEmpty && val.first is List) {
            currentIds = List<int>.from(
              (val.first as List).map((e) => (e as num).toInt()),
            );
          } else if (val is List<int>) {
            currentIds = List<int>.from(val);
          }
          if (currentIds.isNotEmpty) break;
        }

        if (currentIds.isEmpty) {
          throw StateError(
            "Autoregressive generation: could not extract input token IDs.",
          );
        }

        // Trim padding from initial input before generating
        final int padId = _tokenizer!.padId;
        currentIds = currentIds.where((id) => id != padId).toList();

        final int expectedSeqLen = _getModelInputSeqLen(
          fallback: currentIds.length,
        );

        if (kDebugMode) {
          debugPrint(
            '[Generate] Dispatching to background isolate: '
            'promptLen=${currentIds.length}, maxNewTokens=$maxNewTokens, '
            'eosId=$eosTokenId, padId=$padId, seqLen=$expectedSeqLen',
          );
          debugPrint(
            '[Generate] Prompt decoded: "${_tokenizer!.decode(currentIds, skipSpecialTokens: false)}"',
          );
        }

        if (!isLocalFile) {
          throw StateError(
            'Autoregressive generation requires a local model file. '
            'Asset-bundled models are not supported for autoregressive text generation.',
          );
        }

        // Run the generate loop in a background Dart isolate so the main
        // thread — and Flutter's rendering loop — are never blocked.
        final List<int> generatedTokens = await compute(
          _runGenerateIsolate,
          _GenerateRequest(
            modelPath: modelPath,
            inputIds: currentIds,
            seqLen: expectedSeqLen,
            padId: padId,
            eosTokenId: eosTokenId,
            maxNewTokens: maxNewTokens,
            temperature: temperature,
            doSample: doSample,
            repetitionPenalty: repetitionPenalty,
          ),
        );

        processedOutput = generatedTokens;
        if (kDebugMode) {
          debugPrint(
            '[Generate] Done: ${generatedTokens.length} tokens. '
            'First 20 IDs: ${generatedTokens.take(20).toList()}',
          );
        }

      // decode a list of token IDs back to a string
      case 'decode_tokens':
        if (_tokenizer == null) {
          throw StateError(
            "Postprocessing step 'decode_tokens' requires a tokenizer, but none was loaded.",
          );
        }
        final bool skipSpecial =
            step.params['skip_special_tokens'] as bool? ?? true;

        // Flatten nested lists to a flat List<int>
        List<int> ids = _flattenToIntList(processedOutput);
        processedOutput = _tokenizer!.decode(
          ids,
          skipSpecialTokens: skipSpecial,
        );

        if (kDebugMode) {
          debugPrint(
            "[InferenceService] Decoded ${ids.length} token IDs to text.",
          );
        }

      case 'decode_segmentation_mask':
        final int numClasses = (step.params['num_classes'] as num).toInt();
        final String? colorMap = step.params['color_map'] as String?;
        final palette = _buildSegmentationPalette(numClasses, colorMap);
        // Supports two output layouts:
        //   [1, H, W, C] — per-class scores (float), argmax required
        //   [1, H, W]    — class indices already argmaxed (int/uint8)
        final tensor = processedOutput as List;
        final batch0 = tensor[0] as List;
        final int H = batch0.length;
        final int W = (batch0[0] as List).length;
        final maskImg = img.Image(width: W, height: H);
        for (int h = 0; h < H; h++) {
          for (int w = 0; w < W; w++) {
            final cell = (batch0[h] as List)[w];
            int bestClass;
            if (cell is List) {
              // [1, H, W, C]: argmax over channel dim
              bestClass = 0;
              double bestScore = double.negativeInfinity;
              for (int c = 0; c < cell.length; c++) {
                final s = (cell[c] as num).toDouble();
                if (s > bestScore) {
                  bestScore = s;
                  bestClass = c;
                }
              }
            } else {
              // [1, H, W]: value is already the class index
              bestClass = (cell as num).toInt();
            }
            final rgb = palette[bestClass.clamp(0, palette.length - 1)];
            maskImg.setPixel(w, h, img.ColorRgb8(rgb[0], rgb[1], rgb[2]));
          }
        }
        processedOutput = maskImg;

      case 'ctc_decode':
        // Input: logits tensor [1, T, vocab_size] from the model output.
        // Algorithm: greedy CTC decode — argmax per timestep, collapse duplicates, remove blanks.
        final int blankId = (step.params['blank_id'] as num?)?.toInt() ?? 0;
        final String wordDelimiter =
            step.params['word_delimiter'] as String? ?? '|';

        List logitsTensor = processedOutput as List;
        // Strip batch dim only for 3-D tensors [1, T, V] → [T, V].
        // For 2-D [T, V] tensors, logitsTensor[0] is a frame (List of numbers)
        // whose first element is a number, not a List — skip the strip.
        if (logitsTensor.isNotEmpty &&
            logitsTensor[0] is List &&
            (logitsTensor[0] as List).isNotEmpty &&
            (logitsTensor[0] as List)[0] is List) {
          logitsTensor = logitsTensor[0] as List;
        }

        // Greedy argmax over vocab dimension for each timestep
        final List<int> rawIds = [];
        for (final timestep in logitsTensor) {
          final frame = timestep as List;
          int bestId = 0;
          double bestScore = double.negativeInfinity;
          for (int v = 0; v < frame.length; v++) {
            final score = (frame[v] as num).toDouble();
            if (score > bestScore) {
              bestScore = score;
              bestId = v;
            }
          }
          rawIds.add(bestId);
        }

        // CTC collapse: remove consecutive duplicates, then remove blanks
        final List<int> collapsedIds = [];
        int? prevId;
        for (final id in rawIds) {
          if (id != prevId) {
            if (id != blankId) collapsedIds.add(id);
            prevId = id;
          }
        }

        // Map token IDs to characters using _labels vocabulary
        final StringBuffer sb = StringBuffer();
        final vocab = _labels ?? [];
        for (final id in collapsedIds) {
          if (id < vocab.length) {
            final token = vocab[id];
            sb.write(token == wordDelimiter ? ' ' : token);
          }
        }
        processedOutput = sb.toString().trim();

      case 'decode_image':
        // Converts a raw pixel tensor [1, H, W, C] (HWC) or [1, C, H, W] (CHW)
        // to an img.Image. Params: channel_format ("HWC"|"CHW"), value_range ("0_1"|"neg1_1"|"0_255").
        final channelFormat =
            (step.params['channel_format'] as String?) ?? 'HWC';
        final valueRange = (step.params['value_range'] as String?) ?? '0_1';

        int scaleToUint8(num raw) {
          final double v = raw.toDouble();
          final double scaled;
          switch (valueRange) {
            case 'neg1_1':
              scaled = (v + 1.0) / 2.0 * 255.0;
            case '0_255':
              scaled = v;
            default: // '0_1'
              scaled = v * 255.0;
          }
          return scaled.round().clamp(0, 255);
        }

        final batch = (processedOutput as List)[0] as List;
        final img.Image decodedImg;

        if (channelFormat == 'CHW') {
          // batch: [C][H][W]
          final int C = batch.length;
          final int H = (batch[0] as List).length;
          final int W = ((batch[0] as List)[0] as List).length;
          decodedImg = img.Image(width: W, height: H);
          for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
              final r = scaleToUint8(((batch[0] as List)[h] as List)[w] as num);
              final g =
                  C > 1
                      ? scaleToUint8(((batch[1] as List)[h] as List)[w] as num)
                      : r;
              final b =
                  C > 2
                      ? scaleToUint8(((batch[2] as List)[h] as List)[w] as num)
                      : r;
              decodedImg.setPixel(w, h, img.ColorRgb8(r, g, b));
            }
          }
        } else {
          // HWC: batch: [H][W][C]
          final int H = batch.length;
          final int W = (batch[0] as List).length;
          decodedImg = img.Image(width: W, height: H);
          for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
              final pixel = (batch[h] as List)[w] as List;
              final r = scaleToUint8(pixel[0] as num);
              final g = pixel.length > 1 ? scaleToUint8(pixel[1] as num) : r;
              final b = pixel.length > 2 ? scaleToUint8(pixel[2] as num) : r;
              decodedImg.setPixel(w, h, img.ColorRgb8(r, g, b));
            }
          }
        }
        processedOutput = decodedImg;

      default:
        if (kDebugMode) {
          debugPrint("Warning: Unsupported postprocessing step: ${step.step}");
        }
    }

    return processedOutput;
  }

  /// Returns the actual output tensor shape from the TFLite interpreter for [index].
  /// Falls back to [fallbackShape] if the interpreter is unavailable.
  List<int> _getActualOutputShape(int index, List<int> fallbackShape) {
    if (_interpreter != null) {
      try {
        final tensors = _interpreter!.getOutputTensors();
        if (index < tensors.length) {
          return List<int>.from(tensors[index].shape);
        }
      } catch (_) {}
    }
    return fallbackShape;
  }

  /// Returns the actual TFLite input tensor shape for [index].
  /// Only returns the shape when all dimensions are positive (i.e. fixed shape).
  /// Falls back to [fallbackShape] for dynamic-shape models.
  List<int> _getActualInputShape(int index, List<int> fallbackShape) {
    if (_interpreter != null) {
      try {
        final tensors = _interpreter!.getInputTensors();
        if (index < tensors.length) {
          final shape = List<int>.from(tensors[index].shape);
          if (shape.every((d) => d > 0)) return shape;
        }
      } catch (_) {}
    }
    return fallbackShape;
  }

  /// Returns the actual TFLite input tensor dtype string for [index],
  /// or an empty string if unavailable.
  String _getActualInputDtype(int index) {
    if (_interpreter != null) {
      try {
        final tensors = _interpreter!.getInputTensors();
        if (index < tensors.length) {
          return _tensorTypeToDtype(tensors[index].type);
        }
      } catch (_) {}
    }
    return '';
  }

  /// Returns the actual input sequence length from the TFLite interpreter.
  /// Falls back to the pipeline-declared shape, then to [fallback].
  int _getModelInputSeqLen({int fallback = 512}) {
    if (_interpreter != null) {
      try {
        final tensors = _interpreter!.getInputTensors();
        if (tensors.isNotEmpty && tensors[0].shape.length >= 2) {
          return tensors[0].shape[1];
        }
      } catch (_) {}
    }
    if (modelPipeline != null &&
        modelPipeline!.inputs.isNotEmpty &&
        modelPipeline!.inputs[0].shape.length >= 2) {
      return modelPipeline!.inputs[0].shape[1];
    }
    return fallback;
  }

  // Flatten any nested list structure to a flat List<int>
  List<int> _flattenToIntList(dynamic value) {
    if (value is List<int>) return value;
    if (value is List) {
      final result = <int>[];
      for (final item in value) {
        result.addAll(_flattenToIntList(item));
      }
      return result;
    }
    if (value is num) return [value.toInt()];
    return [];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Background isolate support for autoregressive generation
// ─────────────────────────────────────────────────────────────────────────────

/// All data needed to run the generate loop in a background isolate.
/// Only sendable types (primitives + integer lists) so it can cross isolate boundaries.
class _GenerateRequest {
  final String modelPath;
  final List<int> inputIds;
  final int seqLen;
  final int padId;
  final int? eosTokenId;
  final int maxNewTokens;
  final double temperature;
  final bool doSample;
  final double repetitionPenalty;

  const _GenerateRequest({
    required this.modelPath,
    required this.inputIds,
    required this.seqLen,
    required this.padId,
    this.eosTokenId,
    required this.maxNewTokens,
    required this.temperature,
    required this.doSample,
    required this.repetitionPenalty,
  });
}

/// Top-level function executed by [compute()].
///
/// Creates a fresh Interpreter inside the background isolate — native objects
/// cannot be shared across isolate boundaries. Runs the full autoregressive
/// loop, closes the interpreter, and returns the generated token IDs.
List<int> _runGenerateIsolate(_GenerateRequest req) {
  // Bounded exp helper — avoids float overflow in softmax.
  double boundedExp(double x) {
    if (x > 88) return 1.5e38;
    if (x < -88) return 0.0;
    return math.exp(x);
  }

  // Build output buffers from the interpreter's actual tensor metadata.
  Map<int, Object> buildBuffers(Interpreter interp) {
    final tensors = interp.getOutputTensors();
    final buffers = <int, Object>{};
    for (int i = 0; i < tensors.length; i++) {
      final shape = List<int>.from(tensors[i].shape);
      final total = shape.reduce((a, b) => a * b);
      buffers[i] =
          tensors[i].type == TensorType.float32
              ? ListShape(List<double>.filled(total, 0.0)).reshape(shape)
              : ListShape(List<int>.filled(total, 0)).reshape(shape);
    }
    return buffers;
  }

  final options = InterpreterOptions()..threads = 4;
  final interpreter = Interpreter.fromFile(
    File(req.modelPath),
    options: options,
  );

  List<int> currentIds = List<int>.from(req.inputIds);
  final List<int> generatedTokens = [];

  for (int step = 0; step < req.maxNewTokens; step++) {
    // Sliding window: keep the most recent seqLen tokens.
    final List<int> contextIds =
        currentIds.length > req.seqLen
            ? currentIds.sublist(currentIds.length - req.seqLen)
            : currentIds;

    // Right-pad so real tokens are at the front; causal attention means pad
    // positions at the back cannot influence earlier positions' outputs.
    final List<int> paddedIds =
        contextIds.length < req.seqLen
            ? [
              ...contextIds,
              ...List.filled(req.seqLen - contextIds.length, req.padId),
            ]
            : List<int>.from(contextIds);

    final Map<int, Object> buffers = buildBuffers(interpreter);
    interpreter.runForMultipleInputs([
      [paddedIds],
    ], buffers);

    // logits shape: [1, seqLen, vocabSize] or [1, vocabSize]
    final dynamic raw = buffers[0];
    final int lastRealPos = contextIds.length - 1;
    List<dynamic> lastLogits;
    if (raw is List && raw.isNotEmpty) {
      final batch = raw[0] as List;
      lastLogits =
          (batch.isNotEmpty && batch[0] is List)
              ? batch[lastRealPos] as List<dynamic>
              : batch.cast<dynamic>();
    } else {
      break;
    }

    // Repetition penalty (HuggingFace convention).
    List<double> logits = lastLogits.map((v) => (v as num).toDouble()).toList();
    if (req.repetitionPenalty != 1.0) {
      final seen = <int>{...currentIds, ...generatedTokens};
      for (final id in seen) {
        if (id >= 0 && id < logits.length) {
          logits[id] =
              logits[id] > 0
                  ? logits[id] / req.repetitionPenalty
                  : logits[id] * req.repetitionPenalty;
        }
      }
    }

    // Greedy argmax or temperature sampling.
    int nextToken;
    if (!req.doSample || req.temperature <= 0.0) {
      nextToken = 0;
      double best = logits[0];
      for (int i = 1; i < logits.length; i++) {
        if (logits[i] > best) {
          best = logits[i];
          nextToken = i;
        }
      }
    } else {
      final scaled = logits.map((v) => v / req.temperature).toList();
      final maxVal = scaled.reduce((a, b) => a > b ? a : b);
      final exps = scaled.map((v) => boundedExp(v - maxVal)).toList();
      final sum = exps.reduce((a, b) => a + b);
      final probs = exps.map((v) => v / sum).toList();
      double r = math.Random().nextDouble();
      nextToken = probs.length - 1;
      for (int i = 0; i < probs.length; i++) {
        r -= probs[i];
        if (r <= 0) {
          nextToken = i;
          break;
        }
      }
    }

    generatedTokens.add(nextToken);
    if (req.eosTokenId != null && nextToken == req.eosTokenId) break;
    currentIds = [...currentIds, nextToken];
  }

  interpreter.close();
  return generatedTokens;
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