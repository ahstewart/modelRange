// this file will house the various functions and classes that will enable model inference
// primary tasks handled will be model pre-processing, inference, and post-processing

import 'dart:ffi';
import 'dart:io';
import 'dart:math' as Math;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:model_range/inference/pipeline_schema.txt';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';
import '../inference/pipeline.dart';
 

// file purpose: this is the main widget for image classification, it will facilitate the capturing of images and running the inference of a model
// on the captured image. The widget will also display the results of the inference.
var modelNameReal = 'assets/mobilenetv4_conv_small.e2400_r224_in1k_float32.tflite';
var pipelinePathReal = 'assets/metadata/mobilenet_imageclass.yaml';
var labelName = 'assets/imagenet_classes.txt';


// to generalize the image classification task, we need the image classification widget to accept the model metadata 

class ImageClassificationWidget extends ConsumerStatefulWidget {
  final String pipelinePath;
  final String modelName;
  
  const ImageClassificationWidget({
    super.key, 
    required this.modelName,
    required this.pipelinePath,
  });

  @override
  ConsumerState<ImageClassificationWidget> createState() => _ImageClassificationWidgetState();
}

class _ImageClassificationWidgetState extends ConsumerState<ImageClassificationWidget> {
/*
  // initialize interpreter, labels, etc
  Interpreter? _interpreter;
  List<String>? _labels;

  // preprocessing configs
  final int _inputWidth = 224;
  final int _inputHeight = 224;
  final double _mean = 127.5;
  final double _std = 127.5;
*/

  // instantiate the model inference object
  late final InferenceObject inferenceObject;
  bool _isLoading = false;
  File? _selectedImage;
  List<dynamic>? _recognitions;
  // define the input map
  Map<String, dynamic> inputMap = {};
  // define the final inference results map
  Map<String, dynamic> inferenceResults = {};

  @override
  void initState() {
    super.initState();
    inferenceObject = InferenceObject(
      modelPath: widget.modelName,
      pipelinePath: widget.pipelinePath
    );
    if (kDebugMode) {
      debugPrint("Initializing Image Classification Widget");
    }
  }
    /*
    // initialize model interpreter based on "framework" parameter in model metadata
    // valid vaLues are "tflite", 
    if (widget.metadata['framework'] == "tflite") {
      _initializeInterpreter(widget.metadata['source_repository']);
    }
    else {
      if (kDebugMode) {
        debugPrint("Error loading model: Invalid 'framework' parameter");
      }
    }
    // load in labels using metadata
    try {
      _loadLabels(getLabelPath(widget.metadata));
    }
    catch (e) {
      if (kDebugMode) {
        debugPrint("Error loading labels: $e");
      }
    }
    
  }
*/

  @override dispose() {
    inferenceObject.dispose();
    super.dispose();
  }

/*
    // method to traverse the metadata Map to fetch the label path (label_url)
  String getLabelPath(Map metadata) {
   try {
    final postprocessing = metadata['postprocessing'] as List;
    for (var step in postprocessing) {
      final steps = step['steps'] as List;
      for (var substep in steps) {
        if (substep['step'] == 'map_labels') {
          return substep['params']['labels_url'] as String;
        }
      }
    }
      throw Exception("Could not find 'labels_url' in model metadata postprocessing step.");
    }
    catch (e) {
    if (kDebugMode) {
      debugPrint('Error getting label path: $e');
    }
    return "Error getting label URL";
  }
}


  // create interpreter to load the model
  Future<void> _initializeInterpreter(String modelPath) async {
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

  // load classification labels
  Future<void> _loadLabels(String labelPath) async {
    try {
      final classificationLabels = await rootBundle.loadString(labelPath);
      _labels = classificationLabels.split('\n').map((label) => label.trim()).where((label) => label.isNotEmpty).toList();
      if (kDebugMode) {
        debugPrint("Successfully loaded ${_labels?.length} labels.");
      }
    }
    catch (e) {
      if (kDebugMode) {
        debugPrint("Failed fetching classification labels: $e");
      }
    }
  }

  // apply the softmax function on a given list of floats
  List<double> applySoftmax(List<double> raw_inputs) {
    double max_input = raw_inputs.reduce((a,b) => a > b ? a : b);
    List<double> exps = raw_inputs.map((input) => Math.exp(input - max_input)).toList();
    double sumExps = exps.reduce((a, b) => a + b);
    return exps.map((exp) => exp / sumExps).toList();
  }

*/

// enable the selection of images
  Future<void> _pickImage(ImageSource source) async {
    if (_isLoading) return; // can't pick an image if inference is currently happening

    try {
      final ImagePicker picker = ImagePicker();
      final XFile? image = await picker.pickImage(source: source);

      if (image == null) return;

      setState(() {
        _selectedImage = File(image.path);
        _recognitions = null;
      });
    }

    catch (e) {
      if (kDebugMode) {
        debugPrint("Failed to pick image: $e");
      }
    }

    // run inference on the selected image
    try {
      // check that the inference object is initialized and ready
      if (!inferenceObject.isReady) {
        if (kDebugMode) {
          debugPrint("Could not run inference on image, interpreter or labels haven't been loaded.");
        }
        return;
      }

      setState(() {
        _isLoading = true;
      });

      // create input data map, keyed by input name
      for (input in inferenceObject.modelPipeline!.inputs) {
        inputMap[input.input_name] = _selectedImage;
      }

      // run inference on selected image
      inferenceResults = await inferenceObject.performInference(inputMap);
    }

    catch (e) {
      if (kDebugMode) {
        debugPrint("Error running inference: $e");
      }
      setState(() {
        _recognitions = [{"label": "Error", "confidence": "Could not get confidence"}];
      });
    }

    finally {
      setState(() {
        _isLoading = false;
      });
    }

    // use the inference results to update the UI
    List<Map<String, dynamic>> recongnitions = inferenceResults.values.first;
    // reorder the final recognitions by highest probability first
    recongnitions.sort((a,b) => (b['confidence'] as double).compareTo(a['confidence'] as double));

    setState(() {
      _recognitions = recongnitions;
    });
  }

/*
  // run inference on image
  Future<void> _runInference(File imageFile) async {
    if (_interpreter == null || _labels == null) {
      if (kDebugMode) {
        debugPrint("Could not run inference on image, interpreter or labels haven't been loaded.");
      }
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      // prep image - Preprocessing
      // load image into memory
      img.Image? image = img.decodeImage(await imageFile.readAsBytes());
      if (image == null) throw Exception("Could not decode image");
      // resize image to sizes defined in the model metadata schema (MMS)
      
      img.Image resizedImage = img.copyResize(image, width: _inputWidth, height: _inputHeight);
      // normalize image values around mean and std defined in the MMS
      var imageBytes = resizedImage.getBytes(order: img.ChannelOrder.rgb);
      var inputBytes = Float32List(_inputHeight * _inputWidth * 3);
      // var inputIndex = 0;
      for (int i = 0; i <imageBytes.length; i +=3) {
        inputBytes[i] = (imageBytes[i] - _mean) / _std;
        inputBytes[i+1] = (imageBytes[i+1] - _mean) / _std;
        inputBytes[i+2] = (imageBytes[i+2] - _mean) / _std;
      }
      // reshape normalized image
      final input = inputBytes.reshape([1, _inputWidth, _inputHeight, 3]);

      // define output tensor
      final outputShape = [1, _labels!.length]; // assuming the output tensor matches the list of labels
      final output = List.filled(outputShape.reduce((a,b) => a*b), 0.0).reshape(outputShape);

      // using the TFLite interpreter, run the inference
      _interpreter!.run(input, output);

      // postprocessing, understand the outputted results and display them
      final List<double> outputScores = output[0] as List<double>;

      // apply softmax to recognitions
      final List<double> outputScores_softmax = applySoftmax(outputScores);

      // create map of labels: probabilities (will softmax later)
      List<Map<String, dynamic>> recongnitions = [];
      for (int i=0; i<outputScores_softmax.length; i++) {
        recongnitions.add({
          "index": i,
          "label": _labels![i],
          "confidence": outputScores_softmax[i],
        });
      }

      // reorder the final recognitions by highest probability first
      recongnitions.sort((a,b) => (b['confidence'] as double).compareTo(a['confidence'] as double));

      setState(() {
        _recognitions = recongnitions;
      });


    }

    catch (e) {
      if (kDebugMode) {
        debugPrint("Image classification inference failed: $e");
      }
      setState(() {
        _recognitions = [{"label": "Error", "confidence": "Could not get confidence"}];
      });
    }

    finally {
      setState(() {
        _isLoading = false;
      });
    }
  }
*/

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Image Classification'),
      ),
      body: SingleChildScrollView( // Allow scrolling if content overflows
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              const SizedBox(height: 20),
              // Display selected image
              _selectedImage == null
                  ? Container(
                      height: 250,
                      width: 250,
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.grey),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Center(child: Text('No image selected.')),
                    )
                  : Container(
                      margin: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.grey),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: ClipRRect( // Clip image to rounded corners
                        borderRadius: BorderRadius.circular(11),
                        child: Image.file(
                          _selectedImage!,
                          width: 300,
                          height: 300,
                          fit: BoxFit.cover,
                        ),
                      ),
                    ),
              const SizedBox(height: 20),
              // Buttons to pick image
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  ElevatedButton.icon(
                    icon: const Icon(Icons.image),
                    label: const Text('Gallery'),
                    onPressed: () => _pickImage(ImageSource.gallery),
                  ),
                  ElevatedButton.icon(
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                    onPressed: () => _pickImage(ImageSource.camera),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              // Display loading indicator or results
              _isLoading
                  ? const CircularProgressIndicator()
                  : _recognitions != null
                      ? Padding(
                          padding: const EdgeInsets.all(15.0),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text(
                                'Results:',
                                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                              ),
                              const SizedBox(height: 10),
                              // Display top 3 results (or fewer if less results)
                              ..._recognitions!.take(3).map((rec) {
                                return Text(
                                  '${rec['label']} (${(rec['confidence'] * 100).toStringAsFixed(1)}%)',
                                  style: const TextStyle(fontSize: 16),
                                );
                              }).toList(),
                            ],
                          ),
                        )
                      : Container(), // Show nothing if no results yet
              const SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }
 }
  