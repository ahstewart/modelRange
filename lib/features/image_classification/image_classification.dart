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
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';
import '../../core/services/inferenceService.dart';
import '../../core/data_models/pipeline.dart';
 

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
  // instantiate the model inference object
  late final InferenceService inferenceObject;
  bool _isLoading = false;
  File? _selectedImage;
  img.Image? _decodedImage;
  List<dynamic>? _recognitions;
  // define the input map
  Map<String, dynamic> inputMap = {};
  // define the final inference results map
  Map<String, dynamic> inferenceResults = {};
  // number of results displayed on the screen, default to 3
  int numResults = 3;

  @override
  void initState() {
    super.initState();
    inferenceObject = InferenceService(
      modelPath: widget.modelName,
      pipelinePath: widget.pipelinePath
    );
    if (kDebugMode) {
      debugPrint("Initializing Image Classification Widget");
    }
  }
    

  @override dispose() {
    inferenceObject.dispose();
    super.dispose();
  }


// enable the selection of images
  Future<void> _pickImage(ImageSource source) async {
    if (_isLoading) return; // can't pick an image if inference is currently happening

    try {
      debugPrint("Picking image...");
      final ImagePicker picker = ImagePicker();
      final XFile? image = await picker.pickImage(source: source);
      debugPrint("Image picked.");
      debugPrint("Decoding image so it can fed into model...");
      File imageFile = File(image!.path);
      _decodedImage = img.decodeImage(await imageFile.readAsBytes());

      setState(() {
        _selectedImage = imageFile;
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
      for (var input in inferenceObject.modelPipeline!.inputs) {
        inputMap[input.name] = _decodedImage;
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

    // check that the inference resulted in actual outputs
    if (inferenceResults.isNotEmpty && inferenceResults.values.first != null) {
      // use the inference results to update the UI
      List<Map<String, dynamic>> recongnitions = inferenceResults.values.first;
      // reorder the final recognitions by highest probability first
      recongnitions.removeWhere((r) => r['confidence'] == null);
      recongnitions.sort((a,b) => (b['confidence'] as double).compareTo(a['confidence'] as double));

      setState(() {
        _recognitions = recongnitions;
      });
    }
    else {
      setState(() {
        _recognitions = [{"label": "Error running model", "confidence": 0.0}];
      });
    }

    // try to set the numResults to a custom value, if it's included in the pipeline YAML
    debugPrint("Trying to set the numResults to a custom value, if it's included in the pipeline YAML.");
    try {
      // look for posprocessing block with a map_labels step
      for (ProcessingBlock block in inferenceObject.modelPipeline!.postprocessing) {
        for (ProcessingStep step in block.steps) {
          if (step.step == 'map_labels') {
            numResults = step.params['top_k'];
          }
        }
      }
    } 
    catch (e) {
      debugPrint("Error setting numResults, maybe it's not in the pipeline YAML. Using default value of 3.");
    }

  }


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
                              ..._recognitions!.take(numResults).map((rec) {
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
  