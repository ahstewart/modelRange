import 'dart:core';
import 'package:flutter/material.dart';
import 'package:yaml/yaml.dart';
import 'dart:async';
import 'dart:io';
import 'package:json_annotation/json_annotation.dart';

part 'data_models.g.dart';


@JsonSerializable()
class Pipeline {
  // define top-level components of the model pipeline configuration file
  final List<ModelMetadata> metadata;
  final List<IO> inputs;
  final List<IO> outputs;
  final List<ProcessingBlock> preprocessing;
  final List<ProcessingBlock> postprocessing;

  Pipeline({
    required this.metadata,
    required this.inputs,
    required this.outputs,
    required this.preprocessing,
    required this.postprocessing,
  });

  // factory to generate Dart object from decoded JSON
  factory Pipeline.fromJson(Map<String, dynamic> json) => _$PipelineFromJson(json);

  Map<String, dynamic> toJson() => _$PipelineToJson(this);

}

// metadata component data model
@JsonSerializable()
class ModelMetadata {
  // metadata fields in YAML
  @JsonKey(required: true)
  final String schema_version;
  @JsonKey(required: true)
  final String model_name;
  @JsonKey(defaultValue: '')
  final String model_version;
  @JsonKey(required: true)
  final String model_task;
  @JsonKey(required: true)
  final String framework;
  @JsonKey(required: true)
  final String source_repository;

  ModelMetadata({
    required this.schema_version,
    required this.model_name,
    required this.model_version,
    required this.model_task,
    required this.framework,
    required this.source_repository,
  });

  factory ModelMetadata.fromJson(Map<String, dynamic> json) => _$ModelMetadataFromJson(json);
  Map<String, dynamic> toJson() => _$ModelMetadataToJson(this);
}


// IO component data model
@JsonSerializable()
class IO {
  // input/output fields in YAML
  @JsonKey(required: true)
  final String name;
  @JsonKey(defaultValue: [])
  final List<int> shape;
  @JsonKey(required: true)
  final String dtype;
  @JsonKey(defaultValue: '')
  final String description;

  IO({
    required this.name,
    required this.shape,
    required this.dtype,
    required this.description,
  });

  factory IO.fromJson(Map<String, dynamic> json) => _$IOFromJson(json);
  Map<String, dynamic> toJson() => _$IOToJson(this);
}


// processing block component data model
@JsonSerializable()
class ProcessingBlock {
  // processing block fields in YAML
  @JsonKey(defaultValue: '')
  final String input_name;
  @JsonKey(defaultValue: '')
  final String output_name;
  @JsonKey(defaultValue: '')
  final String expects_type;
  @JsonKey(defaultValue: '')
  final String interpretation;
  @JsonKey(required: true)
  final List<String> source_tensors;
  final List<ProcessingStep> steps;

  ProcessingBlock({
    required this.input_name,
    required this.output_name,
    required this.expects_type,
    required this.interpretation,
    required this.source_tensors,
    required this.steps,
  });

  factory ProcessingBlock.fromJson(Map<String, dynamic> json) => _$ProcessingBlockFromJson(json);
  Map<String, dynamic> toJson() => _$ProcessingBlockToJson(this);
}


// processing step component data model
@JsonSerializable()
class ProcessingStep {
  // processing step fields in YAML
  @JsonKey(required: true)
  final String step;
  final Map<String, dynamic> params;

  ProcessingStep({
    required this.step,
    required this.params,
  });

  factory ProcessingStep.fromJson(Map<String, dynamic> json) => _$ProcessingStepFromJson(json);
  Map<String, dynamic> toJson() => _$ProcessingStepToJson(this);
}


Future<Map> parseMetadata(String metadataPath) async {
  // get string from metadata file
  String metadataContents = await File(metadataPath).readAsString();

  // parse the string using the yaml package and return the parsed map
  return loadYaml(metadataContents) as Map;
}