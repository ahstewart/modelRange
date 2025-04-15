import 'package:flutter/material.dart';
import 'package:yaml/yaml.dart';
import 'dart:async';
import 'dart:io';
import 'package:json_annotation/json_annotation.dart';

part 'data_models.g.dart';


@JsonSerializable()
class Pipeline {
  // define top-level components of the model pipeline configuration file
  final List<MetaData> metadata;
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
  })

  // factory to generate Dart object from decoded JSON
  factory Pipeline.fromJson(Map<String, dynamic> json) => _$PipelineFromJson(json);

  Map<String, dynamic> toJson() => _$PipelineToJson(this);

}

// metadata component data model
@JsonSerializable()
class Metadata {
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

  Metadata({
    required this.schema_version,
    required this.model_name,
    required this.model_version,
    required this.model_task,
    required this.framework,
    required this.source_repository,
  })

  factory Metadata.fromJson(Map<String, dynamic> json) => _$Metadata(json);
  Map<String, dynamic> toJson() => _$Metadata(this);
}






Future<Map> parseMetadata(String metadataPath) async {
  // get string from metadata file
  String metadataContents = await File(metadataPath).readAsString();

  // parse the string using the yaml package and return the parsed map
  return loadYaml(metadataContents) as Map;
}