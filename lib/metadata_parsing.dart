import 'package:yaml/yaml.dart';
import 'dart:async';
import 'dart:io';

Future<Map> parseMetadata(String metadataPath) async {
  // get string from metadata file
  String metadataContents = await File(metadataPath).readAsString();

  // parse the string using the yaml package and return the parsed map
  return loadYaml(metadataContents) as Map;
}