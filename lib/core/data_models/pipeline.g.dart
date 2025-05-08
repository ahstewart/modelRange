// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'pipeline.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

Pipeline _$PipelineFromJson(Map<String, dynamic> json) => Pipeline(
  metadata:
      (json['metadata'] as List<dynamic>)
          .map((e) => ModelMetadata.fromJson(e as Map<String, dynamic>))
          .toList(),
  inputs:
      (json['inputs'] as List<dynamic>)
          .map((e) => IO.fromJson(e as Map<String, dynamic>))
          .toList(),
  outputs:
      (json['outputs'] as List<dynamic>)
          .map((e) => IO.fromJson(e as Map<String, dynamic>))
          .toList(),
  preprocessing:
      (json['preprocessing'] as List<dynamic>)
          .map((e) => ProcessingBlock.fromJson(e as Map<String, dynamic>))
          .toList(),
  postprocessing:
      (json['postprocessing'] as List<dynamic>)
          .map((e) => ProcessingBlock.fromJson(e as Map<String, dynamic>))
          .toList(),
);

Map<String, dynamic> _$PipelineToJson(Pipeline instance) => <String, dynamic>{
  'metadata': instance.metadata,
  'inputs': instance.inputs,
  'outputs': instance.outputs,
  'preprocessing': instance.preprocessing,
  'postprocessing': instance.postprocessing,
};

ModelMetadata _$ModelMetadataFromJson(Map<String, dynamic> json) {
  $checkKeys(
    json,
    requiredKeys: const [
      'schema_version',
      'model_name',
      'model_task',
      'framework',
      'source_repository',
    ],
  );
  return ModelMetadata(
    schema_version: json['schema_version'] as String,
    model_name: json['model_name'] as String,
    model_version: json['model_version'] as String? ?? '',
    model_task: json['model_task'] as String,
    framework: json['framework'] as String,
    source_repository: json['source_repository'] as String,
  );
}

Map<String, dynamic> _$ModelMetadataToJson(ModelMetadata instance) =>
    <String, dynamic>{
      'schema_version': instance.schema_version,
      'model_name': instance.model_name,
      'model_version': instance.model_version,
      'model_task': instance.model_task,
      'framework': instance.framework,
      'source_repository': instance.source_repository,
    };

IO _$IOFromJson(Map<String, dynamic> json) {
  $checkKeys(json, requiredKeys: const ['name', 'dtype']);
  return IO(
    name: json['name'] as String,
    shape:
        (json['shape'] as List<dynamic>?)
            ?.map((e) => (e as num).toInt())
            .toList() ??
        [],
    dtype: json['dtype'] as String,
    description: json['description'] as String? ?? '',
  );
}

Map<String, dynamic> _$IOToJson(IO instance) => <String, dynamic>{
  'name': instance.name,
  'shape': instance.shape,
  'dtype': instance.dtype,
  'description': instance.description,
};

ProcessingBlock _$ProcessingBlockFromJson(Map<String, dynamic> json) =>
    ProcessingBlock(
      input_name: json['input_name'] as String? ?? '',
      output_name: json['output_name'] as String? ?? '',
      expects_type: json['expects_type'] as String? ?? '',
      interpretation: json['interpretation'] as String? ?? '',
      source_tensors:
          (json['source_tensors'] as List<dynamic>?)
              ?.map((e) => e as String)
              .toList() ??
          [],
      steps:
          (json['steps'] as List<dynamic>?)
              ?.map((e) => ProcessingStep.fromJson(e as Map<String, dynamic>))
              .toList() ??
          [],
    );

Map<String, dynamic> _$ProcessingBlockToJson(ProcessingBlock instance) =>
    <String, dynamic>{
      'input_name': instance.input_name,
      'output_name': instance.output_name,
      'expects_type': instance.expects_type,
      'interpretation': instance.interpretation,
      'source_tensors': instance.source_tensors,
      'steps': instance.steps,
    };

ProcessingStep _$ProcessingStepFromJson(Map<String, dynamic> json) {
  $checkKeys(json, requiredKeys: const ['step']);
  return ProcessingStep(
    step: json['step'] as String,
    params: json['params'] as Map<String, dynamic>,
  );
}

Map<String, dynamic> _$ProcessingStepToJson(ProcessingStep instance) =>
    <String, dynamic>{'step': instance.step, 'params': instance.params};
