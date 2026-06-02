// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'model_providers.dart';

// **************************************************************************
// RiverpodGenerator
// **************************************************************************

// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint, type=warning
/// Singleton provider for the API service

@ProviderFor(apiService)
final apiServiceProvider = ApiServiceProvider._();

/// Singleton provider for the API service

final class ApiServiceProvider
    extends $FunctionalProvider<ApiService, ApiService, ApiService>
    with $Provider<ApiService> {
  /// Singleton provider for the API service
  ApiServiceProvider._()
    : super(
        from: null,
        argument: null,
        retry: null,
        name: r'apiServiceProvider',
        isAutoDispose: true,
        dependencies: null,
        $allTransitiveDependencies: null,
      );

  @override
  String debugGetCreateSourceHash() => _$apiServiceHash();

  @$internal
  @override
  $ProviderElement<ApiService> $createElement($ProviderPointer pointer) =>
      $ProviderElement(pointer);

  @override
  ApiService create(Ref ref) {
    return apiService(ref);
  }

  /// {@macro riverpod.override_with_value}
  Override overrideWithValue(ApiService value) {
    return $ProviderOverride(
      origin: this,
      providerOverride: $SyncValueProvider<ApiService>(value),
    );
  }
}

String _$apiServiceHash() => r'd76dea2a3d4afd840c19952cd59fe889ce36151d';

/// Fetch all models from the backend

@ProviderFor(allModels)
final allModelsProvider = AllModelsProvider._();

/// Fetch all models from the backend

final class AllModelsProvider
    extends
        $FunctionalProvider<
          AsyncValue<List<MLModel>>,
          List<MLModel>,
          FutureOr<List<MLModel>>
        >
    with $FutureModifier<List<MLModel>>, $FutureProvider<List<MLModel>> {
  /// Fetch all models from the backend
  AllModelsProvider._()
    : super(
        from: null,
        argument: null,
        retry: null,
        name: r'allModelsProvider',
        isAutoDispose: true,
        dependencies: null,
        $allTransitiveDependencies: null,
      );

  @override
  String debugGetCreateSourceHash() => _$allModelsHash();

  @$internal
  @override
  $FutureProviderElement<List<MLModel>> $createElement(
    $ProviderPointer pointer,
  ) => $FutureProviderElement(pointer);

  @override
  FutureOr<List<MLModel>> create(Ref ref) {
    return allModels(ref);
  }
}

String _$allModelsHash() => r'b00c9a2e48bf5567d008e26ae2dfb6d5c084878f';

/// Fetch a specific model by ID

@ProviderFor(modelById)
final modelByIdProvider = ModelByIdFamily._();

/// Fetch a specific model by ID

final class ModelByIdProvider
    extends $FunctionalProvider<AsyncValue<MLModel>, MLModel, FutureOr<MLModel>>
    with $FutureModifier<MLModel>, $FutureProvider<MLModel> {
  /// Fetch a specific model by ID
  ModelByIdProvider._({
    required ModelByIdFamily super.from,
    required String super.argument,
  }) : super(
         retry: null,
         name: r'modelByIdProvider',
         isAutoDispose: true,
         dependencies: null,
         $allTransitiveDependencies: null,
       );

  @override
  String debugGetCreateSourceHash() => _$modelByIdHash();

  @override
  String toString() {
    return r'modelByIdProvider'
        ''
        '($argument)';
  }

  @$internal
  @override
  $FutureProviderElement<MLModel> $createElement($ProviderPointer pointer) =>
      $FutureProviderElement(pointer);

  @override
  FutureOr<MLModel> create(Ref ref) {
    final argument = this.argument as String;
    return modelById(ref, argument);
  }

  @override
  bool operator ==(Object other) {
    return other is ModelByIdProvider && other.argument == argument;
  }

  @override
  int get hashCode {
    return argument.hashCode;
  }
}

String _$modelByIdHash() => r'a7bc877a29303e7b9da3ea57e2b6f2310796857b';

/// Fetch a specific model by ID

final class ModelByIdFamily extends $Family
    with $FunctionalFamilyOverride<FutureOr<MLModel>, String> {
  ModelByIdFamily._()
    : super(
        retry: null,
        name: r'modelByIdProvider',
        dependencies: null,
        $allTransitiveDependencies: null,
        isAutoDispose: true,
      );

  /// Fetch a specific model by ID

  ModelByIdProvider call(String modelId) =>
      ModelByIdProvider._(argument: modelId, from: this);

  @override
  String toString() => r'modelByIdProvider';
}

/// Get all models that have at least one supported version
/// Uses backend filtering via supported_only=true parameter

@ProviderFor(supportedModels)
final supportedModelsProvider = SupportedModelsProvider._();

/// Get all models that have at least one supported version
/// Uses backend filtering via supported_only=true parameter

final class SupportedModelsProvider
    extends
        $FunctionalProvider<
          AsyncValue<List<MLModel>>,
          List<MLModel>,
          FutureOr<List<MLModel>>
        >
    with $FutureModifier<List<MLModel>>, $FutureProvider<List<MLModel>> {
  /// Get all models that have at least one supported version
  /// Uses backend filtering via supported_only=true parameter
  SupportedModelsProvider._()
    : super(
        from: null,
        argument: null,
        retry: null,
        name: r'supportedModelsProvider',
        isAutoDispose: true,
        dependencies: null,
        $allTransitiveDependencies: null,
      );

  @override
  String debugGetCreateSourceHash() => _$supportedModelsHash();

  @$internal
  @override
  $FutureProviderElement<List<MLModel>> $createElement(
    $ProviderPointer pointer,
  ) => $FutureProviderElement(pointer);

  @override
  FutureOr<List<MLModel>> create(Ref ref) {
    return supportedModels(ref);
  }
}

String _$supportedModelsHash() => r'6ba324d98b0e97bac3e3250789d886ea288923bb';

/// Fetch all versions for a specific model

@ProviderFor(modelVersions)
final modelVersionsProvider = ModelVersionsFamily._();

/// Fetch all versions for a specific model

final class ModelVersionsProvider
    extends
        $FunctionalProvider<
          AsyncValue<List<ModelVersion>>,
          List<ModelVersion>,
          FutureOr<List<ModelVersion>>
        >
    with
        $FutureModifier<List<ModelVersion>>,
        $FutureProvider<List<ModelVersion>> {
  /// Fetch all versions for a specific model
  ModelVersionsProvider._({
    required ModelVersionsFamily super.from,
    required String super.argument,
  }) : super(
         retry: null,
         name: r'modelVersionsProvider',
         isAutoDispose: true,
         dependencies: null,
         $allTransitiveDependencies: null,
       );

  @override
  String debugGetCreateSourceHash() => _$modelVersionsHash();

  @override
  String toString() {
    return r'modelVersionsProvider'
        ''
        '($argument)';
  }

  @$internal
  @override
  $FutureProviderElement<List<ModelVersion>> $createElement(
    $ProviderPointer pointer,
  ) => $FutureProviderElement(pointer);

  @override
  FutureOr<List<ModelVersion>> create(Ref ref) {
    final argument = this.argument as String;
    return modelVersions(ref, argument);
  }

  @override
  bool operator ==(Object other) {
    return other is ModelVersionsProvider && other.argument == argument;
  }

  @override
  int get hashCode {
    return argument.hashCode;
  }
}

String _$modelVersionsHash() => r'7faf6bf6aa8bca7368c805fa884e5735f5d23a9d';

/// Fetch all versions for a specific model

final class ModelVersionsFamily extends $Family
    with $FunctionalFamilyOverride<FutureOr<List<ModelVersion>>, String> {
  ModelVersionsFamily._()
    : super(
        retry: null,
        name: r'modelVersionsProvider',
        dependencies: null,
        $allTransitiveDependencies: null,
        isAutoDispose: true,
      );

  /// Fetch all versions for a specific model

  ModelVersionsProvider call(String modelId) =>
      ModelVersionsProvider._(argument: modelId, from: this);

  @override
  String toString() => r'modelVersionsProvider';
}

/// Get only supported versions for a model

@ProviderFor(supportedVersions)
final supportedVersionsProvider = SupportedVersionsFamily._();

/// Get only supported versions for a model

final class SupportedVersionsProvider
    extends
        $FunctionalProvider<
          AsyncValue<List<ModelVersion>>,
          List<ModelVersion>,
          FutureOr<List<ModelVersion>>
        >
    with
        $FutureModifier<List<ModelVersion>>,
        $FutureProvider<List<ModelVersion>> {
  /// Get only supported versions for a model
  SupportedVersionsProvider._({
    required SupportedVersionsFamily super.from,
    required String super.argument,
  }) : super(
         retry: null,
         name: r'supportedVersionsProvider',
         isAutoDispose: true,
         dependencies: null,
         $allTransitiveDependencies: null,
       );

  @override
  String debugGetCreateSourceHash() => _$supportedVersionsHash();

  @override
  String toString() {
    return r'supportedVersionsProvider'
        ''
        '($argument)';
  }

  @$internal
  @override
  $FutureProviderElement<List<ModelVersion>> $createElement(
    $ProviderPointer pointer,
  ) => $FutureProviderElement(pointer);

  @override
  FutureOr<List<ModelVersion>> create(Ref ref) {
    final argument = this.argument as String;
    return supportedVersions(ref, argument);
  }

  @override
  bool operator ==(Object other) {
    return other is SupportedVersionsProvider && other.argument == argument;
  }

  @override
  int get hashCode {
    return argument.hashCode;
  }
}

String _$supportedVersionsHash() => r'b4df60528d775144ab32501b653b5de930724be0';

/// Get only supported versions for a model

final class SupportedVersionsFamily extends $Family
    with $FunctionalFamilyOverride<FutureOr<List<ModelVersion>>, String> {
  SupportedVersionsFamily._()
    : super(
        retry: null,
        name: r'supportedVersionsProvider',
        dependencies: null,
        $allTransitiveDependencies: null,
        isAutoDispose: true,
      );

  /// Get only supported versions for a model

  SupportedVersionsProvider call(String modelId) =>
      SupportedVersionsProvider._(argument: modelId, from: this);

  @override
  String toString() => r'supportedVersionsProvider';
}

/// Fetch a specific model version

@ProviderFor(versionById)
final versionByIdProvider = VersionByIdFamily._();

/// Fetch a specific model version

final class VersionByIdProvider
    extends
        $FunctionalProvider<
          AsyncValue<ModelVersion>,
          ModelVersion,
          FutureOr<ModelVersion>
        >
    with $FutureModifier<ModelVersion>, $FutureProvider<ModelVersion> {
  /// Fetch a specific model version
  VersionByIdProvider._({
    required VersionByIdFamily super.from,
    required String super.argument,
  }) : super(
         retry: null,
         name: r'versionByIdProvider',
         isAutoDispose: true,
         dependencies: null,
         $allTransitiveDependencies: null,
       );

  @override
  String debugGetCreateSourceHash() => _$versionByIdHash();

  @override
  String toString() {
    return r'versionByIdProvider'
        ''
        '($argument)';
  }

  @$internal
  @override
  $FutureProviderElement<ModelVersion> $createElement(
    $ProviderPointer pointer,
  ) => $FutureProviderElement(pointer);

  @override
  FutureOr<ModelVersion> create(Ref ref) {
    final argument = this.argument as String;
    return versionById(ref, argument);
  }

  @override
  bool operator ==(Object other) {
    return other is VersionByIdProvider && other.argument == argument;
  }

  @override
  int get hashCode {
    return argument.hashCode;
  }
}

String _$versionByIdHash() => r'9217bd1d74558be629c7d3a39ea7734e1c284d10';

/// Fetch a specific model version

final class VersionByIdFamily extends $Family
    with $FunctionalFamilyOverride<FutureOr<ModelVersion>, String> {
  VersionByIdFamily._()
    : super(
        retry: null,
        name: r'versionByIdProvider',
        dependencies: null,
        $allTransitiveDependencies: null,
        isAutoDispose: true,
      );

  /// Fetch a specific model version

  VersionByIdProvider call(String versionId) =>
      VersionByIdProvider._(argument: versionId, from: this);

  @override
  String toString() => r'versionByIdProvider';
}

/// Track the currently selected model

@ProviderFor(SelectedModel)
final selectedModelProvider = SelectedModelProvider._();

/// Track the currently selected model
final class SelectedModelProvider
    extends $NotifierProvider<SelectedModel, MLModel?> {
  /// Track the currently selected model
  SelectedModelProvider._()
    : super(
        from: null,
        argument: null,
        retry: null,
        name: r'selectedModelProvider',
        isAutoDispose: true,
        dependencies: null,
        $allTransitiveDependencies: null,
      );

  @override
  String debugGetCreateSourceHash() => _$selectedModelHash();

  @$internal
  @override
  SelectedModel create() => SelectedModel();

  /// {@macro riverpod.override_with_value}
  Override overrideWithValue(MLModel? value) {
    return $ProviderOverride(
      origin: this,
      providerOverride: $SyncValueProvider<MLModel?>(value),
    );
  }
}

String _$selectedModelHash() => r'53c86c07b453acf9acc1e582c3e1e5d05498b19d';

/// Track the currently selected model

abstract class _$SelectedModel extends $Notifier<MLModel?> {
  MLModel? build();
  @$mustCallSuper
  @override
  void runBuild() {
    final ref = this.ref as $Ref<MLModel?, MLModel?>;
    final element =
        ref.element
            as $ClassProviderElement<
              AnyNotifier<MLModel?, MLModel?>,
              MLModel?,
              Object?,
              Object?
            >;
    element.handleCreate(ref, build);
  }
}

/// Track the currently selected model version

@ProviderFor(SelectedModelVersion)
final selectedModelVersionProvider = SelectedModelVersionProvider._();

/// Track the currently selected model version
final class SelectedModelVersionProvider
    extends $NotifierProvider<SelectedModelVersion, ModelVersion?> {
  /// Track the currently selected model version
  SelectedModelVersionProvider._()
    : super(
        from: null,
        argument: null,
        retry: null,
        name: r'selectedModelVersionProvider',
        isAutoDispose: true,
        dependencies: null,
        $allTransitiveDependencies: null,
      );

  @override
  String debugGetCreateSourceHash() => _$selectedModelVersionHash();

  @$internal
  @override
  SelectedModelVersion create() => SelectedModelVersion();

  /// {@macro riverpod.override_with_value}
  Override overrideWithValue(ModelVersion? value) {
    return $ProviderOverride(
      origin: this,
      providerOverride: $SyncValueProvider<ModelVersion?>(value),
    );
  }
}

String _$selectedModelVersionHash() =>
    r'1b07543103e81a7977d7c1199670493b94f186d8';

/// Track the currently selected model version

abstract class _$SelectedModelVersion extends $Notifier<ModelVersion?> {
  ModelVersion? build();
  @$mustCallSuper
  @override
  void runBuild() {
    final ref = this.ref as $Ref<ModelVersion?, ModelVersion?>;
    final element =
        ref.element
            as $ClassProviderElement<
              AnyNotifier<ModelVersion?, ModelVersion?>,
              ModelVersion?,
              Object?,
              Object?
            >;
    element.handleCreate(ref, build);
  }
}

/// Download model file (tflite binary)

@ProviderFor(downloadModelFile)
final downloadModelFileProvider = DownloadModelFileFamily._();

/// Download model file (tflite binary)

final class DownloadModelFileProvider
    extends
        $FunctionalProvider<
          AsyncValue<List<int>>,
          List<int>,
          FutureOr<List<int>>
        >
    with $FutureModifier<List<int>>, $FutureProvider<List<int>> {
  /// Download model file (tflite binary)
  DownloadModelFileProvider._({
    required DownloadModelFileFamily super.from,
    required String super.argument,
  }) : super(
         retry: null,
         name: r'downloadModelFileProvider',
         isAutoDispose: true,
         dependencies: null,
         $allTransitiveDependencies: null,
       );

  @override
  String debugGetCreateSourceHash() => _$downloadModelFileHash();

  @override
  String toString() {
    return r'downloadModelFileProvider'
        ''
        '($argument)';
  }

  @$internal
  @override
  $FutureProviderElement<List<int>> $createElement($ProviderPointer pointer) =>
      $FutureProviderElement(pointer);

  @override
  FutureOr<List<int>> create(Ref ref) {
    final argument = this.argument as String;
    return downloadModelFile(ref, argument);
  }

  @override
  bool operator ==(Object other) {
    return other is DownloadModelFileProvider && other.argument == argument;
  }

  @override
  int get hashCode {
    return argument.hashCode;
  }
}

String _$downloadModelFileHash() => r'f94299d80b5638885fc23ac0da9109df041ea288';

/// Download model file (tflite binary)

final class DownloadModelFileFamily extends $Family
    with $FunctionalFamilyOverride<FutureOr<List<int>>, String> {
  DownloadModelFileFamily._()
    : super(
        retry: null,
        name: r'downloadModelFileProvider',
        dependencies: null,
        $allTransitiveDependencies: null,
        isAutoDispose: true,
      );

  /// Download model file (tflite binary)

  DownloadModelFileProvider call(String downloadUrl) =>
      DownloadModelFileProvider._(argument: downloadUrl, from: this);

  @override
  String toString() => r'downloadModelFileProvider';
}

/// Download asset file (labels, etc)

@ProviderFor(downloadAssetFile)
final downloadAssetFileProvider = DownloadAssetFileFamily._();

/// Download asset file (labels, etc)

final class DownloadAssetFileProvider
    extends
        $FunctionalProvider<
          AsyncValue<List<int>>,
          List<int>,
          FutureOr<List<int>>
        >
    with $FutureModifier<List<int>>, $FutureProvider<List<int>> {
  /// Download asset file (labels, etc)
  DownloadAssetFileProvider._({
    required DownloadAssetFileFamily super.from,
    required String super.argument,
  }) : super(
         retry: null,
         name: r'downloadAssetFileProvider',
         isAutoDispose: true,
         dependencies: null,
         $allTransitiveDependencies: null,
       );

  @override
  String debugGetCreateSourceHash() => _$downloadAssetFileHash();

  @override
  String toString() {
    return r'downloadAssetFileProvider'
        ''
        '($argument)';
  }

  @$internal
  @override
  $FutureProviderElement<List<int>> $createElement($ProviderPointer pointer) =>
      $FutureProviderElement(pointer);

  @override
  FutureOr<List<int>> create(Ref ref) {
    final argument = this.argument as String;
    return downloadAssetFile(ref, argument);
  }

  @override
  bool operator ==(Object other) {
    return other is DownloadAssetFileProvider && other.argument == argument;
  }

  @override
  int get hashCode {
    return argument.hashCode;
  }
}

String _$downloadAssetFileHash() => r'1c9a57785735f2a92dfd316eb3c30948ca689223';

/// Download asset file (labels, etc)

final class DownloadAssetFileFamily extends $Family
    with $FunctionalFamilyOverride<FutureOr<List<int>>, String> {
  DownloadAssetFileFamily._()
    : super(
        retry: null,
        name: r'downloadAssetFileProvider',
        dependencies: null,
        $allTransitiveDependencies: null,
        isAutoDispose: true,
      );

  /// Download asset file (labels, etc)

  DownloadAssetFileProvider call(String downloadUrl) =>
      DownloadAssetFileProvider._(argument: downloadUrl, from: this);

  @override
  String toString() => r'downloadAssetFileProvider';
}
