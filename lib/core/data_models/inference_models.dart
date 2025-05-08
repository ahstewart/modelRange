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

