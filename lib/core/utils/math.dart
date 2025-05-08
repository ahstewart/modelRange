import 'dart:math' as Math;
import 'package:flutter/material.dart';
import 'data_types.dart';
  
  
  // apply the softmax function on a given list of floats
  List<dynamic> applySoftmax(List<dynamic> rawInputs) {
    if (rawInputs.any((x) => x == null)) {
      throw Exception("Null value found in inputs to softmax.");
    }
    debugPrint("Applying softmax activation");
    debugPrint("Checking if input is a nested list");
    List<dynamic> softmaxInput = [];
    if (isNestedList(rawInputs)) {
      debugPrint("Input is nested list, flattening list.");
      debugPrint("Input type = ${rawInputs.runtimeType}");
      softmaxInput = rawInputs.expand((x) => x).cast<dynamic>().toList();
    }
    else {
      softmaxInput = rawInputs;
    }
    double max_input = softmaxInput.reduce((a,b) => a > b ? a : b);
    List<double> exps = softmaxInput.map((input) => Math.exp(input - max_input)).toList();
    double sumExps = exps.reduce((a, b) => a + b);
    return exps.map((exp) => exp / sumExps).toList();
  }

