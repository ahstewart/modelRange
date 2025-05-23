import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
//import 'package:freezed_annotation/freezed_annotation.dart';
//import 'package:flutter/foundation.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';
import 'features/image_classification/image_classification.dart';
import 'core/services/inferenceService.dart';
import 'core/data_models/pipeline.dart';
import 'package:yaml/yaml.dart';
import 'dart:convert';
import 'features/object_detection/object_detection.dart';


// optional: Since our Person class is serializable, we must add this line.
// But if Person was not serializable, we could skip it.
//part 'main.g.dart';

void main() {
  runApp(ProviderScope(child: const MyApp()));
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pocket AI',
      theme: ThemeData(
      primarySwatch: Colors.indigo,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: const Color.fromARGB(255, 23, 148, 12)),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.indigo[700],
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
            textStyle: const TextStyle(fontSize: 16),
          ),
        ),
      ),
      home: const ModelList(title: 'Pocket AI'),
    );
  }
}


// a Model class using freezed
class Model {
  final dynamic name;
  final dynamic description;
  final dynamic pipelineTag;
  final dynamic likes;


  const Model({required this.name, 
                required this.description,
                required this.pipelineTag, 
                required this.likes,
              }
  );
  
  //factory Model.fromJson(Map<String, Object?> json) => _$ModelFromJson(json);

}

final selectedModelProvider = StateProvider<Model>((ref) {
  Model selectedModel = Model(name: '', description: '', pipelineTag: '', likes: 0);
  return selectedModel;
});

// riverpod provider for the model class
final modelProvider = Provider<List<Model>>((ref) {
  return const [
    Model(
      name: 'yourmom/Moball_TFLite_2244',
      description: 'A small MobileNetV4 model for image classification',
      pipelineTag: 'image-classification',
      likes: 10,),
    Model(
      name: 'astewart/sweetmodel',
      description: '',
      pipelineTag: 'text-to-image',
      likes: 4,),
    Model(
      name: 'someone/lamemodel',
      description: '',
      pipelineTag: 'text-classification',
      likes: 0,),
    Model(
      name: 'byoussef/MobileNetV4_Conv_Small_TFLite_224',
      description: 'Image classification model optimized for mobile',
      pipelineTag: 'image-classification',
      likes: 1,),
  ];
});


// Widget containing the model tile list
class Models extends ConsumerWidget {
  const Models({super.key});

  static final Map model1 = {
  'name': 'yourmom/Moball_TFLite_224',
  'description': '',
  'pipeline_tag': 'text-to-text',
  'likes': 10
  };

  static final Map model3 = {
  'name': 'astewart/sweetmodel',
  'description': '',
  'pipeline_tag': 'text-to-image',
  'likes': 4
  };

  static final Map model2 = {
  'name': 'someone/lamemodel',
  'description': '',
  'pipeline_tag': 'text-classification',
  'likes': 0
  };

   static final Map model4 = {
  'name': 'byoussef/MobileNetV4_Conv_Small_TFLite_224',
  'description': 'Image classification model optimized for mobile',
  'pipeline_tag': 'image-classification',
  'likes': 1
  };

  

  //final List modelList = [model1, model2, model3];

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final modelList = ref.watch(modelProvider);

    return Wrap(
      spacing: 16.0,
      runSpacing: 16.0,
      children: [
        for (Model m in modelList)
          SizedBox(
            width: 300, // Set a fixed width for each card to mimic grid tiles
            child: Card(
              child: ListTile(
                title: Text(m.name),
                subtitle: Text(m.pipelineTag),
                leading: Icon(Icons.image),
                onTap: () {
                  ref.read(selectedModelProvider.notifier).state = m;
                  ref.read(selectedIndexProvider.notifier).state = 1;
                },
              ),
            ),
          ),
      ],
    );
  }
}



// widget for the model range
class ModelRange extends ConsumerWidget {
  const ModelRange({super.key});

  dynamic _convertYamlToJson(dynamic yaml) {
    if (yaml is YamlMap) {
      return yaml.map((k, v) => MapEntry(k.toString(), _convertYamlToJson(v)));
    }
    if (yaml is YamlList) {
      return yaml.map((e) => _convertYamlToJson(e)).toList();
    }
    return yaml;
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final selectedModel = ref.watch(selectedModelProvider);
    
    // check if no model has been selected
    if ((selectedModel.name == '') && (selectedModel.pipelineTag == '') && (selectedModel.description == 0)) {
      return Text('Select a model to check out the range!');
    }

    // check if the selected model is the one we've built inference for
    else if (selectedModel.name == "byoussef/MobileNetV4_Conv_Small_TFLite_224") {
      String metadataPath = 'assets/mobilenet_imageclass.yaml';
      String modelPath = 'assets/mobilenetv4_conv_small.e2400_r224_in1k_float32.tflite';
      //final yamlMap = loadYaml(metadataPath);
      //final jsonMap = _convertYamlToJson(yamlMap);
      return ImageClassificationWidget(
        modelName: modelPath,
        pipelinePath: metadataPath,
      );
    }

      // check if the selected model is the one we've built inference for
    else if (selectedModel.name == "someone/lamemodel") {
      String metadataPath = 'assets/mobilenet_objectdetect.yaml';
      String modelPath = 'assets/ssd_mobilenet_v1_objectdetection.tflite';
      //final yamlMap = loadYaml(metadataPath);
      //final jsonMap = _convertYamlToJson(yamlMap);
      return ObjectDetectionWidget(
        modelName: modelPath,
        pipelinePath: metadataPath,
      );
    }
    
    //if model has been selected, show the model range
    return Column(
        children: [
          ElevatedButton(
            onPressed: () {
              ref.read(selectedIndexProvider.notifier).state = 0;
            },
            child: Text('Back to Model List Yo!'),
          ),
          Text("You're looking at the model range for: "),
          Text('Model Name: ${selectedModel.name}'),
          Text('Pipeline Tag: ${selectedModel.pipelineTag}'),
          Text('Likes: ${selectedModel.likes}'),
        ],
      );
    }
  }


// widget for the user 
class Profile extends StatelessWidget {
  const Profile({super.key});

  @override
  Widget build(BuildContext context) {
    return Text('This is where the user profile will show up');
  }
}

// provider for selected page
final selectedIndexProvider = StateProvider<int>((ref) {
  return 0;
});


class ModelList extends ConsumerStatefulWidget {
  const ModelList({super.key, required this.title});

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".

  final String title;

  @override
  ConsumerState<ModelList> createState() => _ModelList();
}

class _ModelList extends ConsumerState<ModelList> {
  late List<Widget> pages;

  @override
  void initState() {
    super.initState();
    pages = [
      Models(),
      ModelRange(),
      Profile(),
    ];
  }

  @override
  Widget build(BuildContext context) {
    var _selectedIndex = ref.watch(selectedIndexProvider);
    // This method is rerun every time setState is called, for instance as done
    // by the _incrementCounter method above.
    //
    // The Flutter framework has been optimized to make rerunning build methods
    // fast, so that you can just rebuild anything that needs updating rather
    // than having to individually change instances of widgets.
    return Scaffold(
      appBar: AppBar(
        // TRY THIS: Try changing the color here to a specific color (to
        // Colors.amber, perhaps?) and trigger a hot reload to see the AppBar
        // change color while the other colors stay the same.
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        // Here we take the value from the MyHomePage object that was created by
        // the App.build method, and use it to set our appbar title.
        title: Center(child: Text(widget.title)),
      ),
      body: Center(
        // Center is a layout widget. It takes a single child and positions it
        // in the middle of the parent.
        child: Column(
          // Column is also a layout widget. It takes a list of children and
          // arranges them vertically. By default, it sizes itself to fit its
          // children horizontally, and tries to be as tall as its parent.
          //
          // Column has various properties to control how it sizes itself and
          // how it positions its children. Here we use mainAxisAlignment to
          // center the children vertically; the main axis here is the vertical
          // axis because Columns are vertical (the cross axis would be
          // horizontal).
          //
          // TRY THIS: Invoke "debug painting" (choose the "Toggle Debug Paint"
          // action in the IDE, or press "p" in the console), to see the
          // wireframe for each widget.
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Expanded(child: pages[_selectedIndex],
          ),
            
            // This is the nav bar
            SafeArea(child: NavigationBar(destinations: [
                NavigationDestination(
                  icon: Icon(Icons.home),
                  label: 'Home',
                ),
                NavigationDestination(
                  icon: Icon(Icons.science),
                  label: 'Range',
                ),
                NavigationDestination(
                  icon: Icon(Icons.person),
                  label: 'Profile',
                ),
              ], 
              selectedIndex: _selectedIndex,
              onDestinationSelected: (int index) {
                setState(() {
                  ref.read(selectedIndexProvider.notifier).state = index;
                });
              },
            ))
          ],
        ),
      ), // This trailing comma makes auto-formatting nicer for build methods.
    );
  }
}
