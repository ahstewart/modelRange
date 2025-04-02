import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
//import 'package:freezed_annotation/freezed_annotation.dart';
//import 'package:flutter/foundation.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'main.freezed.dart';
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
      title: 'Flutter Demo',
      theme: ThemeData(
        useMaterial3: true,
        // This is the theme of your application.
        //
        // TRY THIS: Try running your application with "flutter run". You'll see
        // the application has a purple toolbar. Then, without quitting the app,
        // try changing the seedColor in the colorScheme below to Colors.green
        // and then invoke "hot reload" (save your changes or press the "hot
        // reload" button in a Flutter-supported IDE, or press "r" if you used
        // the command line to start the app).
        //
        // Notice that the counter didn't reset back to zero; the application
        // state is not lost during the reload. To reset the state, use hot
        // restart instead.
        //
        // This works for code too, not just values: Most code changes can be
        // tested with just a hot reload.
        colorScheme: ColorScheme.fromSeed(seedColor: const Color.fromARGB(255, 12, 21, 148)),
      ),
      home: const ModelList(title: 'Model Range'),
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
      name: 'byoussef/MobileNetV4_Conv_Small_TFLite_224',
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
  ];
});


// Widget containing the model tile list
class Models extends ConsumerWidget {
  const Models({super.key});

  static final Map model1 = {
  'name': 'byoussef/MobileNetV4_Conv_Small_TFLite_224',
  'description': '',
  'pipeline_tag': 'image-classification',
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

  //final List modelList = [model1, model2, model3];

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final modelList = ref.watch(modelProvider);

    return GridView(gridDelegate: SliverGridDelegateWithMaxCrossAxisExtent(maxCrossAxisExtent: 400),
              children:
                [for (Model m in modelList)
                  Card(
                    child: ListTile(
                      //selected: _selected,
                      title: Text(m.name),
                      subtitle: Text(m.pipelineTag),
                      leading: Icon(Icons.image),
                      onTap: () {
                            ref.read(selectedModelProvider.notifier).state = m;
                            ref.read(selectedIndexProvider.notifier).state = 1;
                            }
                    ),
                  ),
                ],
    );
  }
}


// widget for the model range
class ModelRange extends ConsumerWidget {
  const ModelRange({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final selectedModel = ref.watch(selectedModelProvider);
    
    // check if no model has been selected
    if ((selectedModel.name == '') && (selectedModel.pipelineTag == '') && (selectedModel.description == 0)) {
      return Text('Select a model to check out the range!');
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
