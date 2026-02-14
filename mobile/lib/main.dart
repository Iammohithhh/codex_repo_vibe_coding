import 'package:flutter/material.dart';
import 'screens/home_screen.dart';
import 'screens/project_detail_screen.dart';
import 'screens/visual_map_screen.dart';
import 'screens/settings_screen.dart';
import 'services/api_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Firebase initialization would go here:
  // await Firebase.initializeApp();
  runApp(const Paper2ProductApp());
}

class Paper2ProductApp extends StatelessWidget {
  const Paper2ProductApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Paper2Product',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.dark(
          primary: const Color(0xFF6366F1),
          secondary: const Color(0xFF818CF8),
          surface: const Color(0xFF1A1D27),
          background: const Color(0xFF0F1117),
          error: const Color(0xFFEF4444),
          onPrimary: Colors.white,
          onSurface: const Color(0xFFE4E6F0),
        ),
        scaffoldBackgroundColor: const Color(0xFF0F1117),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF1A1D27),
          elevation: 0,
          centerTitle: true,
        ),
        cardTheme: CardTheme(
          color: const Color(0xFF1A1D27),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
            side: const BorderSide(color: Color(0xFF2D3148)),
          ),
        ),
      ),
      initialRoute: '/',
      routes: {
        '/': (context) => const MainNavigation(),
        '/project': (context) => const ProjectDetailScreen(),
        '/visual-map': (context) => const VisualMapScreen(),
        '/settings': (context) => const SettingsScreen(),
      },
    );
  }
}

class MainNavigation extends StatefulWidget {
  const MainNavigation({super.key});

  @override
  State<MainNavigation> createState() => _MainNavigationState();
}

class _MainNavigationState extends State<MainNavigation> {
  int _selectedIndex = 0;

  final _pages = const [
    HomeScreen(),
    ProjectListScreen(),
    NotificationsScreen(),
    SettingsScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _pages[_selectedIndex],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (i) => setState(() => _selectedIndex = i),
        backgroundColor: const Color(0xFF1A1D27),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.home_outlined), selectedIcon: Icon(Icons.home), label: 'Home'),
          NavigationDestination(icon: Icon(Icons.folder_outlined), selectedIcon: Icon(Icons.folder), label: 'Projects'),
          NavigationDestination(icon: Icon(Icons.notifications_outlined), selectedIcon: Icon(Icons.notifications), label: 'Alerts'),
          NavigationDestination(icon: Icon(Icons.settings_outlined), selectedIcon: Icon(Icons.settings), label: 'Settings'),
        ],
      ),
    );
  }
}

class ProjectListScreen extends StatelessWidget {
  const ProjectListScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Projects')),
      body: FutureBuilder<List<dynamic>>(
        future: ApiService.instance.listProjects(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          }
          final projects = snapshot.data ?? [];
          if (projects.isEmpty) {
            return const Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.science_outlined, size: 64, color: Color(0xFF8B8FA3)),
                  SizedBox(height: 16),
                  Text('No projects yet', style: TextStyle(color: Color(0xFF8B8FA3), fontSize: 16)),
                  SizedBox(height: 8),
                  Text('Create a project from the web app', style: TextStyle(color: Color(0xFF8B8FA3), fontSize: 13)),
                ],
              ),
            );
          }
          return ListView.builder(
            padding: const EdgeInsets.all(16),
            itemCount: projects.length,
            itemBuilder: (context, i) {
              final p = projects[i];
              return Card(
                margin: const EdgeInsets.only(bottom: 12),
                child: ListTile(
                  title: Text(p['ingest']?['title'] ?? 'Untitled'),
                  subtitle: Text(p['status'] ?? '', style: const TextStyle(fontSize: 12)),
                  trailing: Text('${((p['confidence_score'] ?? 0) * 100).toStringAsFixed(0)}%',
                    style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                  onTap: () => Navigator.pushNamed(context, '/project', arguments: p),
                ),
              );
            },
          );
        },
      ),
    );
  }
}

class NotificationsScreen extends StatelessWidget {
  const NotificationsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Notifications')),
      body: const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.notifications_none, size: 64, color: Color(0xFF8B8FA3)),
            SizedBox(height: 16),
            Text('No notifications', style: TextStyle(color: Color(0xFF8B8FA3), fontSize: 16)),
            SizedBox(height: 8),
            Text('Build notifications will appear here', style: TextStyle(color: Color(0xFF8B8FA3), fontSize: 13)),
          ],
        ),
      ),
    );
  }
}
