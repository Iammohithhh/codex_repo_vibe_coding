import 'package:flutter/material.dart';
import '../services/api_service.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Paper2Product'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => (context as Element).markNeedsBuild(),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Hero card
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF6366F1), Color(0xFF818CF8)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(16),
              ),
              child: const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('AI Research OS', style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.white)),
                  SizedBox(height: 8),
                  Text('Transform papers into production-ready products',
                    style: TextStyle(fontSize: 14, color: Colors.white70)),
                ],
              ),
            ),
            const SizedBox(height: 24),

            // Quick actions
            const Text('Quick Actions', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
            const SizedBox(height: 12),
            Row(
              children: [
                _ActionCard(icon: Icons.add_circle_outline, label: 'New Project', color: const Color(0xFF6366F1),
                  onTap: () => _showComingSoon(context, 'Create projects from the web app')),
                const SizedBox(width: 12),
                _ActionCard(icon: Icons.science_outlined, label: 'Experiments', color: const Color(0xFF22C55E),
                  onTap: () => _showComingSoon(context, 'Run experiments from project detail')),
                const SizedBox(width: 12),
                _ActionCard(icon: Icons.approval_outlined, label: 'Reviews', color: const Color(0xFFF59E0B),
                  onTap: () => _showComingSoon(context, 'Review projects from project detail')),
              ],
            ),
            const SizedBox(height: 24),

            // Recent projects
            const Text('Recent Projects', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
            const SizedBox(height: 12),
            FutureBuilder<List<dynamic>>(
              future: ApiService.instance.listProjects(),
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(child: Padding(
                    padding: EdgeInsets.all(32),
                    child: CircularProgressIndicator(),
                  ));
                }
                if (snapshot.hasError) {
                  return Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        children: [
                          const Icon(Icons.cloud_off, size: 40, color: Color(0xFF8B8FA3)),
                          const SizedBox(height: 8),
                          Text('Cannot connect to server', style: TextStyle(color: Colors.grey[400])),
                          const SizedBox(height: 4),
                          Text('Configure server URL in Settings', style: TextStyle(color: Colors.grey[600], fontSize: 12)),
                        ],
                      ),
                    ),
                  );
                }
                final projects = snapshot.data ?? [];
                if (projects.isEmpty) {
                  return const Card(
                    child: Padding(
                      padding: EdgeInsets.all(24),
                      child: Center(child: Text('No projects yet. Create one from the web app.',
                        style: TextStyle(color: Color(0xFF8B8FA3)))),
                    ),
                  );
                }
                return Column(
                  children: projects.take(5).map<Widget>((p) => Card(
                    margin: const EdgeInsets.only(bottom: 8),
                    child: ListTile(
                      leading: CircleAvatar(
                        backgroundColor: const Color(0xFF6366F1).withOpacity(0.2),
                        child: const Icon(Icons.description, color: Color(0xFF6366F1)),
                      ),
                      title: Text(p['ingest']?['title'] ?? 'Untitled', maxLines: 1, overflow: TextOverflow.ellipsis),
                      subtitle: Text(p['status'] ?? '', style: const TextStyle(fontSize: 12)),
                      trailing: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Text('${((p['confidence_score'] ?? 0) * 100).toStringAsFixed(0)}%',
                            style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 18)),
                          const Text('confidence', style: TextStyle(fontSize: 10, color: Color(0xFF8B8FA3))),
                        ],
                      ),
                      onTap: () => Navigator.pushNamed(context, '/project', arguments: p),
                    ),
                  )).toList(),
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  void _showComingSoon(BuildContext context, String msg) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }
}

class _ActionCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback onTap;

  const _ActionCard({required this.icon, required this.label, required this.color, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: color.withOpacity(0.1),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: color.withOpacity(0.3)),
          ),
          child: Column(
            children: [
              Icon(icon, color: color, size: 28),
              const SizedBox(height: 8),
              Text(label, style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w500)),
            ],
          ),
        ),
      ),
    );
  }
}
