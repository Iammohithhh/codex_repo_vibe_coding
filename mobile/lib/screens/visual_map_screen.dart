import 'package:flutter/material.dart';
import '../services/api_service.dart';

class VisualMapScreen extends StatefulWidget {
  const VisualMapScreen({super.key});

  @override
  State<VisualMapScreen> createState() => _VisualMapScreenState();
}

class _VisualMapScreenState extends State<VisualMapScreen> {
  Map<String, dynamic>? project;
  Map<String, dynamic>? visualData;
  bool loading = true;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (project == null) {
      project = ModalRoute.of(context)?.settings.arguments as Map<String, dynamic>?;
      if (project != null) _loadVisualGraph();
    }
  }

  Future<void> _loadVisualGraph() async {
    try {
      final data = await ApiService.instance.getVisualGraph(project!['id']);
      setState(() { visualData = data; loading = false; });
    } catch (e) {
      setState(() { loading = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Visual Maps')),
      body: loading
        ? const Center(child: CircularProgressIndicator())
        : visualData == null
          ? const Center(child: Text('Could not load visual data'))
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Knowledge graph nodes
                  const Text('Knowledge Graph', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
                  const SizedBox(height: 12),
                  ..._buildGraphNodes(),
                  const SizedBox(height: 24),

                  // Data flow timeline
                  const Text('Data Flow Timeline', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
                  const SizedBox(height: 12),
                  ..._buildTimeline(),
                  const SizedBox(height: 24),

                  // Failure mode map
                  const Text('Failure Mode Map', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
                  const SizedBox(height: 12),
                  ..._buildFailureModes(),
                ],
              ),
            ),
    );
  }

  List<Widget> _buildGraphNodes() {
    final graph = visualData?['architecture_graph'] as Map<String, dynamic>? ?? {};
    final nodes = graph['nodes'] as List? ?? [];
    final typeColors = {
      'problem': const Color(0xFFEF4444),
      'method': const Color(0xFF3B82F6),
      'equation': const Color(0xFFA855F7),
      'dataset': const Color(0xFF22C55E),
      'metric': const Color(0xFF22C55E),
      'architecture': const Color(0xFFF59E0B),
      'claim': const Color(0xFF6366F1),
    };

    return nodes.map<Widget>((n) {
      final type = n['type'] ?? 'default';
      final color = typeColors[type] ?? Colors.grey;
      return Card(
        margin: const EdgeInsets.only(bottom: 6),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
          side: BorderSide(color: color.withOpacity(0.4)),
        ),
        child: ListTile(
          dense: true,
          leading: Container(
            width: 8, height: 8,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          title: Text(n['label'] ?? '', style: const TextStyle(fontSize: 13)),
          trailing: Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
            decoration: BoxDecoration(color: color.withOpacity(0.15), borderRadius: BorderRadius.circular(4)),
            child: Text(type, style: TextStyle(fontSize: 10, color: color)),
          ),
        ),
      );
    }).toList();
  }

  List<Widget> _buildTimeline() {
    final timeline = visualData?['data_flow_timeline'] as List? ?? [];
    return timeline.map<Widget>((step) => Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Column(children: [
            Container(
              width: 28, height: 28,
              decoration: BoxDecoration(color: const Color(0xFF6366F1), borderRadius: BorderRadius.circular(14)),
              child: Center(child: Text('${step['step']}', style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold))),
            ),
            if (step != timeline.last) Container(width: 2, height: 20, color: const Color(0xFF2D3148)),
          ]),
          const SizedBox(width: 12),
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(step['name'] ?? '', style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14)),
            Text(step['description'] ?? '', style: const TextStyle(fontSize: 12, color: Color(0xFF8B8FA3))),
          ])),
        ],
      ),
    )).toList();
  }

  List<Widget> _buildFailureModes() {
    final modes = visualData?['failure_mode_map'] as List? ?? [];
    final severityColors = {
      'high': const Color(0xFFEF4444),
      'medium': const Color(0xFFF59E0B),
      'low': const Color(0xFF22C55E),
    };

    return modes.map<Widget>((f) {
      final severity = f['severity'] ?? 'medium';
      final color = severityColors[severity] ?? Colors.grey;
      return Card(
        margin: const EdgeInsets.only(bottom: 8),
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              Text(f['scenario'] ?? '', style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14)),
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(color: color.withOpacity(0.15), borderRadius: BorderRadius.circular(4)),
                child: Text(severity, style: TextStyle(fontSize: 10, color: color, fontWeight: FontWeight.bold)),
              ),
            ]),
            const SizedBox(height: 4),
            Text('Impact: ${f['impact'] ?? ''}', style: const TextStyle(fontSize: 12, color: Color(0xFF8B8FA3))),
            Text('Mitigation: ${f['mitigation'] ?? ''}', style: const TextStyle(fontSize: 12, color: Color(0xFF22C55E))),
          ]),
        ),
      );
    }).toList();
  }
}
