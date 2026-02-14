import 'package:flutter/material.dart';
import '../services/api_service.dart';

class ProjectDetailScreen extends StatefulWidget {
  const ProjectDetailScreen({super.key});

  @override
  State<ProjectDetailScreen> createState() => _ProjectDetailScreenState();
}

class _ProjectDetailScreenState extends State<ProjectDetailScreen> with SingleTickerProviderStateMixin {
  late TabController _tabController;
  Map<String, dynamic>? project;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 5, vsync: this);
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    project ??= ModalRoute.of(context)?.settings.arguments as Map<String, dynamic>?;
  }

  @override
  Widget build(BuildContext context) {
    if (project == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Project')),
        body: const Center(child: Text('No project data')),
      );
    }

    final spec = project!['paper_spec'] as Map<String, dynamic>? ?? {};
    final repro = project!['reproducibility'] as Map<String, dynamic>? ?? {};
    final confidence = (project!['confidence_score'] ?? 0).toDouble();

    return Scaffold(
      appBar: AppBar(
        title: Text(project!['ingest']?['title'] ?? 'Project', maxLines: 1, overflow: TextOverflow.ellipsis),
        bottom: TabBar(
          controller: _tabController,
          isScrollable: true,
          tabs: const [
            Tab(text: 'Overview'),
            Tab(text: 'Spec'),
            Tab(text: 'Reproducibility'),
            Tab(text: 'Agents'),
            Tab(text: 'Actions'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _buildOverview(spec, repro, confidence),
          _buildSpec(spec),
          _buildReproducibility(repro),
          _buildAgents(),
          _buildActions(),
        ],
      ),
    );
  }

  Widget _buildOverview(Map<String, dynamic> spec, Map<String, dynamic> repro, double confidence) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          Row(
            children: [
              _StatCard(value: '${(confidence * 100).toStringAsFixed(0)}%', label: 'Confidence',
                color: confidence >= 0.7 ? const Color(0xFF22C55E) : const Color(0xFFF59E0B)),
              const SizedBox(width: 12),
              _StatCard(value: '${((repro['overall'] ?? 0) * 100).toStringAsFixed(0)}%', label: 'Reproducibility',
                color: (repro['overall'] ?? 0) >= 0.5 ? const Color(0xFF22C55E) : const Color(0xFFEF4444)),
            ],
          ),
          const SizedBox(height: 16),
          Card(child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Problem', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 14, color: Color(0xFF8B8FA3))),
                const SizedBox(height: 4),
                Text(spec['problem'] ?? '', style: const TextStyle(fontSize: 14)),
                const SizedBox(height: 16),
                const Text('Method', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 14, color: Color(0xFF8B8FA3))),
                const SizedBox(height: 4),
                Text(spec['method'] ?? '', style: const TextStyle(fontSize: 14)),
              ],
            ),
          )),
          const SizedBox(height: 12),
          Card(child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Datasets & Metrics', style: TextStyle(fontWeight: FontWeight.w600)),
                const SizedBox(height: 8),
                Wrap(spacing: 6, runSpacing: 6, children: [
                  ...(spec['datasets'] as List? ?? []).map((d) => Chip(label: Text(d.toString(), style: const TextStyle(fontSize: 12)),
                    backgroundColor: const Color(0xFF3B82F6).withOpacity(0.2))),
                  ...(spec['metrics'] as List? ?? []).map((m) => Chip(label: Text(m.toString(), style: const TextStyle(fontSize: 12)),
                    backgroundColor: const Color(0xFF22C55E).withOpacity(0.2))),
                ]),
              ],
            ),
          )),
        ],
      ),
    );
  }

  Widget _buildSpec(Map<String, dynamic> spec) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _section('Key Equations', (spec['key_equations'] as List? ?? []).map((eq) =>
            Card(child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(eq['raw'] ?? eq.toString(), style: const TextStyle(fontFamily: 'monospace', fontSize: 13)),
                if (eq is Map && eq['intuition'] != null) ...[
                  const SizedBox(height: 4),
                  Text(eq['intuition'], style: const TextStyle(fontSize: 12, color: Color(0xFF8B8FA3))),
                ],
              ]),
            ))
          ).toList()),
          _section('Claims', (spec['claims'] as List? ?? []).map((c) =>
            Padding(padding: const EdgeInsets.only(bottom: 8),
              child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Icon(Icons.check_circle_outline, size: 16, color: Color(0xFF22C55E)),
                const SizedBox(width: 8),
                Expanded(child: Text(c.toString(), style: const TextStyle(fontSize: 13))),
              ]))
          ).toList()),
          _section('Limitations', (spec['limitations'] as List? ?? []).map((l) =>
            Padding(padding: const EdgeInsets.only(bottom: 8),
              child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Icon(Icons.warning_amber, size: 16, color: Color(0xFFF59E0B)),
                const SizedBox(width: 8),
                Expanded(child: Text(l.toString(), style: const TextStyle(fontSize: 13))),
              ]))
          ).toList()),
          _section('Architecture', (spec['architecture_components'] as List? ?? []).map((a) =>
            Chip(label: Text(a.toString(), style: const TextStyle(fontSize: 12)),
              backgroundColor: const Color(0xFFA855F7).withOpacity(0.2))
          ).toList(), wrap: true),
        ],
      ),
    );
  }

  Widget _buildReproducibility(Map<String, dynamic> repro) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          Card(child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(children: [
              Text('${((repro['overall'] ?? 0) * 100).toStringAsFixed(0)}%',
                style: TextStyle(fontSize: 48, fontWeight: FontWeight.bold,
                  color: (repro['overall'] ?? 0) >= 0.5 ? const Color(0xFF22C55E) : const Color(0xFFEF4444))),
              const Text('Overall Score', style: TextStyle(color: Color(0xFF8B8FA3))),
            ]),
          )),
          const SizedBox(height: 16),
          ...['code_available', 'data_available', 'hyperparams_complete', 'compute_specified'].map((k) =>
            Padding(padding: const EdgeInsets.only(bottom: 8), child: _ProgressRow(
              label: k.replaceAll('_', ' '),
              value: (repro[k] ?? 0).toDouble(),
            ))
          ),
          const SizedBox(height: 16),
          if ((repro['blockers'] as List? ?? []).isNotEmpty)
            Card(child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Text('Blockers', style: TextStyle(fontWeight: FontWeight.w600, color: Color(0xFFEF4444))),
                const SizedBox(height: 8),
                ...(repro['blockers'] as List).map((b) => Padding(
                  padding: const EdgeInsets.only(bottom: 6),
                  child: Row(children: [
                    const Icon(Icons.error_outline, size: 16, color: Color(0xFFEF4444)),
                    const SizedBox(width: 8),
                    Expanded(child: Text(b.toString(), style: const TextStyle(fontSize: 13))),
                  ]),
                )),
              ]),
            )),
          if ((repro['fixes'] as List? ?? []).isNotEmpty)
            Card(child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Text('Suggested Fixes', style: TextStyle(fontWeight: FontWeight.w600, color: Color(0xFF22C55E))),
                const SizedBox(height: 8),
                ...(repro['fixes'] as List).map((f) => Padding(
                  padding: const EdgeInsets.only(bottom: 6),
                  child: Row(children: [
                    const Icon(Icons.lightbulb_outline, size: 16, color: Color(0xFF22C55E)),
                    const SizedBox(width: 8),
                    Expanded(child: Text(f.toString(), style: const TextStyle(fontSize: 13))),
                  ]),
                )),
              ]),
            )),
        ],
      ),
    );
  }

  Widget _buildAgents() {
    final messages = project!['agent_messages'] as List? ?? [];
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: messages.length,
      itemBuilder: (context, i) {
        final m = messages[i] as Map<String, dynamic>;
        final role = m['role'] ?? 'unknown';
        final colors = {
          'reader': const Color(0xFF3B82F6),
          'skeptic': const Color(0xFFF59E0B),
          'implementer': const Color(0xFF22C55E),
          'verifier': const Color(0xFFA855F7),
          'explainer': const Color(0xFF6366F1),
        };
        final color = colors[role] ?? Colors.grey;
        return Card(
          margin: const EdgeInsets.only(bottom: 8),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
            side: BorderSide(color: color.withOpacity(0.3)),
          ),
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Row(children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(color: color.withOpacity(0.2), borderRadius: BorderRadius.circular(4)),
                  child: Text(role.toString().toUpperCase(), style: TextStyle(fontSize: 10, fontWeight: FontWeight.bold, color: color)),
                ),
                const Spacer(),
                Text('${((m['confidence'] ?? 1) * 100).toStringAsFixed(0)}%', style: TextStyle(fontSize: 11, color: Colors.grey[500])),
              ]),
              const SizedBox(height: 6),
              Text(m['content'] ?? '', style: const TextStyle(fontSize: 13)),
            ]),
          ),
        );
      },
    );
  }

  Widget _buildActions() {
    final pid = project!['id'] ?? '';
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          _ActionTile(icon: Icons.map_outlined, title: 'View Visual Maps', subtitle: 'Architecture graphs, data flow, failure modes',
            onTap: () => Navigator.pushNamed(context, '/visual-map', arguments: project)),
          _ActionTile(icon: Icons.cloud_upload, title: 'Generate Productization', subtitle: 'API contract, deployment plan, launch package',
            onTap: () async {
              try {
                await ApiService.instance.productize(pid);
                if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Productization complete!')));
              } catch (e) {
                if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
              }
            }),
          _ActionTile(icon: Icons.science, title: 'Create Experiment', subtitle: 'Run a new experiment with current config',
            onTap: () async {
              try {
                final hp = (project!['paper_spec'] as Map?)?['hyperparameters'] as Map? ?? {};
                await ApiService.instance.createExperiment(pid, Map<String, dynamic>.from(hp));
                if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Experiment created!')));
              } catch (e) {
                if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
              }
            }),
          _ActionTile(icon: Icons.rate_review, title: 'Add Review', subtitle: 'Leave a review comment on this project',
            onTap: () async {
              try {
                await ApiService.instance.addReview(pid, 'mobile_user', 'Review from mobile app');
                if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Review added!')));
              } catch (e) {
                if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
              }
            }),
          _ActionTile(icon: Icons.check_circle, title: 'Approve Project', subtitle: 'Mark project as approved for production',
            onTap: () async {
              try {
                await ApiService.instance.approveProject(pid, 'mobile_admin');
                if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Project approved!')));
              } catch (e) {
                if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
              }
            }),
        ],
      ),
    );
  }

  Widget _section(String title, List<Widget> children, {bool wrap = false}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
        const SizedBox(height: 8),
        if (wrap) Wrap(spacing: 6, runSpacing: 6, children: children)
        else ...children,
        const SizedBox(height: 20),
      ],
    );
  }
}

class _StatCard extends StatelessWidget {
  final String value, label;
  final Color color;
  const _StatCard({required this.value, required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Expanded(child: Card(child: Padding(
      padding: const EdgeInsets.all(20),
      child: Column(children: [
        Text(value, style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: color)),
        const SizedBox(height: 4),
        Text(label, style: const TextStyle(fontSize: 12, color: Color(0xFF8B8FA3))),
      ]),
    )));
  }
}

class _ProgressRow extends StatelessWidget {
  final String label;
  final double value;
  const _ProgressRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Card(child: Padding(
      padding: const EdgeInsets.all(12),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
          Text(label, style: const TextStyle(fontSize: 13, textBaseline: TextBaseline.alphabetic)),
          Text('${(value * 100).toStringAsFixed(0)}%', style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
        ]),
        const SizedBox(height: 6),
        LinearProgressIndicator(value: value, backgroundColor: const Color(0xFF232635),
          color: value >= 0.5 ? const Color(0xFF22C55E) : const Color(0xFFF59E0B)),
      ]),
    ));
  }
}

class _ActionTile extends StatelessWidget {
  final IconData icon;
  final String title, subtitle;
  final VoidCallback onTap;
  const _ActionTile({required this.icon, required this.title, required this.subtitle, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: ListTile(
        leading: Icon(icon, color: const Color(0xFF6366F1)),
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.w500)),
        subtitle: Text(subtitle, style: const TextStyle(fontSize: 12, color: Color(0xFF8B8FA3))),
        trailing: const Icon(Icons.chevron_right),
        onTap: onTap,
      ),
    );
  }
}
