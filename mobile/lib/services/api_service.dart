import 'dart:convert';
import 'package:http/http.dart' as http;

/// API service for communicating with Paper2Product backend.
class ApiService {
  static final ApiService instance = ApiService._();
  ApiService._();

  String _baseUrl = 'http://10.0.2.2:8000'; // Android emulator localhost
  String _apiKey = '';

  void configure({required String baseUrl, String apiKey = ''}) {
    _baseUrl = baseUrl;
    _apiKey = apiKey;
  }

  Map<String, String> get _headers => {
    'Content-Type': 'application/json',
    if (_apiKey.isNotEmpty) 'Authorization': 'Bearer $_apiKey',
  };

  Future<Map<String, dynamic>> _get(String path) async {
    final resp = await http.get(Uri.parse('$_baseUrl$path'), headers: _headers);
    if (resp.statusCode != 200) throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
    return json.decode(resp.body);
  }

  Future<Map<String, dynamic>> _post(String path, Map<String, dynamic> body) async {
    final resp = await http.post(Uri.parse('$_baseUrl$path'), headers: _headers, body: json.encode(body));
    if (resp.statusCode != 200 && resp.statusCode != 201) {
      throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
    }
    return json.decode(resp.body);
  }

  // Projects
  Future<List<dynamic>> listProjects() async {
    final data = await _get('/api/v2/projects');
    return data['projects'] ?? [];
  }

  Future<Map<String, dynamic>> getProject(String id) => _get('/api/v2/projects/$id');

  Future<Map<String, dynamic>> ingestPaper({
    required String title,
    required String abstract_,
    required String methodText,
    String framework = 'pytorch',
    String persona = 'ml_engineer',
    String? sourceUrl,
  }) {
    return _post('/api/v2/projects/ingest', {
      'title': title,
      'abstract': abstract_,
      'method_text': methodText,
      'framework': framework,
      'persona': persona,
      if (sourceUrl != null) 'source_url': sourceUrl,
    });
  }

  // Visual Graph
  Future<Map<String, dynamic>> getVisualGraph(String projectId) =>
    _get('/api/v2/projects/$projectId/visual-graph');

  // Distillation
  Future<Map<String, dynamic>> getDistillation(String projectId) =>
    _get('/api/v2/projects/$projectId/distillation');

  // Reproducibility
  Future<Map<String, dynamic>> getReproducibility(String projectId) =>
    _get('/api/v2/projects/$projectId/reproducibility');

  // Agent Messages
  Future<Map<String, dynamic>> getAgentMessages(String projectId) =>
    _get('/api/v2/projects/$projectId/agent-messages');

  // Productize
  Future<Map<String, dynamic>> productize(String projectId, {String archType = 'server'}) =>
    _post('/api/v2/projects/$projectId/productize', {'architecture_type': archType});

  // Experiments
  Future<Map<String, dynamic>> getExperiments(String projectId) =>
    _get('/api/v2/projects/$projectId/experiments');

  Future<Map<String, dynamic>> createExperiment(String projectId, Map<String, dynamic> config) =>
    _post('/api/v2/projects/$projectId/experiments', {'config': config});

  // Reviews
  Future<Map<String, dynamic>> getReviews(String projectId) =>
    _get('/api/v2/projects/$projectId/reviews');

  Future<Map<String, dynamic>> addReview(String projectId, String author, String content) =>
    _post('/api/v2/projects/$projectId/reviews', {'author': author, 'content': content});

  Future<Map<String, dynamic>> approveProject(String projectId, String approver) =>
    _post('/api/v2/projects/$projectId/approve', {'approver': approver});
}
