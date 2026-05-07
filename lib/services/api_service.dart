import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = "https://pcos-ml-model.onrender.com"; // ✅ UPDATED

  static Future<Map<String, dynamic>> predictPCOS(Map<String, dynamic> data) async {
    try {
      await http.get(Uri.parse("https://pcos-ml-model.onrender.com"));
      final response = await http
          .post(
        Uri.parse("https://pcos-ml-model.onrender.com/predict"),
        headers: {
          "Content-Type": "application/json",
          "Connection": "keep-alive"
        },
        body: jsonEncode(data),
      )
          .timeout(const Duration(seconds: 40));

      print("Response: ${response.body}");
      // ⏱️ increased timeout (Render is slow first time)
      // print("Response Status: ${response.statusCode}");
      // print("Response Body: ${response.body}");

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception("Server Error: ${response.statusCode} - ${response.body}");
      }
    } catch (e) {
      throw Exception("API Error: $e");
    }
  }
}