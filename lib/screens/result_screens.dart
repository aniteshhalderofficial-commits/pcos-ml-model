import 'package:flutter/material.dart';

class ResultScreen extends StatelessWidget {
  final Map<String, dynamic> data;

  const ResultScreen({super.key, required this.data});

  @override
  Widget build(BuildContext context) {

    double probability = ((data["risk_probability"] ?? 0.0) * 100);
    String riskStage = data["risk_stage"] ?? "Unknown";
    String confidence = data["prediction_confidence"] ?? "";
    List recommendations = data["lifestyle_recommendations"] ?? [];
    String sleepAdvice = data["sleep_advisory"] ?? "";

    Color riskColor;
    if (riskStage.toLowerCase().contains("very high") ||
        riskStage.toLowerCase().contains("high")) {
      riskColor = Colors.red;
    } else if (riskStage.toLowerCase().contains("moderate")) {
      riskColor = Colors.orange;
    } else {
      riskColor = Colors.green;
    }

    return Scaffold(
      backgroundColor: Colors.grey[100],
      appBar: AppBar(
        title: const Text("PCOS Result"),
        backgroundColor: Colors.pink,
        elevation: 0,
      ),

      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [

            /// 🔥 RISK CARD
            Card(
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
              elevation: 5,
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  children: [

                    // Circular Indicator
                    Stack(
                      alignment: Alignment.center,
                      children: [
                        SizedBox(
                          height: 120,
                          width: 120,
                          child: CircularProgressIndicator(
                            value: probability / 100,
                            strokeWidth: 10,
                            color: riskColor,
                            backgroundColor: Colors.grey[300],
                          ),
                        ),
                        Text(
                          "${probability.toStringAsFixed(0)}%",
                          style: const TextStyle(
                            fontSize: 22,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 15),

                    Text(
                      riskStage,
                      style: TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.bold,
                        color: riskColor,
                      ),
                    ),

                    const SizedBox(height: 5),

                    Text(
                      confidence,
                      style: const TextStyle(color: Colors.grey),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 20),

            /// 💡 RECOMMENDATIONS CARD
            Card(
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
              elevation: 3,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [

                    const Text(
                      "Recommendations",
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),

                    const SizedBox(height: 10),

                    ...recommendations.map((rec) => Padding(
                      padding: const EdgeInsets.symmetric(vertical: 4),
                      child: Row(
                        children: [
                          const Icon(Icons.check_circle, color: Colors.green, size: 18),
                          const SizedBox(width: 8),
                          Expanded(child: Text(rec)),
                        ],
                      ),
                    )),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 20),

            /// 😴 SLEEP CARD
            if (sleepAdvice.isNotEmpty)
              Card(
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
                color: Colors.pink.shade50,
                elevation: 2,
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    children: [
                      const Icon(Icons.bedtime, color: Colors.pink),
                      const SizedBox(width: 10),
                      Expanded(child: Text(sleepAdvice)),
                    ],
                  ),
                ),
              ),

            const SizedBox(height: 30),

            /// 🔁 BUTTON
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.pink,
                  padding: const EdgeInsets.symmetric(vertical: 15),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                onPressed: () {
                  Navigator.pop(context);
                },
                child: const Text("Check Again"),
              ),
            ),
          ],
        ),
      ),
    );
  }
}