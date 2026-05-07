import 'package:flutter/material.dart';
import 'result_screens.dart';
import '../services/api_service.dart';

class FormScreen extends StatefulWidget {
  const FormScreen({super.key});

  @override
  State<FormScreen> createState() => _FormScreenState();
}

class _FormScreenState extends State<FormScreen> {
  final _formKey = GlobalKey<FormState>();

  final ageController = TextEditingController();
  final cycleLengthController = TextEditingController();
  final weightController = TextEditingController();
  final heightController = TextEditingController();
  final sleepController = TextEditingController();

  bool weightGain = false;
  bool hairGrowth = false;
  bool skinDarkening = false;
  bool hairLoss = false;
  bool pimples = false;
  bool fastFood = false;
  bool exercise = false;

  int cycleRegularity = 0;

  // 🔥 API CALL
  void submitData() async {
    if (_formKey.currentState!.validate()) {
      Map<String, dynamic> requestData = {
        "Age_yrs": int.parse(ageController.text),
        "Cycle_R_I": cycleRegularity,
        "Cycle_length_days": int.parse(cycleLengthController.text),
        "Weight_gain_Y_N": weightGain ? 1 : 0,
        "hair_growth_Y_N": hairGrowth ? 1 : 0,
        "Skin_darkening_Y_N": skinDarkening ? 1 : 0,
        "Hair_loss_Y_N": hairLoss ? 1 : 0,
        "Pimples_Y_N": pimples ? 1 : 0,
        "Fast_food_Y_N": fastFood ? 1 : 0,
        "Reg_Exercise_Y_N": exercise ? 1 : 0,
        "Weight_kg": double.parse(weightController.text),
        "Height_cm": double.parse(heightController.text),
        "Sleep_Rating_1_10": int.parse(sleepController.text),
      };

      try {
        final result = await ApiService.predictPCOS(requestData);

        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => ResultScreen(data: result),
          ),
        );
      } catch (e) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("API Error: $e")),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: true,

      appBar: AppBar(
        title: const Text("PCOS Assessment"),
        backgroundColor: Colors.pink,
      ),

      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),

          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [

                const Text(
                  "Fill Your Health Details",
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),

                const SizedBox(height: 15),

                /// AGE
                TextFormField(
                  controller: ageController,
                  decoration: const InputDecoration(labelText: "Age (yrs)"),
                  keyboardType: TextInputType.number,
                  validator: (value) => value!.isEmpty ? "Enter age" : null,
                ),

                /// CYCLE
                DropdownButtonFormField<int>(
                  value: cycleRegularity,
                  items: const [
                    DropdownMenuItem(value: 0, child: Text("Regular")),
                    DropdownMenuItem(value: 1, child: Text("Irregular")),
                  ],
                  onChanged: (val) => setState(() => cycleRegularity = val!),
                  decoration: const InputDecoration(labelText: "Cycle (R/I)"),
                ),

                /// CYCLE LENGTH
                TextFormField(
                  controller: cycleLengthController,
                  decoration: const InputDecoration(labelText: "Cycle Length (days)"),
                  keyboardType: TextInputType.number,
                  validator: (value) => value!.isEmpty ? "Enter cycle length" : null,
                ),

                /// WEIGHT
                TextFormField(
                  controller: weightController,
                  decoration: const InputDecoration(labelText: "Weight (kg)"),
                  keyboardType: TextInputType.number,
                  validator: (value) => value!.isEmpty ? "Enter weight" : null,
                ),

                /// HEIGHT
                TextFormField(
                  controller: heightController,
                  decoration: const InputDecoration(labelText: "Height (cm)"),
                  keyboardType: TextInputType.number,
                  validator: (value) => value!.isEmpty ? "Enter height" : null,
                ),

                /// SLEEP
                TextFormField(
                  controller: sleepController,
                  decoration: const InputDecoration(labelText: "Sleep Rating (1–10)"),
                  keyboardType: TextInputType.number,
                  validator: (value) => value!.isEmpty ? "Enter sleep rating" : null,
                ),

                const SizedBox(height: 10),

                /// CHECKBOX SECTION TITLE
                const Text(
                  "Symptoms",
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),

                /// CHECKBOXES
                CheckboxListTile(
                  title: const Text("Weight Gain"),
                  value: weightGain,
                  onChanged: (val) => setState(() => weightGain = val!),
                ),

                CheckboxListTile(
                  title: const Text("Hair Growth"),
                  value: hairGrowth,
                  onChanged: (val) => setState(() => hairGrowth = val!),
                ),

                CheckboxListTile(
                  title: const Text("Skin Darkening"),
                  value: skinDarkening,
                  onChanged: (val) => setState(() => skinDarkening = val!),
                ),

                CheckboxListTile(
                  title: const Text("Hair Loss"),
                  value: hairLoss,
                  onChanged: (val) => setState(() => hairLoss = val!),
                ),

                CheckboxListTile(
                  title: const Text("Pimples"),
                  value: pimples,
                  onChanged: (val) => setState(() => pimples = val!),
                ),

                CheckboxListTile(
                  title: const Text("Fast Food"),
                  value: fastFood,
                  onChanged: (val) => setState(() => fastFood = val!),
                ),

                CheckboxListTile(
                  title: const Text("Regular Exercise"),
                  value: exercise,
                  onChanged: (val) => setState(() => exercise = val!),
                ),

                const SizedBox(height: 30),

                /// 🔥 BIG BUTTON
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: submitData,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.pink,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    child: const Text(
                      "Check PCOS Risk",
                      style: TextStyle(fontSize: 16),
                    ),
                  ),
                ),

                const SizedBox(height: 50), // 🔥 IMPORTANT (fix hidden button)
              ],
            ),
          ),
        ),
      ),
    );
  }
}