class UserData {
  int age;
  int cycleRegularity;
  int cycleLength;
  int weightGain;
  int hairGrowth;
  int skinDarkening;
  int hairLoss;
  int pimples;
  int fastFood;
  int exercise;
  double weight;
  double height;
  int sleep;

  UserData({
    required this.age,
    required this.cycleRegularity,
    required this.cycleLength,
    required this.weightGain,
    required this.hairGrowth,
    required this.skinDarkening,
    required this.hairLoss,
    required this.pimples,
    required this.fastFood,
    required this.exercise,
    required this.weight,
    required this.height,
    required this.sleep,
  });

  /// 🔥 Convert to JSON (for FastAPI request)
  Map<String, dynamic> toJson() {
    return {
      "Age_yrs": age,
      "Cycle_R_I": cycleRegularity,
      "Cycle_length_days": cycleLength,
      "Weight_gain_Y_N": weightGain,
      "hair_growth_Y_N": hairGrowth,
      "Skin_darkening_Y_N": skinDarkening,
      "Hair_loss_Y_N": hairLoss,
      "Pimples_Y_N": pimples,
      "Fast_food_Y_N": fastFood,
      "Reg_Exercise_Y_N": exercise,
      "Weight_kg": weight,
      "Height_cm": height,
      "Sleep_Rating_1_10": sleep,
    };
  }
}