import 'package:flutter/material.dart';
import 'screens/splash_screen.dart';
import 'screens/login_screen.dart';
import 'screens/home_screen.dart';
import 'screens/form_screen.dart';
import 'screens/result_screens.dart';

void main() {
  runApp(const PCOSApp());
}

class PCOSApp extends StatelessWidget {
  const PCOSApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PCOS Detector',
      debugShowCheckedModeBanner: false,

      // 🎨 APP THEME (HEALTH APP STYLE)
      theme: ThemeData(
        useMaterial3: true,

        // 🌸 Primary Color (Health Pink)
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.pink,
          primary: Colors.pink,
        ),

        scaffoldBackgroundColor: const Color(0xFFF8F9FB),

        // 🧾 Text Theme
        textTheme: const TextTheme(
          titleLarge: TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.bold,
          ),
          bodyMedium: TextStyle(
            fontSize: 16,
          ),
        ),

        // 🔘 Button Style
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.pink,
            foregroundColor: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            padding: const EdgeInsets.symmetric(vertical: 14),
          ),
        ),

        // 📦 Card Style
        cardTheme: CardThemeData(
          elevation: 4,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
      ),

      // 🚀 START SCREEN
      initialRoute: '/',

      // 🔀 ROUTES (VERY IMPORTANT)
      routes: {
        '/': (context) => const SplashScreen(),
        '/login': (context) => const LoginScreen(),
        '/home': (context) => const HomeScreen(),
        '/form': (context) => const FormScreen(),
        '/result': (context) => const ResultScreen(data: {}),
      },
    );
  }
}