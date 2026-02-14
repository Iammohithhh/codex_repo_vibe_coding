// This is a template file showing Firebase configuration options for Paper2Product.
// Copy this to lib/config/firebase_options.dart and fill in your actual Firebase project values.

// For Firebase setup:
// 1. Go to Firebase Console: https://console.firebase.google.com/
// 2. Create a new project or select "Paper2Product"
// 3. Add an Android app with package ID: com.paper2product.app
// 4. Download the google-services.json from Firebase Console
// 5. Place it at android/app/google-services.json
// 6. Configure the options below with values from Firebase Console

import 'package:firebase_core/firebase_core.dart';

class FirebaseOptionsTemplate {
  static const FirebaseOptions android = FirebaseOptions(
    // Replace with your project ID
    projectId: 'paper2product-prod',

    // Replace with your App ID from Firebase Console
    appId: 'YOUR_APP_ID_HERE',

    // Replace with your API key
    apiKey: 'YOUR_API_KEY_HERE',

    // Replace with your sender ID (for FCM)
    messagingSenderId: 'YOUR_MESSAGING_SENDER_ID_HERE',

    // Replace with your database URL (if using Realtime Database)
    databaseURL: 'https://paper2product-prod.firebaseio.com',

    // Replace with your storage bucket
    storageBucket: 'paper2product-prod.appspot.com',
  );

  // Initialize Firebase with:
  // await Firebase.initializeApp(
  //   options: DefaultFirebaseOptions.currentPlatform,
  // );
}

// Environment-specific Firebase options structure:
// For Development: paper2product-dev
// For Staging: paper2product-staging
// For Production: paper2product-prod

// Additional Firebase services to enable:
// - Authentication (Email/Password, Google Sign-In)
// - Cloud Firestore (Database)
// - Realtime Database
// - Cloud Storage
// - Cloud Messaging (Push Notifications)
// - Crashlytics (Crash Reporting)
// - Analytics (User Analytics)
// - Remote Config (Feature Flags)
