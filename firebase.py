import firebase_admin
from firebase_admin import credentials, auth

# Path to your Firebase service account key
cred = credentials.Certificate("webar-96b1c-firebase-adminsdk-gx9dv-0938b86dfe.json")
firebase_admin.initialize_app(cred)
