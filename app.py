import re
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import firebase_admin
from firebase_admin import credentials, auth, firestore
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import requests
import json
from flask_compress import Compress
app = Flask(__name__)
Compress(app)

PROFILE_DIR = os.path.join(os.getcwd(), 'profiles')
if not os.path.exists(PROFILE_DIR):
    os.makedirs(PROFILE_DIR)

def save_profile_to_json(user_id, profile_data):
    """Save profile data to a JSON file."""
    filepath = os.path.join(PROFILE_DIR, f"{user_id}.json")
    with open(filepath, 'w') as json_file:
        json.dump(profile_data, json_file, indent=4)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("webar-96b1c-firebase-adminsdk-gx9dv-0938b86dfe.json")  # Set correct path to your Firebase admin SDK JSON
firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Set your own secure secret key
QUIZZES_DIR = os.path.join(os.getcwd(), 'quizzes')
if not os.path.exists(QUIZZES_DIR):
    os.makedirs(QUIZZES_DIR)

# Allow image file types for upload
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# ORB + FLANN-based feature matching function
def match_image(uploaded_image_path):
    image_database_path = 'static/images'  # Make sure images here are pre-processed
    image_files = os.listdir(image_database_path)  

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Read and resize the uploaded image
    uploaded_image = cv2.imread(uploaded_image_path, cv2.IMREAD_GRAYSCALE)
    uploaded_image = cv2.resize(uploaded_image, (800, 600))  # Resize to a smaller size (you can adjust the dimensions)

    # Key points and descriptors of the uploaded image
    kp1, des1 = orb.detectAndCompute(uploaded_image, None)

    for file in image_files:
        if file.endswith((".png", ".jpg", ".jpeg")):
            stored_image_path = os.path.join(image_database_path, file)
            stored_image = cv2.imread(stored_image_path, cv2.IMREAD_GRAYSCALE)

            # Resize the stored image as well to reduce processing time
            stored_image = cv2.resize(stored_image, (800, 600))  # Resize to the same size as uploaded image

            # Key points and descriptors of the stored image
            kp2, des2 = orb.detectAndCompute(stored_image, None)

            # FLANN-based matcher
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)  # Higher values improve accuracy but slow down processing
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            # Perform FLANN-based matching
            matches = flann.knnMatch(des1, des2, k=2)

            # Apply the ratio test to filter good matches
            good_matches = []
            for match in matches:
                if len(match) == 2:  # Check if there are two matches to unpack
                    m, n = match
                    if m.distance < 0.7 * n.distance:  # Ratio test
                        good_matches.append(m)
                else:
                    print("Skipping match: not enough matches")

            # If good matches are found, return the image name
            if len(good_matches) > 10:  # Minimum number of good matches
                return file.split('.')[0]  # Return image name without extension

    return None



@app.route('/welcome')
def welcome():
    if 'user_id' in session:
        return redirect(url_for('learning'))  # Redirect logged-in users to learning page
    return render_template('welcome.html')

@app.route('/')
def home_redirect():
    if 'user_id' in session:
        return redirect(url_for('learning'))  # Redirect logged-in users to learning page
    return redirect(url_for('welcome'))  # Redirect to the welcome page if not logged in

@app.route('/about')
def about():
    return "<h1>About Page</h1><p>This is a brief description of the app.</p>"

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            # Use Firebase REST API to verify the email and password
            response = requests.post(
                "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword",
                params={"key": "AIzaSyCcbS5lh7pfJ4g9CQVqUiyImAAtGYPlveY"},
                json={"email": email, "password": password, "returnSecureToken": True},
            )
            response_data = response.json()

            if response.status_code == 200:
                session['user_id'] = response_data["localId"]  # Store user ID in session
                flash('Login successful!', 'success')
                return redirect(url_for('learning'))  # Redirect to a protected page (e.g., learning)
            else:
                error_message = response_data.get("error", {}).get("message", "Login failed.")
                
                # Custom error handling
                if "EMAIL_NOT_FOUND" in error_message:
                    flash("Invalid email address.", 'error')
                elif "INVALID_PASSWORD" in error_message:
                    flash("Incorrect password.", 'error')
                else:
                    flash(error_message, 'error')  # Generic error message

        except Exception as e:
            flash(f'Error: {str(e)}', 'error')

    return render_template('login.html')

# Route for the registration page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Validate email format
        if not email or not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash('Invalid email format. Please enter a valid email address.', 'error')
            return render_template('register.html')

        try:
            # Create a new user using Firebase Auth (no email verification sending here)
            user = auth.create_user(email=email, password=password)

            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))

        except firebase_admin.exceptions.FirebaseError as e:
            flash(f'Error: {str(e)}', 'error')  # Handle Firebase errors like email already in use
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')

    return render_template('register.html')

def load_profile_from_json(user_id):
    """Load profile data from a JSON file."""
    filepath = os.path.join(PROFILE_DIR, f"{user_id}.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as json_file:
            return json.load(json_file)
    return None
@app.route('/home')
def home():
    return render_template('welcome.html')  # or whatever page you'd like to show for the home route

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return jsonify({'error': 'User not logged in'}), 401

    user_id = session['user_id']

    if request.method == 'POST':
        name = request.form.get('name')
        bio = request.form.get('bio')

        # Save to JSON
        profile_data = {'name': name, 'bio': bio}
        save_profile_to_json(user_id, profile_data)

        return jsonify({'message': 'Profile updated successfully'}), 200

    # For GET requests, load profile data
    profile = load_profile_from_json(user_id)
    if profile:
        return jsonify(profile), 200
    else:
        return jsonify({'error': 'Profile not found'}), 404

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')  # or return your chatbot logic
import os
import json

TOPIC_COUNTS_FILE = "topic_counts.json"

# Ensure the file exists and is initialized
if not os.path.exists(TOPIC_COUNTS_FILE) or os.path.getsize(TOPIC_COUNTS_FILE) == 0:
    with open(TOPIC_COUNTS_FILE, 'w') as file:
        json.dump({}, file)






# Path to the topic counts file
TOPIC_COUNTS_FILE = 'topic_counts.json'

# Function to load the current topic counts from the JSON file
def load_topic_counts():
    # Check if the file exists
    if os.path.exists(TOPIC_COUNTS_FILE):
        with open(TOPIC_COUNTS_FILE, 'r') as file:
            try:
                # Try to load the data, return empty dictionary if there's an error
                return json.load(file)
            except json.JSONDecodeError:
                return {}
    else:
        return {}

# Function to save the updated topic counts back to the JSON file
def save_topic_counts(topic_counts):
    with open(TOPIC_COUNTS_FILE, 'w') as file:
        json.dump(topic_counts, file, indent=4)

@app.route('/update-topic-count', methods=['POST'])
def update_topic_count():
    data = request.get_json()  # Get JSON data from the request
    topic = data.get('topic')  # Get the topic from the data

    if not topic:
        return jsonify({"error": "Topic not provided"}), 400

    # Load existing topic counts
    topic_counts = load_topic_counts()

    # Update the count for the given topic
    if topic in topic_counts:
        topic_counts[topic] += 1  # Increment the existing count
    else:
        topic_counts[topic] = 1  # Initialize the count if the topic is new

    # Save the updated topic counts to the file
    save_topic_counts(topic_counts)

    return jsonify({"message": f"Topic count for '{topic}' updated successfully"})
# Route for resending email verification
@app.route('/resend_verification')
def resend_verification():
    if 'user_id' not in session:
        flash('You need to log in first', 'error')
        return redirect(url_for('login'))

    try:
        user = auth.get_user(session['user_id'])
        auth.send_email_verification(user.uid)  # Resend the email verification
        flash('Verification email has been sent again. Please check your inbox.', 'success')
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')

    return redirect(url_for('learning'))  # Redirect to the learning page or login

# Route for the learning page
@app.route('/learning')
def learning():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Get the user from Firebase using the stored user_id from session
    user = auth.get_user(session['user_id'])

    # Pass user data to the template
    return render_template('learning.html', user=user)

# Route for the index page
@app.route('/index')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if user is not logged in
    return render_template('index.html')

# Route for logout
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

# Image upload route
# Place this above your routes or inside a utility function
image_descriptions = {
    "lion": "The lion is a large cat of the genus Panthera, native to Africa and India. It has a muscular, broad-chested body; a short, rounded head; round ears; and a dark, hairy tuft at the tip of its tail. It is sexually dimorphic; adult male lions are larger than females and have a prominent mane.",
    "solar": "The solar system is located in the Milky Way galaxy, a barred spiral galaxy with two major arms and two minor arms. The Sun is in the Orion Arm, between the Sagittarius and Perseus arms. The solar system orbits the center of the galaxy at about 515,000 mph (828,000 kph) and takes about 230 million years to complete one orbit",
    "human":"Brain: Inside the head, it helps us think, remember, and control our body.Heart: Pumps blood to the entire body.Lungs: Help us breathe in oxygen and breathe out carbon dioxide.Eyes: Help us see the world around us.Ears: Help us hear sounds.Nose: Helps us smell and breathe.Mouth: Helps us talk, eat, and taste food.",
    # Add more images and their descriptions here
}
RESULTS_DIR = os.path.join(os.getcwd(), 'results')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"message": "No file part"}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)

        # Call the match_image function with the uploaded image
        matched_image = match_image(filepath)

        if matched_image:
            # Extract image name without extension for static description lookup
            matched_image_name = matched_image.split('.')[0]  # Remove extension
            description = image_descriptions.get(matched_image_name, "Description not available.")

            video_url = f"/static/videos/{matched_image_name}_video.mp4"
            audio_url = f"/static/audio/{matched_image_name}_audio.mp3"

            return jsonify({
                "message": f"Image Matched with {matched_image_name}",
                "audio": audio_url,
                "video": video_url,
                "description": description
            })
        else:
            return jsonify({"message": "No Match Found"}), 404
    else:
        return jsonify({"message": "Invalid file type. Only jpg, jpeg, png, gif allowed."}), 400
@app.route('/playback')
def playback():
    video_url = request.args.get('video')
    audio_url = request.args.get('audio')
    description = request.args.get('description')
    index = 1  # Or whatever value you need for 'index'

    return render_template('playback.html', video=video_url, audio=audio_url, description=description, index=index)


@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Sample quiz data (you can load this from a file or database if needed)
    quiz_data = {
        'question1': {
            'question': 'What is the capital of France?',
            'choices': ['Berlin', 'Madrid', 'Paris', 'Rome'],
            'answer': 'Paris'
        },
        'question2': {
            'question': 'What is the largest planet in our solar system?',
            'choices': ['Earth', 'Mars', 'Jupiter', 'Saturn'],
            'answer': 'Jupiter'
        },
        # Add more questions as needed
    }

    if request.method == 'POST':
        score = 0
        for question_id, question_data in quiz_data.items():
            selected_answer = request.form.get(question_id)
            if selected_answer == question_data['answer']:
                score += 1
        return render_template('quiz_result.html', score=score, total=len(quiz_data))

    return render_template('quiz.html', quiz_data=quiz_data)

from datetime import datetime

@app.route('/submit-results', methods=['POST'])
def submit_results():
    data = request.json  # Get JSON data sent from the client

    # Add the current date and time to the data
    data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Save the data to a file
    try:
        with open('results.json', 'a') as file:
            file.write(json.dumps(data) + '\n')  # Append JSON data with newline
        return jsonify({"message": "Results saved successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)