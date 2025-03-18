import random
import subprocess
import sys

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import cv2
import numpy as np
import mediapipe as mp
import face_recognition
import psycopg2
import base64
from psycopg2 import errors as psycopg2_errors
from supabase import create_client, Client
from flask_cors import CORS
import subprocess
import uuid

from voice_gui import generate_ticket_number, store_ticket_in_supabase

app = Flask(__name__)
result_number = None

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Supabase configuration
SUPABASE_URL = "https://bliqlgvbgwfedjqpghcn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJsaXFsZ3ZiZ3dmZWRqcXBnaGNuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDIyMDYzMDAsImV4cCI6MjA1Nzc4MjMwMH0.bFRoPx5PIgEQg-L-UxFUs12H7bMd7TkTiAAYBSm03U8"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Function to extract face ROI using MediaPipe landmarks
def extract_face_roi(image, face_landmarks):
    h, w, _ = image.shape
    landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark], dtype=np.int32)
    x_min, x_max = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
    y_min, y_max = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
    padding = 20
    x_min, x_max = max(0, x_min - padding), min(w, x_max + padding)
    y_min, y_max = max(0, y_min - padding), min(h, y_max + padding)
    return image[y_min:y_max, x_min:x_max]

# Function to generate face embeddings
def get_face_embedding(face_image):
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    try:
        face_encodings = face_recognition.face_encodings(face_rgb)
        return face_encodings[0] if len(face_encodings) > 0 else None
    except Exception as e:
        print("Error generating embedding:", e)
        return None

# Function to store the embedding in the PostgreSQL database
def store_embedding_in_db(account_number, embedding):
    conn = None
    try:
        conn = psycopg2.connect(
            "postgresql://neondb_owner:npg_6jSUyIMPCcn9@ep-wandering-silence-a866q9l9-pooler.eastus2.azure.neon.tech/neondb?sslmode=require"
        )
        cur = conn.cursor()

        # Convert embedding into a PostgreSQL array of FLOAT[]
        query = """
        INSERT INTO user_embeddings (account_number, embedding)
        VALUES (%s, %s::FLOAT[])
        """
        cur.execute(query, (account_number, embedding.tolist()))
        conn.commit()
        print("Embedding successfully stored in the database.")
        return True, None  # Success, no error
    except psycopg2_errors.UniqueViolation:
        print("Duplicate account number detected.")
        return False, "Account number already exists. Please use a different account number."
    except Exception as e:
        print("Error storing embedding:", e)
        return False, str(e)
    finally:
        if conn:
            cur.close()
            conn.close()

# Function to check if the face exists in the database
def check_embedding_in_db(embedding):
    conn = None
    try:
        conn = psycopg2.connect(
            "postgresql://neondb_owner:npg_6jSUyIMPCcn9@ep-wandering-silence-a866q9l9-pooler.eastus2.azure.neon.tech/neondb?sslmode=require"
        )
        cur = conn.cursor()
        query = """
        SELECT account_number, embedding <-> %s::vector AS similarity
        FROM user_embeddings
        ORDER BY similarity ASC
        LIMIT 1
        """
        cur.execute(query, (embedding.tolist(),))
        result = cur.fetchone()
        if result:
            return result[0], result[1]  # Return account number and similarity
        return None, None
    except Exception as e:
        print("Error checking embedding:", e)
        return None, None
    finally:
        if conn:
            cur.close()
            conn.close()

@app.route('/')
def index():
    return render_template('index3.html')

# Registration Flow
@app.route('/register_capture', methods=['POST'])
def register_capture():
    print("Register Capture API called")  # Debugging
    data = request.json
    if 'image' not in data:
        return jsonify({"error": "No image data"}), 400

    # Decode the base64 image
    image_data = base64.b64decode(data['image'])
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    # Process the image with MediaPipe Face Mesh
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks is not None:
        face_landmarks = results.multi_face_landmarks[0]
        face_roi = extract_face_roi(image, face_landmarks)

        # Generate 3 face embeddings (reduced from 5 for faster processing)
        embeddings = []
        for _ in range(3):  # Capture 3 embeddings
            embedding = get_face_embedding(face_roi)
            if embedding is not None:
                embeddings.append(embedding)

        if embeddings:
            # Aggregate embeddings by taking the mean
            aggregated_embedding = np.mean(embeddings, axis=0)
            return jsonify({"status": "success", "aggregated_embedding": aggregated_embedding.tolist()})
        else:
            return jsonify({"error": "No face encodings found"}), 400
    else:
        return jsonify({"error": "No face detected in the image"}), 400

@app.route('/register', methods=['POST'])
def register():
    print("Register API called")  # Debugging
    data = request.json
    account_number = data.get('account_number')
    aggregated_embedding = np.array(data.get('aggregated_embedding'))

    if not account_number or aggregated_embedding is None:
        return jsonify({"error": "Invalid input"}), 400

    # Store the aggregated embedding in the database
    success, error_message = store_embedding_in_db(account_number, aggregated_embedding)
    if success:
        return jsonify({"status": "success", "message": "Registration successful!"})
    else:
        return jsonify({"status": "error", "message": error_message}), 400
@app.route('/capture', methods=['POST'])
def capture():
    data = request.json
    if 'image' not in data:
        return jsonify({"error": "No image data"}), 400

    # Decode the base64 image
    image_data = base64.b64decode(data['image'])
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    # Process the image with MediaPipe Face Mesh
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks is not None:
        face_landmarks = results.multi_face_landmarks[0]
        face_roi = extract_face_roi(image, face_landmarks)

        # Generate 3 face embeddings
        embeddings = []
        for _ in range(3):  # Capture 3 embeddings
            embedding = get_face_embedding(face_roi)
            if embedding is not None:
                embeddings.append(embedding)

        if embeddings:
            # Aggregate embeddings by taking the mean
            aggregated_embedding = np.mean(embeddings, axis=0)
            return jsonify({"status": "success", "aggregated_embedding": aggregated_embedding.tolist()})
        else:
            return jsonify({"error": "No face encodings found"}), 400
    else:
        return jsonify({"error": "No face detected in the image"}), 400

@app.route('/verify', methods=['POST'])
def verify():
    global result_number
    data = request.json
    embedding = np.array(data.get('embedding'))

    if embedding is None:
        return jsonify({"error": "Invalid input: No embedding provided"}), 400

    # Check the embedding in the database
    account_number, similarity = check_embedding_in_db(embedding)

    if account_number is not None:
        # Define a similarity threshold (adjust as needed)
        threshold = 0.6  # Lower values mean stricter matching
        result_number = account_number
        if similarity < threshold:
            return jsonify({
                "status": "success",
                "message": "Login successful!",
                "account_number": account_number,
                "similarity": similarity
            })

        else:
            return jsonify({
                "status": "error",
                "message": "Face not recognized. Please register.",
                "similarity": similarity
            }), 400
    else:
        return jsonify({"status": "error", "message": "No match found"}), 400

@app.route('/dashboard', methods=['GET'])
def dashboard():
    global result_number
    if not result_number:
        return jsonify({"status": "error", "message": "No account number found"}), 400

    try:
        # Fetch user information from Supabase
        response = supabase.table('users').select("*").eq('account_number', result_number).execute()
        user_result = response.data

        if user_result:
            # Render the dashboard template with user data
            print(user_result[0])
            return render_template('dashboard.html', user=user_result[0])
        else:
            return jsonify({"status": "error", "message": "User not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
@app.route('/create_account', methods=['POST'])
def create_account():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    phone = data.get('phone')
    pan_number = data.get('pan_number')
    aadhaar_number = data.get('aadhaar_number')

    if not name or not email or not phone or not pan_number or not aadhaar_number:
        return jsonify({"status": "error", "message": "All fields are required"}), 400

    try:
        # Insert the new user into the Supabase database
        response = supabase.table('users').insert({
            "full_name": name,
            "email": email,
            "mobile_number": phone,
            "pan_number": pan_number,
            "aadhaar_number": aadhaar_number
        }).execute()

        if response.data:
            # Fetch the newly created account number from the response
            new_user = response.data[0]  # Supabase returns the inserted row
            account_number = new_user.get('account_number')  # Ensure this column exists in your table

            return jsonify({
                "status": "success",
                "message": "Account created successfully!",
                "account_number": account_number  # Return the account number
            })
        else:
            return jsonify({"status": "error", "message": "Failed to create account"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/start-voice', methods=['GET'])
def start_voice():
    # Debugging: Print Python executable path
    print(f"Python executable path: {sys.executable}")

    # Use the same Python executable as the Flask server
    python_executable = sys.executable
    subprocess.Popen([python_executable, "voice_gui.py"])
    return jsonify({"status": "success", "message": "Voice GUI started!"})

@app.route('/account')
def account():
    # Fetch account_number and ticket_number from query parameters
    account_number = request.args.get('account_number')
    ticket_number = request.args.get('ticket_number')

    if not account_number:
        return "Account number is missing", 400

    # Fetch user details from Supabase
    response = supabase.table('users').select("*").eq('account_number', account_number).execute()
    user_data = response.data[0] if response.data else None

    if not user_data:
        return "User not found", 404

    # Render the account page with user data and ticket number
    return render_template('account.html', user=user_data, ticket_number=ticket_number)
@app.route('/loan')
def loan():
    # Fetch account_number and ticket_number from query parameters
    account_number = request.args.get('account_number')
    ticket_number = request.args.get('ticket_number')

    if not account_number:
        return "Account number is missing", 400

    # Fetch user details from Supabase
    response = supabase.table('users').select("*").eq('account_number', account_number).execute()
    user_data = response.data[0] if response.data else None

    if not user_data:
        return "User not found", 404

    # Render the loan page with user data and ticket number
    return render_template('loan.html', user=user_data, ticket_number=ticket_number)
@app.route('/fixed-deposit')
def fixed_deposit():
    # Fetch account_number and ticket_number from query parameters
    account_number = request.args.get('account_number')
    ticket_number = request.args.get('ticket_number')

    if not account_number:
        return "Account number is missing", 400

    # Fetch user details from Supabase
    response = supabase.table('users').select("*").eq('account_number', account_number).execute()
    user_data = response.data[0] if response.data else None

    if not user_data:
        return "User not found", 404

    # Render the fixed deposit page with user data and ticket number
    return render_template('fixed_deposit.html', user=user_data, ticket_number=ticket_number)
@app.route('/demat')
def demat():
    # Fetch account_number and ticket_number from query parameters
    account_number = request.args.get('account_number')
    ticket_number = request.args.get('ticket_number')

    if not account_number:
        return "Account number is missing", 400

    # Fetch user details from Supabase
    response = supabase.table('users').select("*").eq('account_number', account_number).execute()
    user_data = response.data[0] if response.data else None

    if not user_data:
        return "User not found", 404

    # Render the demat page with user data and ticket number
    return render_template('demat.html', user=user_data, ticket_number=ticket_number)

@app.route('/process-query', methods=['POST'])
def process_query():
    data = request.json
    query = data.get('query')
    account_number = data.get('account_number')

    if not query or not account_number:
        return jsonify({"status": "error", "message": "Invalid input"}), 400

    # Analyze the query to determine the intent
    intent = analyze_intent(query)

    if intent:
        # Generate a ticket number
        ticket_number = generate_ticket_number(intent[0], account_number)

        # Store the ticket in Supabase
        success, error_message = store_ticket_in_supabase(ticket_number, account_number, intent[0])
        if not success:
            return jsonify({"status": "error", "message": error_message}), 400

        # Redirect to the appropriate page based on the intent
        if intent[0] == "Account":
            return jsonify({
                "status": "success",
                "redirect_url": f"/account?account_number={account_number}&ticket_number={ticket_number}"
            })
        elif intent[0] == "Loan":
            return jsonify({
                "status": "success",
                "redirect_url": f"/loan?account_number={account_number}&ticket_number={ticket_number}"
            })
        elif intent[0] == "Fixed Deposit":
            return jsonify({
                "status": "success",
                "redirect_url": f"/fixed-deposit?account_number={account_number}&ticket_number={ticket_number}"
            })
        elif intent[0] == "Demat":
            return jsonify({
                "status": "success",
                "redirect_url": f"/demat?account_number={account_number}&ticket_number={ticket_number}"
            })
        else:
            return jsonify({"status": "error", "message": "Unknown intent"}), 400
    else:
        return jsonify({"status": "error", "message": "No valid intent detected"}), 400
def analyze_intent(text):
    """
    Determines banking intent using keyword matching.
    Possible intents: Account, Loan, Fixed Deposit, Demat.
    """
    if not text:
        return []

    text = text.lower()

    # Keywords for banking-related intents
    intents = {
        "account": ["account", "balance", "statement", "passbook"],
        "loan": ["loan", "credit", "borrow", "emi"],
        "fixed deposit": ["fixed deposit", "fd", "investment"],
        "demat": ["demat", "shares", "stocks", "trading"]
    }

    detected_intents = set()

    # Check if specific banking keywords are present
    for intent, keywords in intents.items():
        for keyword in keywords:
            if keyword in text:
                detected_intents.add(intent.capitalize())

    if detected_intents:
        return list(detected_intents)  # Return all matching intents as a list
    return []
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem'), debug=True)
    CORS(app, origins='*')