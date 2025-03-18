from flask import Flask, render_template, request, jsonify, redirect, url_for
import cv2
import numpy as np
import face_recognition
import psycopg2
import os

app = Flask(__name__)

# Function to detect and crop the face using Haar Cascade
def detect_and_crop_face(image):
    # Load the Haar Cascade algorithm
    alg = "haarcascade_frontalface_default.xml"
    haar_cascade = cv2.CascadeClassifier(alg)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = haar_cascade.detectMultiScale(
        gray_img, scaleFactor=1.04, minNeighbors=4, minSize=(70, 70)
    )

    # Check if at least one face is detected
    if len(faces) > 0:
        # Get the first face detected
        x, y, w, h = faces[0]

        # Crop the face
        cropped_face = image[y:y + h, x:x + w]
        return cropped_face
    else:
        return None

# Function to generate face embeddings using face_recognition
def get_face_embedding(face_image):
    # Convert the face image to RGB (face_recognition expects RGB images)
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Generate face embeddings
    try:
        face_encodings = face_recognition.face_encodings(face_rgb)
        if len(face_encodings) > 0:
            return face_encodings[0]  # Return the first face embedding
        else:
            return None
    except Exception as e:
        print("Error generating embedding:", e)
        return None

# Function to store the embedding in the PostgreSQL database
def store_embedding_in_db(aadhaar_number, embedding):
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            "postgresql://neondb_owner:npg_6jSUyIMPCcn9@ep-wandering-silence-a866q9l9-pooler.eastus2.azure.neon.tech/neondb?sslmode=require"
        )
        cur = conn.cursor()

        # Insert the embedding into the aadhaar_embeddings table
        query = "INSERT INTO aadhaar_embeddings (aadhaar_number, embedding) VALUES (%s, %s::FLOAT[])"
        cur.execute(query, (aadhaar_number, embedding.tolist()))
        conn.commit()

        print("Embedding successfully stored in the database.")
    except Exception as e:
        print("Error connecting to the database or executing the query:", e)
    finally:
        if conn:
            cur.close()
            conn.close()

# Function to check if the embedding and Aadhaar number exist in the database
def check_embedding_in_db(aadhaar_number, embedding):
    try:
        conn = psycopg2.connect(
            "postgresql://neondb_owner:npg_6jSUyIMPCcn9@ep-wandering-silence-a866q9l9-pooler.eastus2.azure.neon.tech/neondb?sslmode=require"
        )
        cur = conn.cursor()
        query = """
        SELECT EXISTS (
            SELECT 1
            FROM aadhaar_embeddings
            WHERE aadhaar_number = %s
            AND embedding <-> %s::vector < 0.6
        )
        """
        cur.execute(query, (aadhaar_number, embedding.tolist()))
        exists = cur.fetchone()[0]
        return exists
    except Exception as e:
        return False
    finally:
        if conn:
            cur.close()
            conn.close()

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/capture', methods=['POST'])
def capture():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read the image file
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Detect and crop the face
    cropped_face = detect_and_crop_face(image)

    if cropped_face is not None:
        # Generate face embeddings
        embedding = get_face_embedding(cropped_face)

        if embedding is not None:
            return jsonify({"status": "success", "embedding": embedding.tolist()})
        else:
            return jsonify({"error": "No face encodings found"}), 400
    else:
        return jsonify({"error": "No face detected in the image"}), 400

@app.route('/verify', methods=['POST'])
def verify():
    data = request.json
    aadhaar_number = data.get('aadhaar_number')
    embedding = np.array(data.get('embedding'))

    if not aadhaar_number or embedding is None:
        return jsonify({"error": "Invalid input"}), 400

    if check_embedding_in_db(aadhaar_number, embedding):
        return jsonify({"status": "success", "message": "Login successful!"})
    else:
        return jsonify({"status": "warning", "message": "Please register."}), 400

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    aadhaar_number = data.get('aadhaar_number')
    embedding = np.array(data.get('embedding'))

    if not aadhaar_number or embedding is None:
        return jsonify({"error": "Invalid input"}), 400

    # Store the embedding in the database
    store_embedding_in_db(aadhaar_number, embedding)
    return jsonify({"status": "success", "message": "Registration successful!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem'), debug=True)