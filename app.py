from flask import Flask, render_template, request, jsonify, redirect, url_for
import cv2
import numpy as np
import mediapipe as mp
import face_recognition
import psycopg2
from PIL import Image
import os

app = Flask(__name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

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

# Function to generate facial embeddings
def get_face_embedding(face_roi):
    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    try:
        face_encodings = face_recognition.face_encodings(face_rgb)
        return face_encodings[0] if len(face_encodings) > 0 else None
    except Exception as e:
        return None

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
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read the image file
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Process the image with MediaPipe Face Mesh
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks is not None:
        face_landmarks = results.multi_face_landmarks[0]
        face_roi = extract_face_roi(image, face_landmarks)

        # Generate face embeddings
        embedding = get_face_embedding(face_roi)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem'), debug=True)