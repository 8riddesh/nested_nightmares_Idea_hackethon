import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import face_recognition
import psycopg2
from PIL import Image

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
        st.error(f"Error generating embedding: {e}")
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
        st.error(f"Database error: {e}")
        return False
    finally:
        if conn:
            cur.close()
            conn.close()

# Streamlit App
def main():
    st.set_page_config(page_title="Face Recognition System", page_icon="👤", layout="centered")

    # Custom CSS for better UI
    st.markdown("""
        <style>
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 24px;
            border-radius: 5px;
            border: none;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stTextInput input {
            font-size: 16px;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Face Recognition System")
    st.write("Capture your face and verify your Aadhaar number.")

    # Webcam capture
    st.write("### Step 1: Capture Your Face")
    webcam_image = st.camera_input("Look at the camera and click 'Capture'.")

    if webcam_image is not None:
        # Convert the image to a numpy array
        # image = np.array(Image.open(webcam_image))
        path=input("ENter image path:")
        image=cv2.imread(path)
        # Process the image with MediaPipe Face Mesh
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks is not None:

            face_landmarks = results.multi_face_landmarks[0]
            face_roi = extract_face_roi(image, face_landmarks)

            # Display the cropped face
            st.image(face_roi, caption="Captured Face", use_column_width=True)

            # Generate face embeddings
            embedding = get_face_embedding(face_roi)

            if embedding is not None:
                st.success("Face captured successfully!")

                # Aadhaar number input
                st.write("### Step 2: Enter Your Aadhaar Number")
                aadhaar_number = st.text_input("Aadhaar Number", placeholder="Enter your Aadhaar number")

                if st.button("Verify"):
                    if aadhaar_number:
                        # Check if the embedding and Aadhaar number exist in the database
                        if check_embedding_in_db(aadhaar_number, embedding):
                            st.success("Login successful!")
                        else:
                            st.warning("Please register.")
                    else:
                        st.error("Please enter a valid Aadhaar number.")
            else:
                st.error("No face encodings found.")
        else:
            st.error("No face detected in the image.")

if __name__ == "__main__":
    main()