import cv2
import numpy as np
import face_recognition
import psycopg2
import os


# Function to detect and crop the face using Haar Cascade
def detect_and_crop_face(image_path):
    # Load the Haar Cascade algorithm
    alg = "haarcascade_frontalface_default.xml"
    haar_cascade = cv2.CascadeClassifier(alg)

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load the image. Please check the file path.")
        return None

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = haar_cascade.detectMultiScale(
        gray_img, scaleFactor=1.04, minNeighbors=4, minSize=(70, 70)
    )

    # Check if at least one face is detected
    if len(faces) > 0:
        # Get the first face detected
        x, y, w, h = faces[0]

        # Crop the face
        cropped_face = img[y:y + h, x:x + w]
        return cropped_face
    else:
        print("No face detected in the image.")
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
            print("No face encodings found.")
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


if __name__ == "__main__":
    # Ask for Aadhaar number
    aadhaar_number = input("Enter Aadhaar Number: ")

    # Ask for the image file path
    image_path = input("Enter the path to the image file: ")

    # Check if the file exists
    if not os.path.exists(image_path):
        print("Error: The specified image file does not exist.")
        exit()

    # Detect and crop the face
    cropped_face = detect_and_crop_face(image_path)

    if cropped_face is not None:
        # Generate face embeddings
        embedding = get_face_embedding(cropped_face)

        if embedding is not None:
            # Store the embedding in the database
            store_embedding_in_db(aadhaar_number, embedding)

            # Display the cropped face for 2 seconds
            cv2.imshow("Cropped Face", cropped_face)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
