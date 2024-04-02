# edge_node.py
from flask import Flask, request, redirect, url_for
import cv2
import face_recognition
import numpy as np
import os

app = Flask(__name__)

# Path to the directory containing known face images
known_faces_dir = "./faces"

# Load and encode existing images from the dataset
known_face_encodings = []
known_face_names = []

for file in os.listdir(known_faces_dir):
    if file.endswith(".jpg") or file.endswith(".png"):
        image = face_recognition.load_image_file(os.path.join(known_faces_dir, file))
        encoding = face_recognition.face_encodings(image)[0]
        name = os.path.splitext(file)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name)

# Route for signing up new faces
@app.route('/signup', methods=['POST'])
def signup():
    name = request.form['name']
    photo_data = request.form['photo']
    # Convert base64 image data to OpenCV image
    nparr = np.fromstring(base64.b64decode(photo_data.split(',')[1]), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    encoding = face_recognition.face_encodings(image)
    if len(encoding) == 0:
        return "No face detected in the provided photo."
    elif check_existing_faces(encoding[0]):
        return "This face already exists in the database."
    else:
        save_new_face(image, name)
        return redirect(url_for('index'))

# Route for logging in and performing face recognition
@app.route('/login', methods=['POST'])
def login():
    photo_data = request.form['photo']
    # Convert base64 image data to OpenCV image
    nparr = np.fromstring(base64.b64decode(photo_data.split(',')[1]), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    recognized_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        for i, match in enumerate(matches):
            if match:
                recognized_names.append(known_face_names[i])

    # Return the recognized names
    return ','.join(recognized_names)

if __name__ == '__main__':
    app.run(debug=True)
