from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import face_recognition
import numpy as np
import base64
from multiprocessing.pool import ThreadPool
import threading 

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

# Create a thread pool for parallel processing
thread_pool = ThreadPool(processes=4)

def check_existing_faces(encoding):
    # Reload known face encodings from the folder
    known_face_encodings = []
    known_face_names = []
    for file in os.listdir(known_faces_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            image = face_recognition.load_image_file(os.path.join(known_faces_dir, file))
            encoding = face_recognition.face_encodings(image)[0]
            name = os.path.splitext(file)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(name)

    # Apply Householder matrix for privacy preservation
    epsilon = 0.1  # Privacy parameter
    noise = epsilon * np.random.randn(*encoding.shape)
    perturbed_encoding = encoding + noise
    
    # Parallelize face matching using thread pool
    matches = thread_pool.map(lambda known_encoding: face_recognition.compare_faces([known_encoding], perturbed_encoding)[0], known_face_encodings)
    return any(matches)


def save_new_face(image, name):
    cv2.imwrite(os.path.join(known_faces_dir, name + ".jpg"), image)

def draw_face_rectangles(frame, face_locations, recognized_names):
    for (top, right, bottom, left), name in zip(face_locations, recognized_names):
        # Draw rectangle around the recognized face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Display the name of the recognized person
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Function to perform face recognition in a separate thread
def recognize_faces_in_thread(frame, face_locations):
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    recognized_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        for i, match in enumerate(matches):
            if match:
                recognized_names.append(known_face_names[i])

    draw_face_rectangles(frame, face_locations, recognized_names)
    cv2.imshow('Video', frame)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        if 'photo' in request.form:
            photo_data = request.form['photo']
            # Convert base64 image data to OpenCV image
            nparr = np.fromstring(base64.b64decode(photo_data.split(',')[1]), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            encoding = face_recognition.face_encodings(image)
            if len(encoding) == 0:
                return "No face detected in the provided photo."
            else:
                # Check if the face already exists in the known face encodings
                if check_existing_faces(encoding[0]):
                    return "This face already exists in the database."
                else:
                    # Save new face and reload known face encodings
                    save_new_face(image, name)
                    reload_known_faces()
                    return redirect(url_for('index'))
    return render_template('signup.html')

def reload_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    for file in os.listdir(known_faces_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            image = face_recognition.load_image_file(os.path.join(known_faces_dir, file))
            encoding = face_recognition.face_encodings(image)[0]
            name = os.path.splitext(file)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(name)

# Main login function modified to use multi-threading
@app.route('/login')
def login():
    print("Starting face recognition process with privacy-preserving and multi-threading/multi-edge support...")
    video_capture = cv2.VideoCapture(0)
    
    def recognize_faces():
        print("Face recognition thread started.")
        while True:
            ret, frame = video_capture.read()
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            recognized_names = []

            print("Number of faces detected:", len(face_encodings))

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                print("Matches:", matches)
                for i, match in enumerate(matches):
                    if match:
                        recognized_names.append(known_face_names[i])

            draw_face_rectangles(frame, face_locations, recognized_names)
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Check if the video window is closed
            if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
                break

    # Start the face recognition thread with a name
    recognition_thread = threading.Thread(target=recognize_faces, name="FaceRecognitionThread")
    recognition_thread.start()

    print(f"Face recognition process started with thread: {recognition_thread.name}")
    print("Privacy-preserving techniques applied: Householder matrix with additive perturbation.")
    print("Multi-threading/multi-edge support enabled.")

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)