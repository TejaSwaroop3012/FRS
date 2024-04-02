# central_node.py
from flask import Flask, render_template, request, redirect, url_for
import os
import face_recognition
import numpy as np
import base64
import requests

app = Flask(__name__)

# URL of the edge node for face recognition
EDGE_NODE_URL = "http://edge_node_ip_address:5000"

# Route for signing up new faces
@app.route('/signup', methods=['POST'])
def signup():
    name = request.form['name']
    photo_data = request.form['photo']
    # Convert base64 image data to OpenCV image
    nparr = np.fromstring(base64.b64decode(photo_data.split(',')[1]), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Send signup request to edge node
    files = {'photo': photo_data, 'name': name}
    response = requests.post(f"{EDGE_NODE_URL}/signup", files=files)
    
    return response.text

# Route for logging in and performing face recognition
@app.route('/login', methods=['POST'])
def login():
    photo_data = request.form['photo']
    # Convert base64 image data to OpenCV image
    nparr = np.fromstring(base64.b64decode(photo_data.split(',')[1]), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Send login request to edge node
    files = {'photo': photo_data}
    response = requests.post(f"{EDGE_NODE_URL}/login", files=files)
    
    return response.text

if __name__ == '__main__':
    app.run(debug=True)
