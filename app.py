from flask import Flask, request, jsonify,render_template,send_from_directory,json,Response
from flask_socketio import SocketIO
import base64
import cv2
import os
import face_recognition
from threading import Thread
import numpy as np
import psutil
import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
socketio = SocketIO(app)

# Dictionary to store criminal details
criminals_database = {}
known_faces_folder="known_faces"
output_folder="output_images"
face_names=[]
criminals_data_folder="criminals_data"
small_frame_bytes = b''


@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/Register',methods=['GET'])
def Register():
    return render_template('register.html')

@app.route('/Detect',methods=['GET'])
def Detect():
    return render_template('detect.html')

@app.route('/Video',methods=['GET'])
def Video():
    return render_template('video.html')

@app.route('/output_images/<filename>')
def output_images(filename):
    output_images_folder = os.path.join(os.getcwd(), 'output_images')  # Replace with the actual path
    return send_from_directory(output_images_folder, filename)

def save_criminals_data():
    for criminal_id, criminal_details in criminals_database.items():
        criminal_dir = os.path.join(criminals_data_folder, criminal_id)
        os.makedirs(criminal_dir, exist_ok=True)

        json_file_path = os.path.join(criminal_dir, 'details.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(criminal_details, json_file, indent=4)

def load_criminals_data():
    criminals_data = {}
    for criminal_id in os.listdir(criminals_data_folder):
        criminal_dir = os.path.join(criminals_data_folder, criminal_id)
        json_file_path = os.path.join(criminal_dir, 'details.json')

        if os.path.isfile(json_file_path):
            with open(json_file_path, 'r') as json_file:
                criminal_details = json.load(json_file)
                criminals_data[criminal_id] = criminal_details

    return criminals_data




@app.route('/register_criminal', methods=['POST'])
def register_criminal():
    image_file = request.files.get('image')
    
    if image_file:
        response={'errFlag': 0}
    
    #else:
    details = {
        'name': request.form.get('name'),
        'gender': request.form.get('gender'),
        'dob': request.form.get('dob'),
        'blood_group': request.form.get('blood_group'),
        'identification_mark': request.form.get('identification_mark'),
        'nationality': request.form.get('nationality'),
        'religion': request.form.get('religion'),
        'crimes_done': request.form.get('crimes_done')
    }
    # Create a unique identifier for the criminal (e.g., using the name)
    criminal_id = details['name'].lower().replace(" ", "_")
    # Create a subdirectory for the criminal using their identifier
    #criminal_dir = os.path.join('criminals_data', criminal_id)
    #os.makedirs(criminal_dir, exist_ok=True)
    # Save the images and store their paths in the criminal details
    image_paths = []
    image_file_name=f"{criminal_id}.jpg"
    image_path = os.path.join(known_faces_folder, image_file_name)
    image_file.save(image_path)
    im=cv2.imread(image_path)
    cv2.imwrite(image_path,im)
    image_paths.append(image_path)
    details['image_paths'] = image_paths
    # Store the details in the criminals database
    criminals_database[criminal_id] = details

    save_criminals_data()
    # Prepare the response
    response = {
        "errFlag":1,
        'status': 'Criminal registered successfully',
        'criminal_id': criminal_id
    }
    #print(criminals_database)

    return jsonify(response)

response_list={}
@app.route('/recognize_criminal', methods=['POST'])
def recognize_criminal():
    # Get the image from the request
    image_file = request.files.get('image')
    print(image_file)
    #print(request.files)
    if image_file:
        #name = os.path.splitext(image_file)[0]
        #image_path = os.path.join(test_folder,image_file )
        #image = cv2.imread(image_file)
        im=np.frombuffer(image_file.read(), np.uint8)
        image=cv2.imdecode(im, cv2.IMREAD_COLOR)
        
        face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        rects = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30))
        
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
        encodings = face_recognition.face_encodings(image_rgb, boxes)
        #print(encodings)

        # Detect faces in the input image
        face_locations = face_recognition.face_locations(image_rgb)
        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
        #print("Face Encodings")
        #print(face_encodings)
        known_face_names=[]
        criminals_database = load_criminals_data()
        for criminal_id, criminal_details in criminals_database.items():
           names=criminal_details.get('name')
           known_face_names.append(names)
        #print(known_face_names)
        #print(criminals_database)
        matched=[]
        results=[]
        # Iterate through the known criminals and compare face encodings
        for criminal_id, criminal_details in criminals_database.items():
            for criminal_encoding_path in criminal_details['image_paths']:
                known_criminal_encoding = face_recognition.load_image_file(criminal_encoding_path)
                known_criminal_encoding = face_recognition.face_encodings(known_criminal_encoding)[0]
                
                for unknown_encoding in face_encodings:
                    results = face_recognition.compare_faces([known_criminal_encoding], unknown_encoding)
                #print(results)
                    
                if any(results):
                    face_classifier = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    )
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    rects = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1,
                                                  minNeighbors=5, minSize=(30, 30))
                    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
                    #encodings = face_recognition.face_encodings(image_rgb, boxes)
                    myname=criminal_details.get('name')
                    face_names.append(myname)
                    gender=criminal_details.get('gender')
                    dob=criminal_details.get('dob')
                    identification_mark=criminal_details.get('identification_mark')
                    nationality=criminal_details.get('nationality')
                    religion=criminal_details.get('religion')
                    blood_group=criminal_details.get('blood_group')
                    crimes_done=criminal_details.get('crimes_done')
                    name_for_path=myname.lower().replace(" ", "_")+"_rect"
                    output_image_path=os.path.join(output_folder,name_for_path)
                    for ((top, right, bottom, left), name) in zip(boxes, face_names):
                    # draw the predicted face name on the image
                        cv2.rectangle(image, (left, top), (right, bottom),(0, 0, 255), 2)
                        y = top - 15 if top - 15 > 15 else top + 15
                        cv2.putText(image, myname, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 255), 2)
                    new_dimensions = (400,300)
                    resized_image = cv2.resize(image, new_dimensions)
                    cv2.imwrite(output_image_path+".jpg", image)
                    files = os.listdir(output_folder)
                    # Filter only image files (you may need to adjust the condition based on your file types)
                    image_files = [file for file in files if file.lower().endswith(('.jpg', '.png'))]
                    sorted_image_files = sorted(image_files, key=lambda x: os.path.getmtime(os.path.join(output_folder, x)), reverse=True)
                    if sorted_image_files:
                    # Retrieve the path of the last saved image
                        last_saved_image_path = os.path.join(output_folder, sorted_image_files[0])
                        final_path="/"+last_saved_image_path.replace("\\","/")
                    response={
                        'errFlag':1,
                        'status': 'Criminal Recognized',
                        'image_file_name': name_for_path+".jpg",
                        'name':myname,
                        'gender':gender,
                        'dob':dob,
                        'nationality':nationality,
                        'religion':religion,
                        'identification_mark':identification_mark,
                        'crimes_done':crimes_done,
                        'blood_group':blood_group
                    }
                    return response
        response = {
            'errFlag':0,
            'status': 'Criminal not recognized',
            'criminal_id': None,
            'details': None
        }
    else:
        response="No Image File Found"

    return jsonify(response)  

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(1)
capture_active=True

def generate_frames():
    global capture_active
    
    while capture_active:
        
        success, frame = video_capture.read()
        if not success:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        original_small_frame=small_frame.copy()

        processed_frame = detect_faces(small_frame)

        # Convert processed_frame to JPEG format
        ret, processed_buffer = cv2.imencode('.jpg', processed_frame)
        processed_frame_bytes = processed_buffer.tobytes()

        # Convert original_small_frame to JPEG format
        ret, original_buffer = cv2.imencode('.jpg', original_small_frame)
        original_frame_bytes = original_buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + processed_frame_bytes + b'\r\n'
        )
        
    video_capture.release()

def generate_frames_original():
    global capture_active
    print(capture_active)
    while capture_active:
        success, frame = video_capture.read()
        if not success:
            break

        original_frame = frame.copy()

        # Convert original_frame to JPEG format
        ret, original_buffer = cv2.imencode('.jpg', original_frame)
        original_frame_bytes = original_buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + original_frame_bytes + b'\r\n'
        )
        
    video_capture.release()


def detect_faces(frame):
    face_names = ["Ankith"]
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   
    # Detect faces using OpenCV Haar Cascade
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h),name in zip(faces,face_names):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 1)
        #font = cv2.FONT_HERSHEY_DUPLEX
        #cv2.putText(frame, name, (x + 6, y - 6), font, 0.5, (0, 0, 255), 1)
    
    return frame

@app.route('/video_feed_original')
def video_feed_original():
    global capture_active
    capture_active=True
    return Response(generate_frames_original(), content_type='multipart/x-mixed-replace; boundary=frame')
 

@app.route('/processed_frames_feed')
def processed_frames_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_video_capture',methods=['GET'])
def start_video_capture():
    global capture_active

    if not capture_active:
        video_capture.open(1)
        logging.info('Video capture started.')  # Open the video capture if it's not active
        capture_active = True
        return jsonify({'status': 'success', 'message': 'Video capture started.'})
    else:
        logging.warning('Video capture is already active.')
        return jsonify({'status': 'error', 'message': 'Video capture is already active.'})

@app.route('/stop_video_capture',methods=['GET'])
def stop_video_capture():
    global capture_active

    if capture_active:
        capture_active = False
        video_capture.release()  # Release the video capture object
        logging.info('Video capture stopped.')
        return "Video capture stopped."
    else:
        logging.warning('Video capture is not active.')
        return "Video capture is not active."


if __name__ == '__main__':
    app.run(debug=True,port=8080)
 
