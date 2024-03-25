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
from PIL import Image
from myClass import objModel,objPredict,objSaveImage
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
    image_file = request.files.getlist('image')
    
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
    #image_file_name=f"{criminal_id}.jpg"
    criminal_folder = os.path.join(known_faces_folder, criminal_id)
    os.makedirs(criminal_folder, exist_ok=True)

    for i, image_files in enumerate(image_file, 1):
        # Save each image inside the criminal's folder
        image_file_name = f"{criminal_id}_{i}.jpg"
        image_path = os.path.join(criminal_folder, image_file_name)
        image_files.save(image_path)
        image_paths.append(image_path)
    #image_path = os.path.join(criminal_folder, image_file_name)
    #image_file.save(image_path)
    #im=cv2.imread(image_path)
    #cv2.imwrite(image_path,im)
    #image_paths.append(image_path)
    details['image_paths'] = image_paths
    # Store the details in the criminals database
    criminals_database[criminal_id] = details

    save_criminals_data()

    print("Training KNN classifier...")
    classifier = objModel.train("known_faces", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")
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
        im=np.frombuffer(image_file.read(), np.uint8)
        image=cv2.imdecode(im, cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        predictions = objPredict.predict(image_rgb, model_path="trained_knn_model.clf")
        print(predictions)
        objSaveImage.show_prediction_labels_on_image(pil_image,predictions=predictions)
        
        result=[]
        criminal_folder_path="criminals_data"
        for name, (top, right, bottom, left) in predictions:
             
            if name == "unknown":
                response = {
                'errFlag':0,
                'status': 'Criminal not recognized',
                'criminal_id': None,
                'details': None
            }

            #print(name)
            if name!="unknown":
                criminal_data_folder=os.path.join(criminal_folder_path,name)
                json_file_path=os.path.join(criminal_data_folder,"details.json")
                print(json_file_path)
                if os.path.exists(json_file_path):
                    with open(json_file_path, 'r') as file:
                        data = json.load(file)
                        print(data)
                    res={
                        "errFlag":1,
                        "criminal_name" : data['name'],
                        "blood_group" : data['blood_group'],
                        "crimes_done" : data['crimes_done'],
                        "dob" : data['dob'],
                        "gender" : data['gender'],
                        "identification_mark": data['identification_mark'],
                        "image_path" : name+"_rect.jpg",
                        "nationality" : data['nationality'],
                        "religion" : data['religion']
                        }
                result.append(res)
                response=result
        print(response)
            
                
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
 
