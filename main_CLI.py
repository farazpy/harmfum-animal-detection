import cv2
import imutils
import numpy as np
import time
from collections import deque
from threading import Thread
from queue import Queue
from imutils.video import FPS
import os
from twilio.rest import Client
import pygame
import mysql.connector
from mysql.connector import Error
import random

# Database connection function
def connect_db():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="webmaster",
            password="iFFP@1692",
            database="farm_surveillance"
        )
        if connection.is_connected():
            print("[INFO] Connected to MySQL database")
            return connection
    except Error as e:
        print(f"[ERROR] Failed to connect to MySQL: {e}")
        return None

# Fetch configuration from database
def fetch_config(connection):
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM config WHERE id = 1")
    config = cursor.fetchone()
    if config:
        # Ensure all necessary keys exist with defaults
        config.setdefault('ip_webcam_url', 'http://192.168.31.202:8080/video')
        config.setdefault('camera_name', 'IP Webcam')
        config.setdefault('proto_path', 'MobileNetSSD_deploy.prototxt')
        config.setdefault('model_path', 'MobileNetSSD_deploy.caffemodel')
        config.setdefault('harmful_dir', 'static/harmful_faces')
        config.setdefault('req_classes', 'person')
        config.setdefault('conf_thresh', 0.2)
        config.setdefault('alert_cooldown', 30)
        config.setdefault('siren_path', 'auto')
        config.setdefault('farm_owner_number', '+1234567890')  # Replace with actual number
        config.setdefault('twilio_sid', 'your_twilio_sid')
        config.setdefault('twilio_token', 'your_twilio_token')
        config.setdefault('twilio_number', '+0987654321')  # Replace with Twilio number
    cursor.close()
    return config

# Log detection to database
def log_detection(connection, detected_time, detected_type, confidence=None, footage_path=None, face_id=None, object_id=None):
    cursor = connection.cursor()
    query = """
    INSERT INTO detection_logs (detected_time, detected_type, confidence, footage_path, face_id, object_id)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    confidence_value = float(confidence) if confidence is not None else 0.0
    cursor.execute(query, (detected_time, detected_type, confidence_value, footage_path, face_id, object_id))
    connection.commit()
    cursor.close()
    print(f"[INFO] Logged detection: {detected_type} at {detected_time}, Footage: {footage_path}, Face ID: {face_id}, Object ID: {object_id}")

# Load the pre-trained model for object detection
def load_model(proto_path, model_path):
    if not os.path.exists(proto_path):
        raise FileNotFoundError(f"Prototxt file not found: {proto_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Caffe model file not found: {model_path}")
    return cv2.dnn.readNetFromCaffe(proto_path, model_path)

# Load known faces and objects from database and directory
def load_known_items(connection, harmful_dir):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    known_faces = []
    known_face_labels = []
    face_label_map = {}
    known_objects = {}  # Store object images as a reference (simplified)

    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT id, filename FROM known_faces")
    face_entries = cursor.fetchall()
    cursor.close()

    if not os.path.exists(harmful_dir):
        os.makedirs(harmful_dir)
        print(f"Created '{harmful_dir}' directory. Please add images of known faces or objects.")
        return face_recognizer, known_faces, known_face_labels, face_label_map, known_objects

    for entry in face_entries:
        filename = entry['filename']
        item_id = entry['id']
        path = os.path.join(harmful_dir, filename)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is None:
                print(f"[ERROR] Failed to load image: {filename}")
                continue

            # Check if it's a face
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100, 100))
                known_faces.append(face_roi)
                known_face_labels.append(item_id)
                face_label_map[item_id] = filename.split('.')[0]
                print(f"Loaded face: {filename} with ID {item_id}")
            else:
                # Treat as object (store the full image as a reference)
                known_objects[item_id] = img
                print(f"Loaded object: {filename} with ID {item_id}")

    if known_faces:
        face_recognizer.train(known_faces, np.array(known_face_labels))
        print("[INFO] Face recognizer trained successfully")
    else:
        print("[INFO] No faces loaded for training")

    return face_recognizer, known_faces, known_face_labels, face_label_map, known_objects

# Send an SMS using Twilio
def send_sms(phone_number, detected_time, twilio_sid, twilio_token, twilio_number):
    try:
        client = Client(twilio_sid, twilio_token)
        message = client.messages.create(
            body=f"Alert! Recognized face or object detected on your farm at {detected_time}.",
            from_=twilio_number,
            to=phone_number
        )
        print("[INFO] SMS sent successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send SMS: {e}")
        return False



def play_siren(siren_path):
    print("[INFO] Attempting to play siren...")

    # Check if siren_path is "auto" to select a random file from static/alerts
    if siren_path == "auto":
        alerts_dir = "static/alerts"
        if not os.path.exists(alerts_dir):
            print(f"[ERROR] Alerts directory not found: {alerts_dir}")
            return False

        audio_files = [f for f in os.listdir(alerts_dir) if f.endswith(('.mp3', '.wav'))]
        if not audio_files:
            print(f"[ERROR] No audio files found in {alerts_dir}")
            return False

        selected_file = random.choice(audio_files)
        siren_path = os.path.join(alerts_dir, selected_file)
        print(f"[INFO] Randomly selected siren: {siren_path}")

    if not os.path.exists(siren_path):
        print(f"[ERROR] Siren file not found: {siren_path}")
        return False

    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
            print("[INFO] Pygame mixer initialized")
        pygame.mixer.music.stop()
        pygame.mixer.music.load(siren_path)
        pygame.mixer.music.play()
        print(f"[INFO] Siren playing from: {siren_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to play siren: {e}")
        return False



# Face matching detection using Haar Cascade and LBPH
def face_matching(frame, face_recognizer, label_map):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected = False
    matched_name = None
    matched_id = None

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (100, 100))

        label, confidence = face_recognizer.predict(face_roi)
        if confidence < 80:  # Lower threshold for stricter matching
            detected = True
            matched_id = label
            matched_name = label_map.get(label, "unknown")
            confidence_score = (100 - confidence) / 100
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, f"Face: {matched_name} ({confidence_score:.2f})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print(f"Detected face: {matched_name} with ID {matched_id}, confidence: {confidence_score:.2f}")

    return detected, frame, matched_name, matched_id

# Object matching using template matching (simplified)
def object_matching(frame, known_objects, threshold=0.7):
    detected = False
    matched_id = None
    matched_name = None

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for item_id, template in known_objects.items():
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.resize(template_gray, (100, 100))  # Resize to a standard size
        result = cv2.matchTemplate(gray_frame, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            detected = True
            matched_id = item_id
            matched_name = os.path.splitext(list(known_objects.keys())[list(known_objects.values()).index(template)])[0]
            top_left = max_loc
            h, w = template_gray.shape
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, f"Object: {matched_name} ({max_val:.2f})", 
                       (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Detected object: {matched_name} with ID {matched_id}, confidence: {max_val:.2f}")

    return detected, frame, matched_name, matched_id

# Record footage function
def record_footage(frame_queue, output_path, duration=10, fps=10, frame_width=320, frame_height=240):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    start_time = time.time()
    frame_count = 0
    target_frames = int(duration * fps)

    while frame_count < target_frames and time.time() - start_time < duration:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame = cv2.resize(frame, (frame_width, frame_height))
            out.write(frame)
            frame_count += 1
        
        elapsed_time = time.time() - start_time
        target_time = frame_count / fps
        if elapsed_time < target_time:
            time.sleep(target_time - elapsed_time)

    out.release()
    print(f"[INFO] Footage saved to: {output_path} (Recorded {frame_count} frames)")

# Video stream reader thread with optimized resolution
def video_stream_reader(video_capture, frame_queue):
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        success, frame = video_capture.read()
        if not success:
            print("[ERROR] Failed to read frame from video stream.")
            time.sleep(1)
            video_capture.release()
            video_capture.open(config['ip_webcam_url'])
            video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        frame = imutils.resize(frame, width=320)
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)

# Detection worker thread for offloading heavy processing
def detection_worker(frame_queue, detection_queue, frame_skip, net, face_recognizer, label_map, known_objects, req_classes, classes, colors, config):
    while True:
        if frame_queue.empty():
            time.sleep(0.001)
            continue

        frame = frame_queue.get()
        det_dnn = 0
        det_face = 0
        det_object = 0
        detected_confidence = None
        detected_idx = None
        matched_name = None
        matched_face_id = None
        matched_object_id = None

        # DNN detection for objects in req_classes
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                                   0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        (h, w) = frame.shape[:2]
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > config['conf_thresh']:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                if classes[idx] in req_classes:
                    det_dnn = 1
                    detected_confidence = confidence
                    detected_idx = idx
                    label = f"{classes[idx]}: {confidence * 100:.2f}%"
                    cv2.rectangle(frame, (startX, startY), (endX, endY), 
                                colors[idx], 2)
                    y = startY - 15 if (startY - 15) > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

        # Face matching detection
        det_face, frame, matched_name, matched_face_id = face_matching(frame, face_recognizer, label_map)

        # Object matching using template matching
        det_object, frame, matched_name, matched_object_id = object_matching(frame, known_objects)

        # Combine detections
        det = det_dnn or det_face or det_object
        detection_queue.put((det, frame, detected_confidence, detected_idx, matched_name, matched_face_id, matched_object_id))

# Main detection function
def detect_animals():
    global config
    db_connection = connect_db()
    if not db_connection:
        return

    config = fetch_config(db_connection)
    if not config:
        print("[ERROR] Failed to fetch configuration from database.")
        return

    frame_skip = config.get('frame_skip', 15)
    print(f"[INFO] Using frame skip: {frame_skip}")

    try:
        net = load_model(config['proto_path'], config['model_path'])
    except FileNotFoundError as e:
        print(e)
        return

    face_recognizer, known_faces, known_face_labels, face_label_map, known_objects = load_known_items(db_connection, config['harmful_dir'])
    
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
               "car", "cat", "chair", "cow", "dining-table", "dog", "horse", 
               "motorbike", "person", "potted plant", "sheep", "sofa", "train", "monitor"]
    REQ_CLASSES = set(config['req_classes'].split(','))
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    print(f"[INFO] Starting video stream from {config['camera_name']}...")
    vs = cv2.VideoCapture(config['ip_webcam_url'])
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    vs.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    time.sleep(1.0)
    fps = FPS().start()

    frame_queue = Queue(maxsize=10)
    detection_queue = Queue(maxsize=10)

    reader_thread = Thread(target=video_stream_reader, args=(vs, frame_queue), daemon=True)
    reader_thread.start()

    detection_thread = Thread(target=detection_worker, args=(frame_queue, detection_queue, frame_skip, net, face_recognizer, face_label_map, known_objects, REQ_CLASSES, CLASSES, COLORS, config), daemon=True)
    detection_thread.start()

    detection_buffer = deque(maxlen=3)
    last_detection_time = 0
    footage_dir = "static/footage"
    if not os.path.exists(footage_dir):
        os.makedirs(footage_dir)

    while True:
        if detection_queue.empty():
            cv2.waitKey(1)
            continue

        det, frame, detected_confidence, detected_idx, matched_name, matched_face_id, matched_object_id = detection_queue.get()
        (h, w) = frame.shape[:2]

        current_time = time.time()
        if current_time - last_detection_time < config.get('alert_cooldown', 30):
            remaining_time = config.get('alert_cooldown', 30) - (current_time - last_detection_time)
            cv2.putText(frame, f"Cooldown: {remaining_time:.1f}s remaining", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Animal Detection", frame)
            cv2.waitKey(1)
            continue

        detection_buffer.append(det)

        if det and (current_time - last_detection_time > config.get('alert_cooldown', 30)):
            detected_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[ALERT] Intrusion Detected at {detected_time}!")
            detected_type = CLASSES[detected_idx] if detected_idx is not None else (matched_name if matched_name else "unknown")
            
            if play_siren(config['siren_path']):
                    print("[INFO] Siren triggered successfully")
            else:
                    print("[ERROR] Siren failed to trigger")

            # Prepare footage recording
            footage_filename = f"{detected_time.replace(':', '-')}.avi"
            footage_path = os.path.join(footage_dir, footage_filename)
            
            # Start recording in a separate thread using the frame queue
            recording_thread = Thread(target=record_footage, 
                                    args=(frame_queue, footage_path, 10, 10, w, h), 
                                    daemon=True)
            recording_thread.start()

            # Log detection with footage path, face ID, and object ID
            log_detection(db_connection, detected_time, detected_type, detected_confidence, footage_path, matched_face_id, matched_object_id)

            # Trigger alarm if a known face or object is detected
            if matched_face_id is not None or matched_object_id is not None:
                print(f"[ALERT] Recognized item detected: {matched_name} (Face ID: {matched_face_id}, Object ID: {matched_object_id})")
                if play_siren(config['siren_path']):
                    print("[INFO] Siren triggered successfully")
                else:
                    print("[ERROR] Siren failed to trigger")
                # Uncomment to enable SMS
                # send_sms(config['farm_owner_number'], detected_time, 
                #          config['twilio_sid'], config['twilio_token'], config['twilio_number'])

            last_detection_time = current_time

        cv2.imshow("Animal Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        fps.update()

    fps.stop()
    print(f"[INFO] Elapsed time: {fps.elapsed():.2f} seconds")
    print(f"[INFO] Approx. FPS: {fps.fps():.2f}")
    vs.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    db_connection.close()

if __name__ == "__main__":
    pygame.mixer.init()
    detection_thread = Thread(target=detect_animals, daemon=True)
    detection_thread.start()
    detection_thread.join()