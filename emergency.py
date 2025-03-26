import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import random
import os

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,  # Detect only one face
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Define eye landmarks (based on MediaPipe FaceMesh indices)
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Initialize Pygame for audio alert
pygame.mixer.init()
alert_sound_path = "static/alerts/emergency.wav"  # Ensure this file exists or use 'auto' logic
last_alert_time = 0
alert_cooldown = 15  # 15 seconds cooldown

def play_alert(sound_path):
    global last_alert_time
    current_time = time.time()
    if current_time - last_alert_time < alert_cooldown:
        return False

    print("[INFO] Attempting to play alert...")
    if sound_path == "auto":
        alerts_dir = "static/alerts"
        if not os.path.exists(alerts_dir):
            print(f"[ERROR] Alerts directory not found: {alerts_dir}")
            return False

        audio_files = [f for f in os.listdir(alerts_dir) if f.endswith(('.mp3', '.wav'))]
        if not audio_files:
            print(f"[ERROR] No audio files found in {alerts_dir}")
            return False

        selected_file = random.choice(audio_files)
        sound_path = os.path.join(alerts_dir, selected_file)
        print(f"[INFO] Randomly selected alert: {sound_path}")

    if not os.path.exists(sound_path):
        print(f"[ERROR] Alert file not found: {sound_path}")
        return False

    try:
        pygame.mixer.music.stop()
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
        print(f"[INFO] Alert playing from: {sound_path}")
        last_alert_time = current_time
        return True
    except Exception as e:
        print(f"[ERROR] Failed to play alert: {e}")
        return False

def calculate_ear(eye_landmarks, landmarks, frame_width, frame_height):
    """Calculate Eye Aspect Ratio (EAR) for a given eye using normalized landmarks."""
    # Convert normalized coordinates to pixel coordinates
    eye_points = []
    for idx in eye_landmarks:
        landmark = landmarks[idx]
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        eye_points.append((x, y))
    eye_points = np.array(eye_points)

    # Vertical distances (e.g., points 1 to 5, 2 to 4)
    vert_1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vert_2 = np.linalg.norm(eye_points[2] - eye_points[4])
    # Horizontal distance (e.g., points 0 to 3)
    horiz = np.linalg.norm(eye_points[0] - eye_points[3])
    # EAR calculation
    ear = (vert_1 + vert_2) / (2.0 * horiz)
    return ear

def detect_blink_cycle(landmarks, prev_state, cycle_count, start_time, frame_width, frame_height):
    """
    Detect if eyes are blinking continuously.
    Returns updated cycle_count, prev_state, and start_time.
    """
    if not landmarks or len(landmarks.landmark) < 468:
        print("[DEBUG] No valid landmarks detected")
        return cycle_count, prev_state, start_time

    # Calculate EAR for both eyes
    ear_left = calculate_ear(LEFT_EYE, landmarks.landmark, frame_width, frame_height)
    ear_right = calculate_ear(RIGHT_EYE, landmarks.landmark, frame_width, frame_height)
    ear = (ear_left + ear_right) / 2.0
    print(f"[DEBUG] EAR Left: {ear_left:.3f}, EAR Right: {ear_right:.3f}, Average EAR: {ear:.3f}")

    # Define adjustable blink threshold
    EAR_THRESHOLD = 0.25  # Adjust this value (e.g., 0.2 to 0.3) based on debug output
    is_blinking = ear < EAR_THRESHOLD
    print(f"[DEBUG] Is Blinking: {is_blinking}, Threshold: {EAR_THRESHOLD}")

    # Initialize prev_state if None
    if prev_state is None:
        return cycle_count, is_blinking, time.time()

    # Detect a cycle (transition from open to blink or blink to open)
    if prev_state != is_blinking:
        elapsed_time = time.time() - start_time
        print(f"[DEBUG] State changed, Elapsed Time: {elapsed_time:.2f}s")
        if elapsed_time > 5:  # Reset if more than 5 seconds elapsed
            cycle_count = 1
            start_time = time.time()
            print("[DEBUG] Reset cycle count due to timeout")
        else:
            cycle_count += 1
            print(f"[DEBUG] Cycle count incremented to: {cycle_count}")
        return cycle_count, is_blinking, start_time

    return cycle_count, prev_state, start_time

def main():
    # Use the IP webcam URL for video input
    cap = cv2.VideoCapture("http://192.168.31.202:8080/video")
    if not cap.isOpened():
        print("[ERROR] Cannot open IP camera")
        return

    prev_state = None
    cycle_count = 0
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("[ERROR] Failed to read frame")
            break

        # Resize frame for better performance (optional, adjust resolution)
        frame = cv2.resize(frame, (640, 480))  # Adjust if needed
        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Draw landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                )

                # Detect blink cycle
                cycle_count, prev_state, start_time = detect_blink_cycle(face_landmarks, prev_state, cycle_count, start_time, frame_width, frame_height)
                if cycle_count > 3:
                    cv2.putText(frame, "EMERGENCY DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print("[ALERT] Emergency gesture detected! (4+ blinks)")
                    play_alert(alert_sound_path)  # Trigger alert sound
                    cycle_count = 0  # Reset cycle count after alert
                    start_time = time.time()  # Reset timer
                else:
                    cv2.putText(frame, f"Blinks: {cycle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Blink Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()