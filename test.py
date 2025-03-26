import cv2
import imutils
import time

IP_WEBCAM_URL = "http://192.168.106.9:8080/video"
cap = cv2.VideoCapture(IP_WEBCAM_URL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = imutils.resize(frame, width=500)
    cv2.imshow("Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        filename = f"harmful/cap_handkerchief_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()