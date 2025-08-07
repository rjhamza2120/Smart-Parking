import cv2
import threading
import time
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO

import cvzone
import queue


class WebcamCapture:
    def __init__(self, capture, callback=None):
        self.capture = capture
        self.callback = callback
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.start()

    def _capture_frames(self):
        count = 0
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                print("Error: Failed to read a frame from the camera stream.")
                break

            count += 1
            if count % 3 != 0:
                continue

            frame = cv2.resize(frame, (1020, 400))
            with self.lock:
                self.latest_frame = frame.copy()
            self.queue.put(frame)

            if self.callback:
                self.callback(frame)

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame

    def get_next_frame(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.thread.join()


with open("Ajawan", "rb") as f:
    data = pickle.load(f)
    polylines, area_names = data['polylines'], data['area_names']

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

model = YOLO('Model/yolov8s.pt')

# Replace the URL with the actual URL of your IP camera
ip_cam_url = ""  #Add your IP camera URL here
cap = cv2.VideoCapture(ip_cam_url)

if not cap.isOpened():
    print("Error: Could not open IP camera stream.")
    exit()

# Initialize the WebcamCapture class
webcam_capture = WebcamCapture(cap)

count = 0

firebase_data = {
    'occupied_areas': '',
    'free_areas': ''
}

while True:
    img = webcam_capture.get_latest_frame()
    if img is not None:
        frame_copy = img.copy()
        results = model.predict(img)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")
        list1 = []
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2
            if 'car' in c:
                list1.append([cx, cy])

        counter1 = []
        list2 = []
        occupied_str = "Occupied areas: "
        free_str = "Free areas: "
        for i, polyline in enumerate(polylines):
            list2.append(i)
            is_occupied = False  # Flag to check if the space is occupied
            cv2.polylines(frame_copy, [polyline], True, color=(0, 255, 0), thickness=2)
            for i1 in list1:
                cx1 = i1[0]
                cy1 = i1[1]
                result = cv2.pointPolygonTest(polyline, ((cx1, cy1)), False)
                if result >= 0:
                    cv2.circle(frame_copy, (cx1, cy1), 5, (255, 0, 0), -1)
                    cv2.polylines(frame_copy, [polyline], True, color=(0, 0, 255), thickness=2)
                    counter1.append(cx1)
                    is_occupied = True  # Set the flag if the space is occupied

            color = (0, 255, 0) if not is_occupied else (0, 0, 255)

            cv2.putText(frame_copy, f'{area_names[i]}', tuple(polyline[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                        cv2.LINE_AA)

            if not is_occupied:
                free_str += f"{area_names[i]}, "
            else:
                occupied_str += f"{area_names[i]}, "

        # Print the occupied and free area names
       # cvzone.putTextRect(frame_copy, occupied_str[:-2], (50, 260), 1, 1, (0, 0, 255), 2)  # Print occupied areas
       # cvzone.putTextRect(frame_copy, free_str[:-2], (50, 300), 1, 1, (0, 255, 0), 2)  # Print free areas

        # Rest of the code remains unchanged
        car_count = len(counter1)
        free_space = len(list2) - car_count
        cvzone.putTextRect(frame_copy, f'CARCOUNTER:-{car_count}', (50, 50), 2, 2)
        cvzone.putTextRect(frame_copy, f'FREESPACE:-{free_space}', (50, 100), 2, 2)

        # Add yellow text in the top middle
        text = "Smart Parking System"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (frame_copy.shape[1] - text_size[0]) // 2
        cv2.putText(frame_copy, text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Push data to Firebase
        firebase_data['occupied_areas'] = occupied_str[:-2]
        firebase_data['free_areas'] = free_str[:-2]
        # db.reference('/').update(firebase_data)

        # Get real-time data from Firebase
        # real_time_data = db.reference('/').get()
        # print("Real-time data from Firebase:", real_time_data)

        cv2.imshow('FRAME', frame_copy)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break

# Stop the webcam capture thread
webcam_capture.stop()
cap.release()
cv2.destroyAllWindows()
