import cv2
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone

# Load the parking area data
with open("Ajawan", "rb") as f:
    data = pickle.load(f)
    polylines, area_names = data['polylines'], data['area_names']

# Load class names
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Load YOLO model
model = YOLO('Model/yolov8s.pt')

# Video file path - you can change this to your video file
video_path = "d:/Yolo8 SPMS V2/Yolo8 SPMS V2/Smart Parking/Videos/park3.mp4"  # Using absolute path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video FPS: {fps}")
print(f"Video Resolution: {frame_width}x{frame_height}")

firebase_data = {
    'occupied_areas': '',
    'free_areas': ''
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame")
        break

    # Resize frame to match IP camera version
    frame = cv2.resize(frame, (1020, 400))
    frame_copy = frame.copy()
    
    # Run YOLO detection
    results = model.predict(frame)
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

    # Display car count and free space
    car_count = len(counter1)
    free_space = len(list2) - car_count
    cvzone.putTextRect(frame_copy, f'CARCOUNTER:-{car_count}', (50, 50), 2, 2)
    cvzone.putTextRect(frame_copy, f'FREESPACE:-{free_space}', (50, 100), 2, 2)

    # Add yellow text in the top middle
    text = "Smart Parking System - Video Mode"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (frame_copy.shape[1] - text_size[0]) // 2
    cv2.putText(frame_copy, text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Update Firebase data (commented out as in original)
    firebase_data['occupied_areas'] = occupied_str[:-2]
    firebase_data['free_areas'] = free_str[:-2]
    # db.reference('/').update(firebase_data)

    # Display the frame
    cv2.imshow('Smart Parking - Video Mode', frame_copy)

    # Control playback speed and exit
    key = cv2.waitKey(30) & 0xFF  # 30ms delay for ~33 FPS playback
    if key == 27:  # Press 'Esc' to exit
        break
    elif key == ord('p'):  # Press 'p' to pause
        cv2.waitKey(0)
    elif key == ord('r'):  # Press 'r' to restart video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Clean up
cap.release()
cv2.destroyAllWindows()
