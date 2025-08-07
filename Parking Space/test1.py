import cv2
import numpy as np
import cvzone
import pickle

cap = cv2.VideoCapture('Data/park1.mp4')

drawing = False
area_names = []

try:
    with open("Ajawan", "rb") as f:
        data = pickle.load(f)
        polylines, area_names = data['polylines'], data['area_names']
except FileNotFoundError:
    polylines = []
    # Keep area_names empty if the file doesn't exist

points = []
current_name = " "

def draw(event, x, y, flags, param):
    global points, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_name = input('areaname:-')
        if current_name:
            area_names.append(current_name)
            polylines.append(np.array(points, np.int32))

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', draw)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame = cv2.resize(frame, (1020, 500))

    for i, polyline in enumerate(polylines):
        print(i)
        cv2.polylines(frame, [polyline], True, color=(0, 0, 255), thickness=2)
        cvzone.putTextRect(frame, f'{area_names[i]}', tuple(polyline[0]), 1, 1)

    cv2.imshow('FRAME', frame)

    key = cv2.waitKey(1000) & 0xFF
    if key == ord('s'):
        with open("Ajawan", "wb") as f:
            data = {'polylines': polylines, 'area_names': area_names}
            pickle.dump(data, f)

    elif key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
