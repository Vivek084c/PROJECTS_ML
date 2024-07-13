import cv2
from utils import get_parking_spot_boxes, is_empty, calc_dif
import numpy as np
import matplotlib.pyplot as plt



#defining the input video path
video_path = "dataset/data/parking_1920_1080_loop.mp4"

#definig the mask path
mask_path = "dataset/mask_1920_1080.png"

mask = cv2.imread(mask_path, 0)

#creating opencv video capture
cap = cv2.VideoCapture(video_path)

connexted_componenets = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spot_boxes(connexted_componenets)


#performing classification in every second not 30 times in every second
step = 30
spot_state = [None for j in spots]    
diffs = [None for j in spots]   
previous_frame = None


isFrameAvailabe = True
frame_num = 0
while isFrameAvailabe:
    #reading the frame
    isFrameAvailabe, frame = cap.read()

    if frame_num % step == 0 and previous_frame is not None:
        #itterating over the spots
        for spot_index, spot in enumerate(spots):
            x1, y1, w, h = spot
            
            #extracting a paticular box from frame
            spot_crop = frame[y1: y1+h, x1: x1+w, :]

            diffs[spot_index] =  calc_dif(spot_crop, previous_frame[y1: y1+h, x1: x1+w, :])
       
        

    if frame_num % step == 0:
        if previous_frame is None:
            arr = range(len(spots))
        else:
            arr = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4][::-1]

        #itterating over the spots
        for spot_index in arr:
            spot = spots[spot_index]
            x1, y1, w, h = spot
            
            #extracting a paticular box from frame
            spot_crop = frame[y1: y1+h, x1: x1+w, :]

            #checking if it is empty or not
            spot_status = is_empty(spot_crop)
            spot_state[spot_index] = spot_status

    if frame_num % step == 0:
        previous_frame = frame.copy()

    for spot_index, spot in enumerate(spots):
        spot_status = spot_state[spot_index]
        x1, y1, w, h = spots[spot_index]
        if spot_status:
            #marking spot green if it is empty
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            #marking spot red if it is not empty
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)     
    cv2.putText(frame, "Available spots: {} / {}".format(str(sum(spot_state)), str(len(spot_state))), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    #visualising the frame
    cv2.imshow("frame",  frame)


    #closing the window if q is pressed
    if cv2.waitKey(25)  & 0xFF == ord('q'):
        break

    frame_num += 1


cap.release()
cv2.destroyAllWindows()