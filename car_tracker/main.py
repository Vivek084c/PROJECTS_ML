import cv2
from utils import get_parking_spot_boxes, is_empty

#defining the input video path
video_path = "dataset/data/parking_1920_1080.mp4"

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


isFrameAvailabe = True
frame_num = 0
while isFrameAvailabe:
    #reading the frame
    isFrameAvailabe, frame = cap.read()

    if frame_num % step == 0:


        #itterating over the spots
        for spot_index, spot in enumerate(spots):
            x1, y1, w, h = spot
            
            #extracting a paticular box from frame
            spot_crop = frame[y1: y1+h, x1: x1+w, :]

            #checking if it is empty or not
            spot_status = is_empty(spot_crop)
            spot_state[spot_index] = spot_status


    for spot_index, spot in enumerate(spots):
        spot_status = spot_state[spot_index]
        x1, y1, w, h = spots[spot_index]
        if spot_status:
            #marking spot green if it is empty
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            #marking spot red if it is not empty
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
            
    
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    #visualising the frame
    cv2.imshow("frame",  frame)


    #closing the window if q is pressed
    if cv2.waitKey(25)  & 0xFF == ord('q'):
        break

    frame_num += 1


cap.release()
cv2.destroyAllWindows()