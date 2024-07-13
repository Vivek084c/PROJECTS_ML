import cv2

#defining the input video path
video_path = "dataset/data/parking_crop_loop.mp4"

#definig the mask path
mask_path = "dataset/mask_crop.png"

mask = cv2.imread(mask_path, 0)

#creating opencv video capture
cap = cv2.VideoCapture(video_path)

connexted_componenets = cv2.connectedComponents(mask, 4, cv2.CV_32S)

isFrameAvailabe = True
while isFrameAvailabe:
    #reading the frame
    isFrameAvailabe, frame = cap.read()
    
    #visualising the frame
    cv2.imshow("frame",  frame)

    #closing the window if q is pressed
    if cv2.waitKey(25)  & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()