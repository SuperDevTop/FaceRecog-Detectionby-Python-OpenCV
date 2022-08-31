
import cv2

video = cv2.VideoCapture("face.wmv")

while(1):
    
    ret, frame = video.read()

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    
    if( key == 27):
        print("ESC key entered!!!")
        break


