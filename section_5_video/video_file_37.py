import cv2
import time

cap = cv2.VideoCapture('mysupervideo.mp4')

if cap.isOpened == False:
    print('File not found or wrong CODEC used')


while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        # writer 30 frame per second
        time.sleep(1/30)
        cv2.imshow('frame', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
