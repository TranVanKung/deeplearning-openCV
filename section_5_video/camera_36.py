import cv2

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    'mysupervideo.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

# window -> DIVX
# macos, linux -> XVID

while True:
    ret, frame = cap.read()
    # operation (drawing)
    writer.write(frame)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('frame', gray)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
