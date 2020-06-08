import cv2


# callback function reactangle
def draw_rectangle(event, x, y, flags, param):
    global pt1, pt2, topLeft_clicked, botRight_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # reset the rectangle (check id the rectangle there)
        if topLeft_clicked == True and botRight_clicked == True:
            pt1 = (0, 0)
            pt2 = (0, 0)
            topLeft_clicked = False
            botRight_clicked = False

        if topLeft_clicked == False:
            pt1 = (x, y)
            topLeft_clicked = True

        elif botRight_clicked == False:
            pt2 = (x, y)
            botRight_clicked = True


# global variables
pt1 = (0, 0)
pt2 = (0, 0)
topLeft_clicked = False
botRight_clicked = False

# connect to the callback
cap = cv2.VideoCapture(0)
cv2.namedWindow('Test')
cv2.setMouseCallback('Test', draw_rectangle)


# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# # top left corner
# x = int(width // 2)
# y = int(height // 2)

# # width, height of retangle
# w = int(width // 4)
# h = int(height // 4)
# # bottom right x + w, y + h

while True:
    ret, frame = cap.read()
    # cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=4)
    # cv2.imshow('frame', frame)

    # drawing on the frame based off the global variables
    if topLeft_clicked:
        cv2.circle(frame, center=pt1, radius=5,
                   color=(0, 0, 255), thickness=-1)

    if topLeft_clicked and botRight_clicked:
        cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 3)

    cv2.imshow('Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
