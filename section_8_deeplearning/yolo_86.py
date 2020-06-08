import os
import time
import cv2
import numpy as np
from model.yolo_model import YOLO


def process_image(img):
    """
        resize, reduce and expand image
        # argument:
            img: original image
        # returns
            images: ndarray(64, 64, 3), processed image
    """
    image = cv2.resize(img, (416, 416), interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)
    return image


def get_classes(file):
    """
        get classes name
        # argument:
            file: classes name for database
        # return
            class_names: list, classes name
    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def draw(image, boxes, scores, classes, all_classes):
    """
        # argument:
            image: original image
            boxes: ndarray, boxes of object
            classes: ndarray, classes of objects
            scores: ndarray, scores of object
            all_classes: all classes name
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        top = max(0, np.floor(x+0.5).astype(int))
        left = max(0, np.floor(y+0.5).astype(int))
        right = min(image.shape[1], np.floor(x+w+0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y+h+0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score), (
            top, left-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x, y, w, h'.format(box))

    print()


def dectect_image(image, yolo, all_classes):
    """
        use yolo v3 to detect images
        # arguments:
            image: original image
            yolo: YOLO, yolo model
            all_classes: all classes name
        # returns:  
            image: processed image.
    """
    pimage = process_image(image)
    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()
    print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)
    return image


def dectect_video(video, yolo, all_classes):
    """
        use yolo v3 to detect video
        # argument:
            video: video file
            yolo: YOLO, yolo model
            all_classes: all classes name
    """
    video_path = os.path.join('videos', 'test', video)
    camera = cv2.VideoCapture(video_path)
    cv2.namedWindow('detection', cv2.WINDOW_AUTOSIZE)

    # prepare for saving the detected video
    sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    vout = cv2.VideoWriter_fourcc()
    vout.open(os.path.join('videos', 'res', video), fourcc, 20, sz, True)

    while True:
        res, frame = camera.read()
        if not res:
            break
        image = dectect_image(frame, yolo, all_classes)
        cv2.imshow('detection', image)

        # save the video frame by frame
        vout.write(image)

        if cv2.waitKey(110) & 0xff == 27:
            break
    vout.release()
    camera.release()


yolo = YOLO(0.6, 0.5)
file = '../DATA/coco_classes.txt'
all_classes = get_classes(file)

# detect image
# f = 'jingxiang-gao-489454-unsplash.jpg'
# f = 'demo-2.jpg'
# path = 'images/' + f
# image = cv2.imread(path)
# image = dectect_image(image, yolo, all_classes)
# cv2.imwrite('images/res/' + f, image)

# detect videos one at a time in videos/test folder
video = 'library1.mp4'
dectect_video(video, yolo, all_classes)
