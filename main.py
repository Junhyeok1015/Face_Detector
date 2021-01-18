import cv2, dlib, sys
import numpy as np
from overlay import *

# resizing 
scaler = 0.3

# detector initialize
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


cap = cv2.VideoCapture("videos/video.mp4")
# load overlay image
overlay = cv2.imread("imgs/ryan_transparent.png", cv2.IMREAD_UNCHANGED)

while True:
    ret, img = cap.read()

    if not ret:
        break

    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
    ori = img.copy()

    # detect faces
    faces = detector(img)
    # get first face from image
    face = faces[0]

    # get face landmarks(feature points)
    dlib_shape = predictor(img, face) # input : image, face
    # save data in 2D form for making calculation easy
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    # compute center of face
    # get top left coordinate of points
    top_left = np.min(shape_2d, axis=0)
    # get bottom right coordinate of points
    bottom_right = np.max(shape_2d, axis=0)

    # get face size
    face_size = int(max(bottom_right - top_left) * 1.8)

    # get the center point of face by calculating every point of landmarks
    # astype int in case of fraction
    center_x, center_y = np.mean(shape_2d , axis=0).astype(np.int)

    # put overlay image on center x, y ,resize by overlay_size and apply the original image
    result = overlay_transparent(ori, overlay, center_x, center_y, overlay_size=(face_size, face_size))


    # Visualize
    # img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color = (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    # Print 68 landmarks on face in circle shape
    for s in shape_2d:
        cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    # floating top left point on image
    cv2.circle(img, center=tuple(top_left), radius = 1, color = (255,255, 0), thickness=2, lineType=cv2.LINE_AA)
    # floating bottom right point on image
    cv2.circle(img, center=tuple(bottom_right), radius = 1, color = (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
    cv2.circle(img, center=(center_x, center_y), radius = 1, color = (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    
    cv2.imshow("img", img)
    cv2.imshow("result", result)
    if cv2.waitKey(1) == ord('q'):
        break