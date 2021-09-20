import cv2 as cv
import numpy as np
import mediapipe as mp
import time

mp_objectron = mp.solutions.objectron  # for detecting the objects
mp_draw = mp.solutions.drawing_utils  # for drawing 3-D boxes to classify them

capture = cv.VideoCapture(0)

with mp_objectron.Objectron(static_image_mode=False, max_num_objects=3, min_detection_confidence=0.5,
                            min_tracking_confidence=0.8, model_name='Chair') as objectron:
    while capture.isOpened():
        success, frame = capture.read()
        start = time.time()

        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        img.flags.writeable = False
        results = objectron.process(img)

        img.flags.writeable = True
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        if results.detected_objects:
            for detected_objects in results.detected_objects:
                mp_draw.draw_landmarks(img, detected_objects.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_draw.draw_axis(img, detected_objects.rotation, detected_objects.translation)

        end = time.time()
        Total = end - start
        fps = 1 / Total

        cv.imshow('MediaPipe', img)
        if cv.waitKey(5) & 0xFF == 27:
            break

capture.release()
