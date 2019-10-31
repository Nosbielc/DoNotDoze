import sys
import dlib
from skimage import io
import numpy as np
from scipy.spatial import distance
import os

#https://github.com/tahaemara/sleep-detection

def compute_EAR(vec):
    a = distance.euclidean(vec[1], vec[5])
    b = distance.euclidean(vec[2], vec[4])
    c = distance.euclidean(vec[0], vec[3])
    # compute EAR
    ear = (a + b) / (2.0 * c)

    return ear

predictor_path = "shape_predictor_68_face_landmarks.dat"
image_path = './open/'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# initialize the frame counters and the total number of blinks
sleeping = 0
notSleeping = 0
for r, d, f in os.walk(image_path):
    for file in f:
        if '.jpg' in file:
            #win = dlib.image_window()
            img = io.imread(os.path.join(r, file))
            #win.clear_overlay()
            #win.set_image(img)
            dets = detector(img, 1)
            vec = np.empty([68, 2], dtype=int)

            status = "Not Sleeping"

            # print("Number of faces detected: {}".format(len(dets)))
            for k, d in enumerate(dets):
                # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                #     k, d.left(), d.top(), d.right(), d.bottom()))
                # Get the landmarks/parts for the face in box d.
                shape = predictor(img, d)

                # print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                #                                           shape.part(1)))
                # Draw the face landmarks on the screen.

                for b in range(68):
                    vec[b][0] = shape.part(b).x
                    vec[b][1] = shape.part(b).y

                right_ear = compute_EAR(vec[42:48])  # compute eye aspect ratio for right eye
                left_ear = compute_EAR(vec[36:42])  # compute eye aspect ratio for left eye

                if (right_ear + left_ear) / 2 < 0.2:  # if the avarage eye aspect ratio of lef and right eye less than 0.2, the status is sleeping.
                    status = "sleeping"
                    sleeping += 1
                else:
                    notSleeping += 1

                # print(status)
                print("Detection {} => Result: {} ".format(os.path.join(r, file), status))
                #win.add_overlay(shape)
                #win.add_overlay(dets)

print("Resultado Final => Sleeping {} - Not Sleeping: {} ".format(sleeping, notSleeping))

            #win.add_overlay(dets)
            #win.set_title(status)
            #dlib.hit_enter_to_continue()