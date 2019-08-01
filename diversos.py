import os
import sleep_detection

path = './validacao/'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.jpeg' in file:
            #files.append(os.path.join(r, file))
            sleep_detection.__init__("shape_predictor_68_face_landmarks.dat", file)


#for f in files:
#    print(f)