import cv2

print(cv2.__version__)

imagen = cv2.imread("opencv-python.jpg")
imagenCinza = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", imagen)
cv2.imshow("Cinza", imagenCinza)
cv2.waitKey()