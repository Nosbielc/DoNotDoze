import cv2

classificador = cv2.CascadeClassifier('cascades//haarcascade_frontalface_default.xml')

imagem = cv2.imread('pessoas//pessoas4.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

facesDectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.1,
                                               minNeighbors=9, minSize=(30,30))
print(len(facesDectadas))
print(facesDectadas)

for (x, y, l, a) in facesDectadas:
    print(x, y, l, a)
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)

cv2.imshow("Faces Encontradas", imagem)
cv2.waitKey()

