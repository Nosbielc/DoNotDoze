import cv2

classificadorFace = cv2.CascadeClassifier('cascades//haarcascade_frontalface_default.xml')
classificadorOlhos = cv2.CascadeClassifier('cascades//haarcascade_eye.xml')

imagem = cv2.imread('pessoas//pessoas4.jpg')

imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

faceDectadas = classificadorFace.detectMultiScale(imagemCinza , scaleFactor=1.1,
                                               minNeighbors=9, minSize=(30,30))

for (x , y, l, a) in faceDectadas:
    imagem = cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
    regiao = imagem[y:y+a, x:x+ l]
    regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
    olhosDectados = classificadorOlhos.detectMultiScale(regiaoCinzaOlho, scaleFactor=1.1,
                                               minNeighbors=2)
    print(len(olhosDectados))
    print(olhosDectados)
    for (ox, oy, ol, oa) in olhosDectados:
        cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (255, 0, 255), 2)

cv2.imshow("Faces e olhos detectados", imagem)
cv2.waitKey()

