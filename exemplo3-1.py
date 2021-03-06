import cv2

video = cv2.VideoCapture(0)

classificadorFace = cv2.CascadeClassifier('cascades//haarcascade_frontalface_default.xml')
classificadorOlhos = cv2.CascadeClassifier('cascades//haarcascade_eye.xml')

while True:
    conectado, frame = video.read()
    #print(conectado)

    frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesDectadas = classificadorFace.detectMultiScale(frameCinza, scaleFactor=1.3, minNeighbors=2, minSize=(70, 70))

    for(x, y, l, a) in facesDectadas:
        imagem = cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
        regiao = imagem[y:y + a, x:x + l]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDectados = classificadorOlhos.detectMultiScale(regiaoCinzaOlho, scaleFactor=1.1,
                                                            minNeighbors=2)

        print(len(olhosDectados))

        if len(olhosDectados) == 2:
            for (ox, oy, ol, oa) in olhosDectados:
                cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (255, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
