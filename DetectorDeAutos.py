import torch
import cv2 as cv
import numpy as np
import imutils

# Leemos el modelo ya entrenado
from pandas import DataFrame, isnull

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='./model/carros.pt')

# Realizar la Videocaptura con la cámara
cap = cv.VideoCapture("video.mp4")

# Empezamos
while True:
    # Realizamos lectura de frames
    ret, frame = cap.read(60)

    # Podemos hacer corrección de color
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Se realiza las detecciones sobre cada frame
    detect = model(frame)

    info = detect.pandas().xyxy[0]  # im1 predictions
    # info = detect.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    print(info)
    print(len(info))
    index = ('El número de carros es:', len(info.axes[0]))
    print(index)
    # conteo = info.count()
    # print(conteo)


    # Mostramos FPS
    cv.imshow('Detector de Carros', np.squeeze(detect.render()))

    # Leemos el teclado
    t = cv.waitKey(5)
    if t == 27:
        break

cap.release()
cv.destroyAllWindows()