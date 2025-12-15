# Importacion de los modulos necesarios
from ultralytics import YOLO
import cv2

# Leer nuestro modelo
model = YOLO('models/best_v4.pt')

# Realizar Video Captura
cap = cv2.VideoCapture(0)

while True:
    # Leer fotogramas
    ret, frame = cap.read()

    # Leer los resultados de la deteccion
    results = model.predict(frame, imgsz=640, conf=0.80, verbose=False)

    # Mostrar fotogramas
    anotaciones = results[0].plot()
    cv2.imshow('DETECCION Y SEGMENTACION DE OREJAS', anotaciones)

    # Cerrar el video con la tecla ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
