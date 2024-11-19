import cv2
from ultralytics import YOLO

# Cargar el modelo YOLO
model = YOLO('yolov8n.pt')  # Asegúrate de tener el modelo accesible

# Abrir la cámara del dispositivo
cap = cv2.VideoCapture(0)  # "0" se refiere a la cámara predeterminada
if not cap.isOpened():
    print("Error: No se puede acceder a la cámara.")
    exit()

while True:
    # Leer fotograma de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede leer el frame.")
        break

    # Procesar el fotograma con el modelo YOLO
    results = model(frame)

    # Extraer información de detecciones
    detections = results[0].boxes.xyxy  # Coordenadas de los cuadros delimitadores
    classes = results[0].boxes.cls      # Clases de las detecciones
    confidences = results[0].boxes.conf # Confianza de las detecciones

    people_count = 0

    for i, cls in enumerate(classes):
        if int(cls) == 0:  # Clase "persona" (ID 0 en el modelo COCO)
            people_count += 1

            # Coordenadas del cuadro delimitador
            x1, y1, x2, y2 = map(int, detections[i])

            # Dibujar cuadro alrededor de la persona
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Agregar etiqueta de confianza
            confidence = confidences[i]
            cv2.putText(frame, f'Person: {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el conteo de personas en la parte superior
    cv2.putText(frame, f"Personas detectadas: {people_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar el fotograma con las detecciones
    cv2.imshow('People Counting - Cámara en Vivo', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos al terminar
cap.release()
cv2.destroyAllWindows()

