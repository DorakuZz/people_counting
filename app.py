import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import tempfile
import numpy as np

# Cargar el modelo YOLOv8 preentrenado
model = YOLO('yolov8n.pt')

# Función para detectar personas en un cuadro
def detect_people(frame):
    results = model(frame)  # Procesar con YOLO
    detections = results[0].boxes.xyxy  # Coordenadas de detección
    classes = results[0].boxes.cls  # Clases detectadas
    confidences = results[0].boxes.conf  # Confianza de detección

    people_count = 0
    for i, cls in enumerate(classes):
        if int(cls) == 0:  # Clase "persona"
            people_count += 1
            x1, y1, x2, y2 = map(int, detections[i])
            confidence = confidences[i]
            # Dibujar un rectángulo alrededor de la persona
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person: {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame, people_count

# Interfaz de Streamlit
st.title("People Counting App")

# Selección de modo
option = st.sidebar.selectbox("Selecciona una opción", ["Subir archivo", "Usar cámara en vivo"])

if option == "Subir archivo":
    st.header("Subir archivo de video")
    uploaded_file = st.file_uploader("Sube tu archivo de video", type=["mp4", "avi", "mov"])

    if uploaded_file:
        # Guardar el archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(uploaded_file.read())
            video_path = tmpfile.name

        # Procesar el video
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame, count = detect_people(frame)
            frames.append(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            st.image(frames[-1], caption=f"Personas detectadas: {count}", use_column_width=True)

        cap.release()

elif option == "Usar cámara en vivo":
    st.header("Usar cámara en vivo")
    if st.button("Iniciar cámara"):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("No se pudo acceder a la cámara.")
        else:
            st.write("Presiona 'Stop' para detener la cámara.")

        stop_button = st.button("Detener cámara")
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("No se pudo capturar el cuadro.")
                break
            processed_frame, count = detect_people(frame)
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption=f"Personas detectadas: {count}", use_column_width=True)

        cap.release()
