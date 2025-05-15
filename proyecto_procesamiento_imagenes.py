import cv2
import numpy as np

# Cargar el clasificador Haar para detección de rostro
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar la imagen del sombrero (debe ser PNG con fondo transparente)
hat = cv2.imread("Proyecto\hat.webp", cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
else:
    print("Cámara iniciada correctamente")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el cuadro")
        break

    # Convertir a escala de grises para la detección de rostro
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Redimensionar el sombrero proporcional al ancho del rostro detectado
        hat_width = w
        hat_height = int(hat.shape[0] * (hat_width / hat.shape[1]))
        resized_hat = cv2.resize(hat, (hat_width, hat_height), interpolation=cv2.INTER_AREA)

        # Definir el lugar donde se colocará el sombrero (encima del rostro)
        y1 = max(0, y - hat_height)
        y2 = y1 + hat_height
        x1 = x
        x2 = x1 + hat_width

        # Comprobar los límites de la imagen
        if y2 > frame.shape[0] or x2 > frame.shape[1]:
            continue

        # Extraer canales de la imagen del sombrero (RGBA)
        hat_rgba = resized_hat[:, :, :4]
        hat_rgb = resized_hat[:, :, :3]
        mask = resized_hat[:, :, 3]  # Canal alpha como máscara

        roi = frame[y1:y2, x1:x2]

        # Crear máscaras para fusionar
        mask_inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(hat_rgb, hat_rgb, mask=mask)

        # Combinar ROI y sombrero
        combined = cv2.add(bg, fg)
        frame[y1:y2, x1:x2] = combined

    cv2.imshow('Filtro de Sombrero en Tiempo Real', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.imwrite('captura_con_sombrero.jpg', frame)
        break

cap.release()
cv2.destroyAllWindows()