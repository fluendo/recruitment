import cv2
import numpy as np
import os

path=os.getcwd()
# Cargar el video
cap = cv2.VideoCapture(path +'/rdi/computer_vision_jr/Q_2/video/football.mp4')
if not cap.isOpened(): 
    print("Error al abrir el archivo de video")

# Crear el objeto de sustracción de fondo
backSub = cv2.createBackgroundSubtractorMOG2()

# Definir el rango de color para el equipo A y el equipo B
team_A_color_range = [(200, 200, 200), (255, 255, 255)] 
team_B_color_range = [(0, 0, 100), (100, 100, 200)]

"""
# Definir el tamaños para el filtrado por tamaño
min_contour_area = 200  
max_contour_area = 1500 
"""

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Aplicar sustracción de fondo
    fgMask = backSub.apply(frame)
    # Usar la máscara de primer plano para extraer el primer plano
    foreground = cv2.bitwise_and(frame, frame, mask=fgMask)
    
    # Detectar equipo A
    team_A_mask = cv2.inRange(foreground, team_A_color_range[0], team_A_color_range[1])

    """ Pensé en hacer un filtrado por tamaño a los dos equipos pero debido a los cambios de camara del video, no funcionaria bien

    # Encontrar contornos en la máscara del equipo A
    contours_A, _ = cv2.findContours(team_A_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours_A = [cnt for cnt in contours_A if min_contour_area < cv2.contourArea(cnt) < max_contour_area]
    # Crear una máscara vacía para dibujar los contornos filtrados del equipo A
    filtered_mask_A = np.zeros_like(team_A_mask)
    # Dibujar los contornos filtrados del equipo A en la máscara
    cv2.drawContours(filtered_mask_A, filtered_contours_A, -1, (255), thickness=cv2.FILLED)
    # Aplicar la máscara a la imagen original para obtener la imagen filtrada del equipo A
    team_A = cv2.bitwise_and(foreground, foreground, mask=filtered_mask_A)
    """

    # Aplicamos la mascara 
    team_A = cv2.bitwise_and(foreground, foreground, mask=team_A_mask)


    # Detectar equipo B
    team_B_mask = cv2.inRange(foreground, team_B_color_range[0], team_B_color_range[1])
    
    """ Lo mismo que en el equipo A

    # Encontrar contornos en la máscara del equipo B
    contours_B, _ = cv2.findContours(team_B_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filtrar contornos del equipo B basados en tamaño
    filtered_contours_B = [cnt for cnt in contours_B if min_contour_area < cv2.contourArea(cnt) < max_contour_area]
    # Crear una máscara vacía para dibujar los contornos filtrados del equipo B
    filtered_mask_B = np.zeros_like(team_B_mask)
    # Dibujar los contornos filtrados del equipo B en la máscara
    cv2.drawContours(filtered_mask_B, filtered_contours_B, -1, (255), thickness=cv2.FILLED)
    # Aplicar la máscara a la imagen original para obtener la imagen filtrada del equipo B
    team_B = cv2.bitwise_and(foreground, foreground, mask=filtered_mask_B)
    """

    # Aplicamos la mascara 
    team_B = cv2.bitwise_and(foreground, foreground, mask=team_B_mask)

    # Encontrar contornos
    contours_A, _ = cv2.findContours(team_A_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_B, _ = cv2.findContours(team_B_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dibujar rectángulos de delimitación para cada contorno
    for contour in contours_A:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # cambiar los últimos dos argumentos para cambiar el color y el grosor

    for contour in contours_B:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Mostrar el fotograma original, el equipo A y el equipo B
    cv2.imshow('Fotograma', frame)
    
    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
