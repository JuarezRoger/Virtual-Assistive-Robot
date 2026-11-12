import cv2

def test_camera():
    print("Iniciando prueba de cámara...")
    # El índice '0' generalmente selecciona la cámara web principal.
    # Si usas DroidCam u otra cámara externa, podría ser '1' o '2'.
    cap = cv2.VideoCapture(1)

    # Verificar si la cámara se abrió correctamente
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        print("Asegúrate de que no esté siendo usada por otra aplicación (como Zoom o Teams).")
        return

    print("Cámara conectada. Presiona la tecla 'q' en la ventana de video para salir.")

    while True:
        # Leer frame por frame
        ret, frame = cap.read()

        # Si hay un error leyendo el frame, salimos del bucle
        if not ret:
            print("Error: No se pudo recibir el fotograma (frame).")
            break

        # Opcional: Voltear la imagen horizontalmente (efecto espejo)
        # Esto es más natural cuando te ves a ti mismo en la pantalla
        frame = cv2.flip(frame, 1)

        # Mostrar el frame en una ventana llamada 'ATOM - Sensor Visual'
        cv2.imshow('ATOM - Sensor Visual (Prueba)', frame)

        # Esperar 1 milisegundo a que se presione una tecla
        # Si la tecla presionada es 'q', rompemos el bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Cerrando sensor visual...")
            break

    # Liberar la cámara y destruir todas las ventanas creadas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()