import numpy as np
import cv2

def generar_mapa():
    """Genera el mapa 2D conocido del entorno (800x800 px, 0.05 m/px)."""
    mapa = np.ones((800, 800), dtype=np.uint8) * 255

    # Obstáculos interiores
    cv2.rectangle(mapa, (200, 200), (300, 300), 0, -1)
    cv2.rectangle(mapa, (500, 400), (600, 600), 0, -1)
    cv2.rectangle(mapa, (100, 600), (400, 650), 0, -1)
    cv2.rectangle(mapa, (600, 100), (700, 150), 0, -1)

    # Bordes exteriores
    cv2.rectangle(mapa, (0, 0), (799, 799), 0, 15)

    cv2.imwrite("mapa_gazebo.png", mapa)

    print("Mapa generado con éxito.")
    print(" - Archivo guardado como: 'mapa_gazebo.png'")
    print(" - Resolución: 0.05 metros/pixel")
    print(" - Dimensiones de la imagen: 800x800 pixeles")
    print(" - Dimensiones físicas simuladas: 40x40 metros")

if __name__ == '__main__':
    generar_mapa()
