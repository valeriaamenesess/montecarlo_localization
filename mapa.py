import numpy as np
import cv2

def generar_mapa():
    # B. Generen un layout de su entorno (se supone que el mapa ya es conocido).
    # Vamos a crear una imagen de 800x800 pixeles.
    # Espacio libre = 255 (blanco), Obstáculos = 0 (negro)
    mapa = np.ones((800, 800), dtype=np.uint8) * 255
    
    # C. Decidan las dimensiones del grid (relación metros/pixel).
    # Supongamos que 1 pixel = 0.05 metros (5 cm).
    # Por lo tanto, este mapa representa un entorno físico de 40x40 metros.
    
    # Dibujamos algunos obstáculos (simulando paredes o cajas del mundo de Gazebo)
    # Formato cv2.rectangle: (x_inicio, y_inicio), (x_fin, y_fin)
    cv2.rectangle(mapa, (200, 200), (300, 300), 0, -1)
    cv2.rectangle(mapa, (500, 400), (600, 600), 0, -1)
    cv2.rectangle(mapa, (100, 600), (400, 650), 0, -1)
    cv2.rectangle(mapa, (600, 100), (700, 150), 0, -1)
    
    # Añadimos bordes exteriores al mundo para evitar que las partículas o el robot salgan
    cv2.rectangle(mapa, (0, 0), (799, 799), 0, 15) # 15 pixeles de grosor en los bordes
    
    # Guardamos el mapa en un archivo
    cv2.imwrite("mapa_gazebo.png", mapa)
    
    print("Mapa generado con éxito.")
    print(" - Archivo guardado como: 'mapa_gazebo.png'")
    print(" - Resolución: 0.05 metros/pixel")
    print(" - Dimensiones de la imagen: 800x800 pixeles")
    print(" - Dimensiones físicas simuladas: 40x40 metros")

if __name__ == '__main__':
    generar_mapa()

