# Localización Monte Carlo (MCL)

Implementación de un filtro de partículas (Monte Carlo Localization) para localizar un robot diferencial en un mapa 2D conocido, usando datos de odometría y LiDAR provenientes de una simulación en Gazebo.

## Descripción

El proyecto consta de tres componentes principales:

| Archivo | Descripción |
|---|---|
| `mapa.py` | Genera la imagen del mapa conocido (`mapa_gazebo.png`) con obstáculos y bordes. |
| `mundo_mcl.sdf` | Mundo de Gazebo que replica el entorno del mapa, con un robot equipado con LiDAR. |
| `mcl.py` | Nodo ROS 2 que ejecuta el filtro de partículas: recibe `/odom` y `/scan`, puntúa partículas con un Likelihood Field y muestra la estimación en tiempo real. |

### Visualización

- **Azul**: partículas del filtro.
- **Rojo**: estimación MCL (pose estimada).
- **Verde**: pose real del robot según odometría.

## Requisitos

- Python 3
- ROS 2 (Humble o superior)
- Ignition Gazebo
- `ros_gz_bridge`
- Dependencias de Python: `numpy`, `opencv-python`

## Ejecución

Cada comando debe ejecutarse en una **terminal separada**.

### 1. Generar el mapa

```bash
cd ~/montecarlo_localization
python3 mapa.py
```

### 2. Lanzar la simulación en Gazebo

```bash
ign gazebo mundo_mcl.sdf
```

### 3. Iniciar el bridge ROS 2 ↔ Gazebo

```bash
ros2 run ros_gz_bridge parameter_bridge \
  /cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist \
  /odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry \
  /scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan
```

### 4. Mover el robot

```bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: -0.4}, angular: {z: 0.0}}"
```

> Puedes ajustar los valores de `linear.x` y `angular.z` para cambiar la velocidad y dirección del robot.

### 5. Ejecutar la localización MCL

```bash
python3 mcl.py
```

Se abrirá una ventana de OpenCV mostrando el mapa con las partículas, la estimación MCL y la pose real del robot en tiempo real.
