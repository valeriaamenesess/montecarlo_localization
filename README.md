# Localización Monte Carlo (MCL)

Filtro de partículas para localizar un robot en un mapa 2D conocido usando odometría y LiDAR desde Gazebo.

## Archivos

| Archivo | Qué hace |
|---|---|
| `mapa.py` | Genera la imagen del mapa (`mapa_gazebo.png`) con obstáculos y bordes |
| `mundo_mcl.sdf` | Mundo de Gazebo con el robot y LiDAR |
| `mcl.py` | Nodo que corre el filtro de partículas y muestra la visualización |

## Colores en la visualización

- 🟢 **Verde** — partículas del filtro
- 🔴 **Rojo** — estimación MCL (donde cree que está el robot)
- 🔵 **Azul** — pose real del robot (odometría)

## Cómo correrlo

Necesitas **5 terminales**. En cada una corre lo siguiente:

### Terminal 1 — Generar el mapa

```bash
cd ~/montecarlo_localization
python3 mapa.py
```

### Terminal 2 — Lanzar Gazebo

```bash
ign gazebo mundo_mcl.sdf
```

### Terminal 3 — Bridge ROS 2 ↔ Gazebo

```bash
source /opt/ros/humble/setup.bash

ros2 run ros_gz_bridge parameter_bridge \
  /cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist \
  /odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry
```

### Terminal 4 — Correr el MCL

```bash
cd ~/montecarlo_localization
source /opt/ros/humble/setup.bash

python3 mcl.py
```

Se abre una ventana de OpenCV con el mapa y las partículas en tiempo real.

### Terminal 5 — Mover el robot

Tienes dos opciones:

**Opción A: Teleop (con teclado)**

```bash
source /opt/ros/humble/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel
```

**Opción B: Velocidad fija con topic pub**

```bash
source /opt/ros/humble/setup.bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.4}, angular: {z: 0.0}}"
```

> Puedes cambiar los valores de `x` y `z` para ajustar velocidad y giro.

## Requisitos

- Python 3, `numpy`, `opencv-python`
- ROS 2 Humble
- Ignition Gazebo
- `ros_gz_bridge`
- `teleop_twist_keyboard` (si usas la opción A)
