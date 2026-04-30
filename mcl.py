import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data

import numpy as np
import cv2
import math


def euler_from_quaternion(x, y, z, w):
    """
    Convierte un cuaternión a yaw.
    Sólo usamos yaw porque estamos trabajando en navegación 2D.
    """
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return yaw_z


class MCLNode(Node):
    def __init__(self):
        super().__init__('montecarlo_localization_node')

        # =====================================================
        # CONFIGURACIÓN DEL MAPA
        # =====================================================

        # Relación metros/pixel.
        # Tu mapa es de 800x800 px y representa 40x40 m.
        # Por eso: 40 m / 800 px = 0.05 m/px.
        self.resolution = 0.05

        # Cargar mapa conocido
        self.map_img = cv2.imread('mapa_gazebo.png', cv2.IMREAD_GRAYSCALE)

        if self.map_img is None:
            self.get_logger().error("No se encontró 'mapa_gazebo.png'. Ejecuta mapa.py primero.")
            raise FileNotFoundError("mapa_gazebo.png no encontrado.")

        self.map_h, self.map_w = self.map_img.shape

        # =====================================================
        # PARÁMETROS DEL FILTRO DE PARTÍCULAS
        # =====================================================

        self.num_particles = 300
        self.particles = np.zeros((self.num_particles, 4))  # x, y, theta, peso

        self.prev_odom = None
        self.current_robot_pose = None
        self.particles_initialized_with_odom = False

        # Inicialización global temporal.
        # Después, cuando llegue la primera odometría, se reinicializan cerca del robot.
        self.init_particles_global()

        # =====================================================
        # SUSCRIPTORES ROS2
        # =====================================================

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info("Nodo MCL iniciado. Esperando /odom y /scan...")

    # =====================================================
    # CONVERSIONES MAPA <-> MUNDO
    # =====================================================

    def world_to_pixel(self, x, y):
        """
        Convierte coordenadas de Gazebo/mundo a pixeles del mapa.

        Gazebo:
            origen en el centro del mundo.
            x positivo hacia la derecha.
            y positivo hacia arriba.

        Imagen:
            origen en la esquina superior izquierda.
            x positivo hacia la derecha.
            y positivo hacia abajo.
        """
        px = int((x / self.resolution) + self.map_w / 2.0)
        py = int(self.map_h / 2.0 - (y / self.resolution))
        return px, py

    def pixel_to_world(self, px, py):
        """
        Convierte pixeles del mapa a coordenadas del mundo/Gazebo.
        """
        x = (px - self.map_w / 2.0) * self.resolution
        y = (self.map_h / 2.0 - py) * self.resolution
        return x, y

    def is_free_world(self, x, y):
        """
        Verifica si una coordenada del mundo cae en zona libre del mapa.
        """
        px, py = self.world_to_pixel(x, y)

        if px < 0 or px >= self.map_w or py < 0 or py >= self.map_h:
            return False

        return self.map_img[py, px] > 200

    # =====================================================
    # INICIALIZACIÓN DE PARTÍCULAS
    # =====================================================

    def init_particles_global(self):
        """
        Inicialización global en zonas libres del mapa.
        Se usa sólo como estado inicial temporal antes de recibir odometría.
        """
        self.particles = np.zeros((self.num_particles, 4))

        for i in range(self.num_particles):
            while True:
                px = np.random.randint(0, self.map_w)
                py = np.random.randint(0, self.map_h)

                if self.map_img[py, px] > 200:
                    x, y = self.pixel_to_world(px, py)
                    theta = np.random.uniform(-np.pi, np.pi)
                    self.particles[i] = [x, y, theta, 1.0 / self.num_particles]
                    break

        self.get_logger().info(f"Muestreadas {self.num_particles} partículas globales iniciales.")

    def init_particles_around_pose(self, x, y, theta):
        """
        Inicializa partículas alrededor de la pose real inicial recibida por /odom.
        Esto hace que la ventana de Montecarlo empiece cerca de la posición real en Gazebo.
        """
        self.particles = np.zeros((self.num_particles, 4))

        for i in range(self.num_particles):
            px = x + np.random.normal(0, 0.35)       # ruido de 35 cm
            py = y + np.random.normal(0, 0.35)
            ptheta = theta + np.random.normal(0, 0.25)

            # Si cae en pared o fuera del mapa, volvemos a intentar
            attempts = 0
            while not self.is_free_world(px, py) and attempts < 30:
                px = x + np.random.normal(0, 0.35)
                py = y + np.random.normal(0, 0.35)
                ptheta = theta + np.random.normal(0, 0.25)
                attempts += 1

            self.particles[i] = [px, py, ptheta, 1.0 / self.num_particles]

        self.get_logger().info(
            f"Partículas inicializadas alrededor de odom: x={x:.2f}, y={y:.2f}, theta={theta:.2f}"
        )

    # =====================================================
    # ODOMETRÍA / DEAD RECKONING
    # =====================================================

    def odom_callback(self, msg):
        """
        Recibe odometría del robot y mueve las partículas con dead reckoning.
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        theta = euler_from_quaternion(q.x, q.y, q.z, q.w)

        current_odom = np.array([x, y, theta])
        self.current_robot_pose = current_odom

        # Primera lectura de odometría
        if self.prev_odom is None:
            self.prev_odom = current_odom

            if not self.particles_initialized_with_odom:
                self.init_particles_around_pose(x, y, theta)
                self.particles_initialized_with_odom = True

            return

        # Diferencias de odometría
        dx = current_odom[0] - self.prev_odom[0]
        dy = current_odom[1] - self.prev_odom[1]
        dtheta = current_odom[2] - self.prev_odom[2]
        dtheta = math.atan2(math.sin(dtheta), math.cos(dtheta))

        # Movimiento firmado en el eje frontal del robot.
        # Si avanza: positivo.
        # Si retrocede: negativo.
        prev_theta = self.prev_odom[2]
        delta_forward = math.cos(prev_theta) * dx + math.sin(prev_theta) * dy

        if abs(delta_forward) > 0.005 or abs(dtheta) > 0.005:
            self.move_particles_simple(delta_forward, dtheta)
            self.prev_odom = current_odom

    def move_particles_simple(self, delta_forward, delta_theta):
        """
        Mueve las partículas usando dead reckoning simple.
        Permite avance positivo y retroceso negativo.
        """
        noise_forward = 0.01 + 0.05 * abs(delta_forward)
        noise_theta = 0.01 + 0.05 * abs(delta_theta)

        for i in range(self.num_particles):
            old_x = self.particles[i, 0]
            old_y = self.particles[i, 1]
            old_theta = self.particles[i, 2]
            old_is_free = self.is_free_world(old_x, old_y)

            df = delta_forward + np.random.normal(0, noise_forward)
            dt = delta_theta + np.random.normal(0, noise_theta)

            self.particles[i, 0] += df * math.cos(self.particles[i, 2])
            self.particles[i, 1] += df * math.sin(self.particles[i, 2])
            self.particles[i, 2] += dt

            self.particles[i, 2] = math.atan2(
                math.sin(self.particles[i, 2]),
                math.cos(self.particles[i, 2])
            )

            # Si la partícula estaba libre y chocó contra una pared, revertimos su movimiento
            # (Si por ruido apareció dentro de la pared, le permitimos moverse para escapar)
            new_is_free = self.is_free_world(self.particles[i, 0], self.particles[i, 1])
            if old_is_free and not new_is_free:
                self.particles[i, 0] = old_x
                self.particles[i, 1] = old_y
                self.particles[i, 2] = old_theta

    # =====================================================
    # LIDAR / SCORE DE PARTÍCULAS
    # =====================================================

    def scan_callback(self, msg):
        """
        Procesa el LiDAR y asigna puntajes a cada partícula.
        """
        if self.prev_odom is None:
            return

        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        # Ajuste por si el número de ángulos no coincide exactamente con ranges
        n = min(len(angles), len(ranges))
        angles = angles[:n]
        ranges = ranges[:n]

        # Filtrar lecturas válidas (EXCLUIR los rayos que son iguales a range_max porque no golpearon nada)
        valid_idx = (
            (ranges > msg.range_min) &
            (ranges < msg.range_max - 0.1) &
            ~np.isinf(ranges) &
            ~np.isnan(ranges)
        )

        angles = angles[valid_idx]
        ranges = ranges[valid_idx]

        if len(ranges) == 0:
            self.visualize()
            return

        # Usamos una muestra de rayos para no hacerlo tan pesado
        step = max(1, len(ranges) // 60)
        angles = angles[::step]
        ranges = ranges[::step]

        scores = np.zeros(self.num_particles)

        for i in range(self.num_particles):
            x, y, theta, _ = self.particles[i]

            # Si la partícula está fuera del mapa o en pared, score muy bajo
            if not self.is_free_world(x, y):
                scores[i] = 1e-6
                continue

            global_angles = theta + angles

            hit_x = x + ranges * np.cos(global_angles)
            hit_y = y + ranges * np.sin(global_angles)

            score = 0.0

            for hx, hy in zip(hit_x, hit_y):
                px, py = self.world_to_pixel(hx, hy)

                if 0 <= px < self.map_w and 0 <= py < self.map_h:
                    pixel_val = self.map_img[py, px]

                    # Obstáculo negro = buen match para impacto LiDAR
                    score += (255 - pixel_val)
                else:
                    score += 0.0

            scores[i] = score + 1e-6

        # Normalizar scores
        sum_scores = np.sum(scores)

        if sum_scores > 0:
            scores = scores / sum_scores
        else:
            scores = np.ones(self.num_particles) / self.num_particles

        self.particles[:, 3] = scores

        self.resample_particles()
        self.visualize()

    # =====================================================
    # RESAMPLING
    # =====================================================

    def resample_particles(self):
        """
        Filtra la población y conserva las mejores partículas.
        """
        sorted_indices = np.argsort(self.particles[:, 3])[::-1]
        self.particles = self.particles[sorted_indices]

        top_k = max(1, int(self.num_particles * 0.25))
        best_particles = self.particles[:top_k].copy()

        new_particles = np.zeros_like(self.particles)

        # Mantener mejores directamente
        new_particles[:top_k] = best_particles

        # Rellenar el resto clonando mejores con ruido
        for i in range(top_k, self.num_particles):
            parent_idx = np.random.randint(0, top_k)
            parent = best_particles[parent_idx].copy()

            parent[0] += np.random.normal(0, 0.08)
            parent[1] += np.random.normal(0, 0.08)
            parent[2] += np.random.normal(0, 0.05)

            parent[2] = math.atan2(math.sin(parent[2]), math.cos(parent[2]))

            # Si cae en pared, usar una de las mejores sin tanto ruido
            if not self.is_free_world(parent[0], parent[1]):
                parent = best_particles[parent_idx].copy()
                parent[0] += np.random.normal(0, 0.03)
                parent[1] += np.random.normal(0, 0.03)

            new_particles[i] = parent

        new_particles[:, 3] = 1.0 / self.num_particles
        self.particles = new_particles

    # =====================================================
    # VISUALIZACIÓN
    # =====================================================

    def visualize(self):
        """
        Dibuja el mapa, partículas, mejor estimación y pose real del robot.
        """
        vis_map = cv2.cvtColor(self.map_img, cv2.COLOR_GRAY2BGR)

        # Dibujar partículas azules
        for i in range(self.num_particles):
            px, py = self.world_to_pixel(self.particles[i, 0], self.particles[i, 1])

            if 0 <= px < self.map_w and 0 <= py < self.map_h:
                cv2.circle(vis_map, (px, py), 3, (255, 0, 0), -1)

        # Mejor partícula / estimación MCL en rojo
        best_p = self.particles[0]
        bx, by = self.world_to_pixel(best_p[0], best_p[1])

        if 0 <= bx < self.map_w and 0 <= by < self.map_h:
            cv2.circle(vis_map, (bx, by), 8, (0, 0, 255), 2)

            end_x = int(bx + 20 * math.cos(best_p[2]))
            end_y = int(by - 20 * math.sin(best_p[2]))
            cv2.line(vis_map, (bx, by), (end_x, end_y), (0, 0, 255), 2)

        # Pose real del robot según odometría en verde
        if self.current_robot_pose is not None:
            rx, ry, rt = self.current_robot_pose
            rpx, rpy = self.world_to_pixel(rx, ry)

            if 0 <= rpx < self.map_w and 0 <= rpy < self.map_h:
                cv2.circle(vis_map, (rpx, rpy), 7, (0, 255, 0), -1)

                rend_x = int(rpx + 22 * math.cos(rt))
                rend_y = int(rpy - 22 * math.sin(rt))
                cv2.line(vis_map, (rpx, rpy), (rend_x, rend_y), (0, 255, 0), 2)

        # Reducir tamaño si la pantalla es pequeña
        display = cv2.resize(vis_map, (700, 700))

        cv2.imshow("Montecarlo Localization", display)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = MCLNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Cerrando nodo MCL...")
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
