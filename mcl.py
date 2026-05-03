import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data

import numpy as np
import cv2
import math
from collections import deque


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
        # Mapa 800x800 px, mundo 40x40 m -> 0.05 m/pixel.
        self.resolution = 0.05

        # Cargar mapa conocido
        self.map_img = cv2.imread('mapa_gazebo.png', cv2.IMREAD_GRAYSCALE)

        if self.map_img is None:
            self.get_logger().error("No se encontró 'mapa_gazebo.png'. Ejecuta mapa.py primero.")
            raise FileNotFoundError("mapa_gazebo.png no encontrado.")

        self.map_h, self.map_w = self.map_img.shape

        # -------------------------------------------------
        # Likelihood Field:
        # Mapa de obstáculos difuminado con Gaussiana.
        # Valor alto = cerca de pared.
        # Valor bajo = lejos de pared.
        # -------------------------------------------------
        obstacle_binary = (self.map_img < 128).astype(np.float64)

        sigma_pixels = 5.0
        self.likelihood_field = cv2.GaussianBlur(
            obstacle_binary,
            (0, 0),
            sigmaX=sigma_pixels,
            sigmaY=sigma_pixels
        )

        lf_max = self.likelihood_field.max()
        if lf_max > 0:
            self.likelihood_field /= lf_max

        self.get_logger().info(
            f"Likelihood Field calculado. "
            f"Rango: {self.likelihood_field.min():.4f} - {self.likelihood_field.max():.4f}"
        )

        # =====================================================
        # PARÁMETROS DEL FILTRO DE PARTÍCULAS
        # =====================================================

        self.num_particles = 300
        self.particles = np.zeros((self.num_particles, 4))  # x, y, theta, peso

        self.num_rays_subsample = 60
        self.log_field_floor = -10.0

        # Prior de odometría: evita que el MCL se vaya lejos del robot real.
        # Si quieres que el rojo se pegue más al verde, baja estos valores.
        self.odom_sigma_xy = 0.45
        self.odom_sigma_theta = 0.6

        # Offset del LiDAR por si estuviera rotado respecto al robot.
        # Si ves que el MCL se va raro, prueba math.pi, math.pi/2 o -math.pi/2.
        self.lidar_yaw_offset = 0.0

        # Exploración cerca de odometría.
        # Antes estaba en 0.20; lo bajamos para que el filtro no brinque tanto.
        self.exploration_ratio = 0.10

        # Suavizado temporal de la estimación roja.
        # Más alto = más estable, pero responde más lento.
        self.filtered_estimate = None
        self.smooth_alpha = 0.8

        self.estimate_trail = deque(maxlen=80)
        self.mcl_estimate = None

        self.prev_odom = None
        self.current_robot_pose = None
        self.particles_initialized_with_odom = False

        # Inicialización global temporal.
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
            origen en esquina superior izquierda.
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
        Se usa sólo antes de recibir la primera odometría.
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
        Inicializa partículas alrededor de la pose inicial del robot recibida por /odom.
        """
        self.particles = np.zeros((self.num_particles, 4))

        for i in range(self.num_particles):
            px = x + np.random.normal(0, 0.35)
            py = y + np.random.normal(0, 0.35)
            ptheta = theta + np.random.normal(0, 0.25)

            attempts = 0
            while not self.is_free_world(px, py) and attempts < 30:
                px = x + np.random.normal(0, 0.35)
                py = y + np.random.normal(0, 0.35)
                ptheta = theta + np.random.normal(0, 0.25)
                attempts += 1

            self.particles[i] = [px, py, ptheta, 1.0 / self.num_particles]

        self.filtered_estimate = (x, y, theta)
        self.mcl_estimate = (x, y, theta)

        self.get_logger().info(
            f"Partículas inicializadas alrededor de odom: "
            f"x={x:.2f}, y={y:.2f}, theta={theta:.2f}"
        )

    # =====================================================
    # ODOMETRÍA / DEAD RECKONING
    # =====================================================

    def odom_callback(self, msg):
        """
        Recibe odometría y mueve las partículas.
        Usa dx, dy y dtheta globales para que avance y retroceda correctamente.
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

        # Diferencias globales de odometría
        dx = current_odom[0] - self.prev_odom[0]
        dy = current_odom[1] - self.prev_odom[1]

        dtheta = current_odom[2] - self.prev_odom[2]
        dtheta = math.atan2(math.sin(dtheta), math.cos(dtheta))

        # Movimiento directo en coordenadas globales.
        # Esto hace que si el robot va hacia atrás, las partículas también retrocedan.
        if abs(dx) > 0.002 or abs(dy) > 0.002 or abs(dtheta) > 0.005:
            self.move_particles_world(dx, dy, dtheta)
            self.prev_odom = current_odom

    def move_particles_world(self, dx, dy, dtheta):
        """
        Mueve las partículas usando el desplazamiento global de la odometría.
        """
        dist = math.hypot(dx, dy)

        noise_xy = 0.005 + 0.04 * dist
        noise_theta = 0.005 + 0.04 * abs(dtheta)

        for i in range(self.num_particles):
            old_x = self.particles[i, 0]
            old_y = self.particles[i, 1]
            old_theta = self.particles[i, 2]
            old_is_free = self.is_free_world(old_x, old_y)

            new_x = old_x + dx + np.random.normal(0, noise_xy)
            new_y = old_y + dy + np.random.normal(0, noise_xy)
            new_theta = old_theta + dtheta + np.random.normal(0, noise_theta)

            new_theta = math.atan2(math.sin(new_theta), math.cos(new_theta))

            new_is_free = self.is_free_world(new_x, new_y)

            # Si estaba libre y el movimiento la mete en pared, revertimos.
            if old_is_free and not new_is_free:
                self.particles[i, 0] = old_x
                self.particles[i, 1] = old_y
                self.particles[i, 2] = old_theta
            else:
                self.particles[i, 0] = new_x
                self.particles[i, 1] = new_y
                self.particles[i, 2] = new_theta

    # =====================================================
    # LIDAR / SCORE DE PARTÍCULAS
    # =====================================================

    def scan_callback(self, msg):
        """
        Procesa el LiDAR y asigna puntajes usando Likelihood Field + prior de odometría.
        """
        if self.prev_odom is None:
            return

        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        n = min(len(angles), len(ranges))
        angles = angles[:n]
        ranges = ranges[:n]

        # Filtrar lecturas válidas.
        # Quitamos lecturas en range_max porque normalmente indican que no golpearon nada.
        valid_idx = (
            (ranges > msg.range_min) &
            (ranges < msg.range_max - 0.1) &
            ~np.isinf(ranges) &
            ~np.isnan(ranges)
        )

        angles = angles[valid_idx]
        ranges = ranges[valid_idx]

        if len(ranges) == 0:
            self.compute_estimate()
            self.visualize()
            return

        # Submuestrear rayos para rendimiento
        step = max(1, len(ranges) // self.num_rays_subsample)
        angles = angles[::step]
        ranges = ranges[::step]

        map_w = self.map_w
        map_h = self.map_h
        resolution = self.resolution
        half_w = map_w / 2.0
        half_h = map_h / 2.0
        likelihood_field = self.likelihood_field
        log_floor = self.log_field_floor

        log_scores = np.zeros(self.num_particles)

        for i in range(self.num_particles):
            x, y, theta, _ = self.particles[i]

            if not self.is_free_world(x, y):
                log_scores[i] = -1e6
                continue

            # Ángulos globales de los rayos para esta partícula
            global_angles = theta + angles + self.lidar_yaw_offset

            # Proyectar puntos de impacto
            hit_x = x + ranges * np.cos(global_angles)
            hit_y = y + ranges * np.sin(global_angles)

            # Convertir a pixeles
            hit_px = (hit_x / resolution + half_w).astype(int)
            hit_py = (half_h - hit_y / resolution).astype(int)

            in_bounds = (
                (hit_px >= 0) & (hit_px < map_w) &
                (hit_py >= 0) & (hit_py < map_h)
            )

            log_prob = 0.0

            for j in range(len(ranges)):
                if in_bounds[j]:
                    field_val = likelihood_field[hit_py[j], hit_px[j]]
                else:
                    field_val = 0.0

                if field_val > 1e-6:
                    log_prob += math.log(field_val)
                else:
                    log_prob += log_floor

            # Prior de odometría:
            # evita que el filtro se vaya a una hipótesis muy lejos del robot real.
            if self.current_robot_pose is not None:
                rx, ry, rt = self.current_robot_pose

                dist_xy = math.hypot(x - rx, y - ry)
                dtheta_odom = math.atan2(math.sin(theta - rt), math.cos(theta - rt))

                odom_prior_xy = -0.5 * (dist_xy / self.odom_sigma_xy) ** 2
                odom_prior_theta = -0.5 * (dtheta_odom / self.odom_sigma_theta) ** 2

                log_scores[i] = log_prob + odom_prior_xy + odom_prior_theta
            else:
                log_scores[i] = log_prob

        # Convertir log-scores a pesos normalizados
        max_log = np.max(log_scores)
        scores = np.exp(log_scores - max_log)

        sum_scores = np.sum(scores)

        if sum_scores > 0:
            scores = scores / sum_scores
        else:
            scores = np.ones(self.num_particles) / self.num_particles

        self.particles[:, 3] = scores

        self.compute_estimate()
        self.resample_particles()
        self.visualize()

    # =====================================================
    # ESTIMACIÓN
    # =====================================================

    def compute_estimate(self):
        """
        Calcula la estimación MCL usando promedio ponderado de partículas cercanas
        a la odometría y aplica suavizado temporal para que el punto rojo no brinque.
        """
        if self.current_robot_pose is not None:
            rx, ry, rt = self.current_robot_pose

            distances = np.sqrt(
                (self.particles[:, 0] - rx) ** 2 +
                (self.particles[:, 1] - ry) ** 2
            )

            # Usar partículas cercanas a la odometría.
            nearby = np.where(distances < 1.5)[0]

            if len(nearby) > 0:
                selected = self.particles[nearby]
            else:
                selected = self.particles
        else:
            selected = self.particles

        weights = selected[:, 3].copy()
        w_sum = np.sum(weights)

        if w_sum <= 0:
            weights = np.ones(len(selected)) / len(selected)
        else:
            weights = weights / w_sum

        # Promedio ponderado en x, y
        est_x = np.sum(selected[:, 0] * weights)
        est_y = np.sum(selected[:, 1] * weights)

        # Promedio angular ponderado
        sin_sum = np.sum(np.sin(selected[:, 2]) * weights)
        cos_sum = np.sum(np.cos(selected[:, 2]) * weights)
        est_theta = math.atan2(sin_sum, cos_sum)

        # Suavizado temporal
        if self.filtered_estimate is None:
            self.filtered_estimate = (est_x, est_y, est_theta)
        else:
            prev_x, prev_y, prev_theta = self.filtered_estimate

            filt_x = self.smooth_alpha * prev_x + (1.0 - self.smooth_alpha) * est_x
            filt_y = self.smooth_alpha * prev_y + (1.0 - self.smooth_alpha) * est_y

            dtheta = math.atan2(
                math.sin(est_theta - prev_theta),
                math.cos(est_theta - prev_theta)
            )
            filt_theta = prev_theta + (1.0 - self.smooth_alpha) * dtheta
            filt_theta = math.atan2(math.sin(filt_theta), math.cos(filt_theta))

            self.filtered_estimate = (filt_x, filt_y, filt_theta)

        self.mcl_estimate = self.filtered_estimate
        self.estimate_trail.append((self.mcl_estimate[0], self.mcl_estimate[1]))

    # =====================================================
    # RESAMPLING
    # =====================================================

    def resample_particles(self):
        """
        Low-Variance Systematic Resampling con exploración cerca de odometría.
        """
        weights = self.particles[:, 3].copy()
        w_sum = weights.sum()

        if w_sum > 0:
            weights = weights / w_sum
        else:
            weights = np.ones(self.num_particles) / self.num_particles

        N = self.num_particles

        # Low-Variance Systematic Resampling
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0

        r = np.random.uniform(0, 1.0 / N)
        positions = r + np.arange(N) / N

        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, N - 1)

        new_particles = self.particles[indices].copy()

        # Ruido post-resampling reducido para que no brinque tanto
        new_particles[:, 0] += np.random.normal(0, 0.02, N)
        new_particles[:, 1] += np.random.normal(0, 0.02, N)
        new_particles[:, 2] += np.random.normal(0, 0.03, N)

        new_particles[:, 2] = np.arctan2(
            np.sin(new_particles[:, 2]),
            np.cos(new_particles[:, 2])
        )

        # Partículas de exploración alrededor de la odometría
        num_explore = max(1, int(N * self.exploration_ratio))

        if self.current_robot_pose is not None:
            ox, oy, otheta = self.current_robot_pose

            for i in range(num_explore):
                idx = N - 1 - i

                nx = ox + np.random.normal(0, 0.35)
                ny = oy + np.random.normal(0, 0.35)
                ntheta = otheta + np.random.normal(0, 0.25)

                attempts = 0
                while not self.is_free_world(nx, ny) and attempts < 15:
                    nx = ox + np.random.normal(0, 0.35)
                    ny = oy + np.random.normal(0, 0.35)
                    attempts += 1

                new_particles[idx] = [
                    nx,
                    ny,
                    math.atan2(math.sin(ntheta), math.cos(ntheta)),
                    1.0 / N
                ]

        # Verificar que ninguna partícula quede en pared
        for i in range(N):
            if not self.is_free_world(new_particles[i, 0], new_particles[i, 1]):
                parent_idx = indices[i] if i < len(indices) else 0
                new_particles[i, 0] = self.particles[parent_idx, 0]
                new_particles[i, 1] = self.particles[parent_idx, 1]
                new_particles[i, 2] = self.particles[parent_idx, 2]

        new_particles[:, 3] = 1.0 / N
        self.particles = new_particles

    # =====================================================
    # VISUALIZACIÓN
    # =====================================================

    def visualize(self):
        """
        Dibuja mapa, partículas, estimación MCL y pose real por odometría.
        """
        vis_map = cv2.cvtColor(self.map_img, cv2.COLOR_GRAY2BGR)

        # Trail de estimaciones
        trail_list = list(self.estimate_trail)

        for j in range(len(trail_list)):
            tx, ty = trail_list[j]
            tpx, tpy = self.world_to_pixel(tx, ty)

            if 0 <= tpx < self.map_w and 0 <= tpy < self.map_h:
                alpha = int(80 + 175 * (j / max(len(trail_list), 1)))
                cv2.circle(vis_map, (tpx, tpy), 2, (0, alpha, alpha), -1)

        # Partículas azules
        for i in range(self.num_particles):
            px, py = self.world_to_pixel(self.particles[i, 0], self.particles[i, 1])

            if 0 <= px < self.map_w and 0 <= py < self.map_h:
                cv2.circle(vis_map, (px, py), 3, (255, 0, 0), -1)

        # Estimación MCL roja
        if self.mcl_estimate is not None:
            est_x, est_y, est_theta = self.mcl_estimate
            bx, by = self.world_to_pixel(est_x, est_y)

            if 0 <= bx < self.map_w and 0 <= by < self.map_h:
                cv2.circle(vis_map, (bx, by), 8, (0, 0, 255), 2)

                end_x = int(bx + 20 * math.cos(est_theta))
                end_y = int(by - 20 * math.sin(est_theta))
                cv2.line(vis_map, (bx, by), (end_x, end_y), (0, 0, 255), 2)

        # Pose real por odometría verde
        if self.current_robot_pose is not None:
            rx, ry, rt = self.current_robot_pose
            rpx, rpy = self.world_to_pixel(rx, ry)

            if 0 <= rpx < self.map_w and 0 <= rpy < self.map_h:
                cv2.circle(vis_map, (rpx, rpy), 7, (0, 255, 0), -1)

                rend_x = int(rpx + 22 * math.cos(rt))
                rend_y = int(rpy - 22 * math.sin(rt))
                cv2.line(vis_map, (rpx, rpy), (rend_x, rend_y), (0, 255, 0), 2)

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