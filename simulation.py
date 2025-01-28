import holoocean
import holoocean.environments
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from pynput import keyboard
import sys
sys.path.append('/home/lh/Desktop/HoloOceanUtils')
import holoOceanUtils
import teaserpp_python
from scipy.spatial.transform import Rotation as Rot
import os

class ROVSimulator:
    def __init__(self, scenario="Dam-HoveringImagingSonar", force=25, max_intensity=0.5, auto_control=True, sonar_image=True):
        self.scenario = scenario
        self.sonar_image = sonar_image
        self.force = force
        self.max_intensity = max_intensity
        self.pressed_keys = []
        self.config = holoocean.packagemanager.get_scenario(scenario)
        self.sonar_config = self.config['agents'][0]['sensors'][-1]["configuration"]
        self.env = holoocean.make(scenario)
        self.auto_control = auto_control
        self.azi = self.sonar_config['Azimuth']
        self.minR = self.sonar_config['RangeMin']
        self.maxR = self.sonar_config['RangeMax']
        self.binsR = self.sonar_config['RangeBins']
        self.binsA = self.sonar_config['AzimuthBins']
        self.pos_inicial = self.config['agents'][0]['location']
        self.rot_inicial = self.config['agents'][0]['rotation']
        self.theta = np.linspace(-self.azi/2, self.azi/2, self.binsA) * np.pi/180
        self.r = np.linspace(self.minR, self.maxR, self.binsR)
        self.T, self.R = np.meshgrid(self.theta, self.r)
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        self.pid_controller_linear = holoOceanUtils.PIDController(kp=0.5, ki=0.0, kd=0.01)
        self.pid_controller_angular = holoOceanUtils.PIDController(kp=0.01, ki=0.0, kd=0.01)
        self.dt = 1 / 200
        self.command = None
        self.reached_waypoints = 0
    
    def reachedWaypoint(self, waypoints)->bool:#verifica se alinhou
        self.waypoints = waypoints
        tresh_hold_distance=0.1
        tresh_hold_angle=1.0
        x_rov=self.position[0]
        y_rov=self.position[1]
        z_rov=self.position[2]
        roll_rov=self.rotation[0]
        pitch_rov=self.rotation[1]
        yaw_rov=self.rotation[2]
        
        if roll_rov>180:
            roll_rov-=360
        if pitch_rov>180:
            pitch-=360
        if yaw_rov>180:
            yaw_rov-=360

        x=self.actual_waypoint[0]
        y=self.actual_waypoint[1]
        z=self.actual_waypoint[2]
        roll=self.actual_waypoint[3]
        pitch=self.actual_waypoint[4]
        yaw=self.actual_waypoint[5]

        if roll>180:
            roll-=360
        if pitch>180:
            pitch-=360
        if yaw>180:
            yaw-=360

        if np.linalg.norm(np.array([x,y,z])-np.array([x_rov,y_rov,z_rov]))<=tresh_hold_distance and np.linalg.norm(roll-roll_rov)<=tresh_hold_angle and np.linalg.norm(pitch-pitch_rov)<=tresh_hold_angle and np.linalg.norm(yaw-yaw_rov)<=tresh_hold_angle:
            self.reached_waypoints+=1
            if self.reached_waypoints<len(self.waypoints):
                self.actual_waypoint=waypoints[self.reached_waypoints]
                return True
            else:
                return False
        return False
    
    def calculateVelocities(self)->None:#calcula o command para o rov baseado na nova posicao
        position_error = self.actual_waypoint - self.position
        
        desired_linear_velocity = self.pid_controller_linear.update(np.linalg.norm(position_error), self.dt)
        linear_velocity = position_error / np.linalg.norm(position_error) * desired_linear_velocity

        # erro_orientacao = self.actual_waypoint[3:] - self.rotation

        # erro_orientacao[erro_orientacao > 180] -= 360
        # erro_orientacao[erro_orientacao < -180] += 360

        # desired_angular_velocity = self.pid_controller_angular.update(np.linalg.norm(erro_orientacao), self.dt)
        # angular_velocity = erro_orientacao / np.linalg.norm(erro_orientacao) * desired_angular_velocity
        # angular_velocity=[0,0,angular_velocity[2]]
        self.command = np.concatenate((linear_velocity, [0,0,0]), axis=None)
        #print(self.command)

    def fineshedMission(self)->bool:#verifica se todos os waypoints foram visitados
        if self.reached_waypoints-1>len(self.waypoints):
            self.command=[0,0,0,0,0,0]
            plt.close('all')
            return True
        else:
            return False
        
    def parse_keys(self, val): #Comandos de teclado para o ROV
        command = np.zeros(8)
        if 'i' in self.pressed_keys:
            command[0:4] += val
        if 'k' in self.pressed_keys:
            command[0:4] -= val
        if 'j' in self.pressed_keys:
            command[[4,7]] += val
            command[[5,6]] -= val
        if 'l' in self.pressed_keys:
            command[[4,7]] -= val
            command[[5,6]] += val
        if 'w' in self.pressed_keys:
            command[4:8] += val
        if 's' in self.pressed_keys:
            command[4:8] -= val
        if 'a' in self.pressed_keys:
            command[[4,6]] += val
            command[[5,7]] -= val
        if 'd' in self.pressed_keys:
            command[[4,6]] -= val
            command[[5,7]] += val
        return command

    def on_press(self, key):
        if hasattr(key, 'char'):
            self.pressed_keys.append(key.char)
            self.pressed_keys = list(set(self.pressed_keys))

    def on_release(self, key):
        if hasattr(key, 'char'):
            self.pressed_keys.remove(key.char)

    def pose_sensor_update(self, state):  # Atualiza a rotação e posição do ROV
        pose = state['PoseSensor']
        self.rotation_matrix = pose[:3, :3]
        rotation = Rot.from_matrix(self.rotation_matrix)
        self.rotation = rotation.as_euler('xyz', degrees=True)
        self.position = pose[:3, 3]
    
    def visualize_image_sonar(self): #Cria uma imagem que se atualiza durante a simulação mostrando a visão do sonar
        plt.ion()
        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 5))
        self.ax.set_theta_zero_location("N")

        self.ax.set_thetamin(-self.azi / 2)
        self.ax.set_thetamax(self.azi / 2)

        theta = np.linspace(-self.azi / 2, self.azi / 2, self.binsA) * np.pi / 180
        r = np.linspace(self.minR, self.maxR, self.binsR)
        T, R = np.meshgrid(theta, r)
        z = np.zeros_like(T)
        
        plt.grid(False)
        self.plot = self.ax.pcolormesh(T, R, z, cmap='gray', shading='auto', vmin=0, vmax=1)
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def alinhar_clouds(self, source_cloud, target_cloud): #usa o teaser++ para alinhar as nuvens

            if type(source_cloud) == str:
                src_pcd = o3d.io.read_point_cloud(source_cloud)
            else:
                src_pcd = source_cloud

            if type(target_cloud) == str:
                dst_pcd = o3d.io.read_point_cloud(target_cloud)
            else:
                dst_pcd = target_cloud

            src_pcd.paint_uniform_color([1, 0.5, 0])
            dst_pcd.paint_uniform_color([1, 0.8, 0])

            src = np.asarray(src_pcd.points).T
            dst = np.asarray(dst_pcd.points).T

            solver_params = teaserpp_python.RobustRegistrationSolver.Params()
            solver_params.cbar2 = 1
            solver_params.noise_bound = 0.01
            solver_params.estimate_scaling =False
            solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
            solver_params.rotation_gnc_factor = 1.4
            solver_params.rotation_max_iterations = 100
            solver_params.rotation_cost_threshold = 1e-12
            solver = teaserpp_python.RobustRegistrationSolver(solver_params)
            solver.solve(src, dst)

            solution = solver.getSolution()
            #scale = solution.scale
            translation = solution.translation
            rotation = solution.rotation

            array = np.eye(4)
            array[:3, :3] = rotation
            array[:3, 3] = translation

            aligned_cloud = o3d.geometry.PointCloud()
            aligned_cloud.points = o3d.utility.Vector3dVector(np.dot(np.asarray(src_pcd.points), array[:3, :3].T) + array[:3, 3])

            #o3d.visualization.draw_geometries([aligned_cloud, dst_pcd])

            return aligned_cloud + dst_pcd
    
    def coord_translate(self, raio, azimuth, elev):
        x = raio * np.cos(azimuth) * np.sin(elev)
        y = raio * np.sin(azimuth) * np.sin(elev)
        z = raio * np.cos(elev)
        return x, y, z

    def create_pointcloud(self, all_coords, name=str, visualize_pcd=True): #Salva os pontos em uma point cloud
        # if self.sonar_image:
        #     visualize_pcd = False
        #     #Essa verificacao existe pois duas interfaces graficas ao mesmo tempo da ruim com o python
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_coords)
        if not os.path.exists(os.getcwd()+"/data"):
            os.makedirs(os.getcwd()+"/data/simulation_cloud")
        o3d.io.write_point_cloud("data/simulation_cloud/" + name + ".xyz", pcd)
        if visualize_pcd:
            o3d.visualization.draw_geometries([pcd])
        return pcd
   
    def update_sonar_image(self, state): #Atualiza a imagem do campo de visão do sonar a cada step
        s = state['ImagingSonar']
        self.plot.set_array(s.ravel())
        raio = np.sqrt((self.mean_point[0] - self.position[0])**2 + (self.mean_point[1] - self.position[1])**2)
        theta = np.arctan2(self.mean_point[1] - self.position[1], self.mean_point[0] - self.position[0])
        theta_deg = np.degrees(theta)
        if theta_deg < -self.azi / 2:
            theta_deg += 360  
        elif theta_deg > self.azi / 2:
            theta_deg -= 360
        self.a = self.ax.scatter(np.radians(theta_deg), raio, color='red', s=10, label='Mean Point')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # def calculateVelocities(self,point,rot)->None: #calcula o vetor de velocidades para que o hov chegue até algum determinado local
    #     position_error = point - self.position
    #     desired_linear_velocity = self.pid_controller_linear.update(np.linalg.norm(position_error), self.dt)
    #     linear_velocity = position_error / np.linalg.norm(position_error) * desired_linear_velocity
    #     erro_orientacao = rot - self.rotation
    #     erro_orientacao[erro_orientacao > 180] -= 360
    #     erro_orientacao[erro_orientacao < -180] += 360
    #     desired_angular_velocity = self.pid_controller_angular.update(np.linalg.norm(erro_orientacao), self.dt)
    #     angular_velocity = erro_orientacao / np.linalg.norm(erro_orientacao) * desired_angular_velocity
    #     angular_velocity=[0,0,angular_velocity[2]]
    #     self.command = np.concatenate(([0,0,0], angular_velocity), axis=None)
    #     if np.isnan(self.command[5]):
    #         self.env.reset()

    def get_coord_from_sonar(self, state): #traduz a matriz de intensidade do ROV para pontos no espaço 3D
        sonar_data = state['ImagingSonar']
        dist_vals, self.angul_vals = np.meshgrid(self.r, self.theta, indexing="ij")
        x, y, z = self.coord_translate(dist_vals, self.angul_vals, np.radians(90 - self.rotation[1]))
        mask_points = sonar_data > (np.max(sonar_data) * self.max_intensity)
        mask_distances = np.sqrt(x**2 + y**2 + z**2) < 30.0
        mask = mask_points & mask_distances

        coords = np.column_stack((x[mask] + self.position[0], 
                                 y[mask] + self.position[1], 
                                 z[mask] + self.position[2]))
        
        coords = np.dot(coords, self.rotation_matrix)
        return coords

    def align_to_dense_region(self, state): #verifica a região mais densa do campo de visão do Rov e alinha ele para essa região
        #************* = pendencias aqui
        sonar_data = state['ImagingSonar']
        ponto_A = sonar_data[0][self.binsA//2]
        ponto_A = self.coord_translate(self.maxR, self.binsA/2, np.radians(90 - self.rotation[1]))
        vetor_v = [ponto_A[0]-self.position[0], ponto_A[1]-self.position[1], ponto_A[2]-self.position[2]]
        coords = self.get_coord_from_sonar(state)
        self.mean_point = np.mean(coords, axis=0)
        vetor_u = [self.mean_point[0]-self.position[0], self.mean_point[1]-self.position[1], self.mean_point[2]-self.position[2]]
        cos_alfa = ((vetor_v[0]*vetor_u[0]) + (vetor_v[1]*vetor_u[1]) + (vetor_v[2]*vetor_u[2])) / (np.linalg.norm(vetor_u)*np.linalg.norm(vetor_v))
        angulo = np.degrees(np.arccos(cos_alfa))
        self.calculateVelocities(self.position,[0,0,self.rotation[2]-angulo])

    def run_simulation(self): #código que roda a simulação em si
        self.env.set_control_scheme("auv0", 2)
        all_coords = [] 
        if self.sonar_image:
            self.visualize_image_sonar()
        n = 0
        temps = [] 
        while True:
            if 'q' in self.pressed_keys:
                plt.close() 
                break
            state = self.env.tick()
            self.pose_sensor_update(state)
            if 'ImagingSonar' in state: 
                
                coords = self.get_coord_from_sonar(state)
                # pcd = self.create_pointcloud(np.asarray(coords), f'sub/cloud_{n}', visualize_pcd=False)
                # temps.append(pcd)
                
                # n += 1
                all_coords.append(coords)

                if self.auto_control:#-215 -26 -50
                    pass
                    #self.align_to_dense_region(state)
                else:
                    self.env.set_control_scheme("auv0", 0)
                    self.command = self.parse_keys(25) 
                self.reached_waypoints([[-215.0, 4.0, -50.0], [-185.0, -26.0, -50.0], [-215.0, -56.0, -50.0]])
                self.calculateVelocities()
                self.env.step(self.command)

                if self.sonar_image:
                    self.update_sonar_image(state)
                    self.a.remove()

        # for i in range(len(temps)):
        #     if i == 0:
        #         align = self.alinhar_clouds(temps[i], temps[i+1])
        #     elif i+1 < len(temps):
        #         align = self.alinhar_clouds(align, temps[i+1])
        #o3d.visualization.draw_geometries([align])

        #testes de alinhamento de apenas duas nuvens
        # a = self.alinhar_clouds('simulation_cloud/sub/cloud_0.xyz','simulation_cloud/sub/cloud_1.xyz')
        # o3d.visualization.draw_geometries([a])
        # b = self.alinhar_clouds(a,"simulation_cloud/sub/cloud_2.xyz")
        # o3d.visualization.draw_geometries([b])
        # a = self.alinhar_clouds(temps[0],temps[1])
        # b = self.alinhar_clouds(a,"simulation_cloud/sub/cloud_2.xyz")
        # o3d.visualization.draw_geometries([b])

        all_coords = np.vstack(all_coords)
        self.create_pointcloud(all_coords, 'cloud', visualize_pcd=True)

hov_simulator = ROVSimulator(auto_control=True,sonar_image=False)
hov_simulator.run_simulation()