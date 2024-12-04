import holoocean
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from pynput import keyboard
import sys
import math
sys.path.append('/home/lh/Desktop/HoloOceanUtils')
import holoOceanUtils
from scipy.spatial.transform import Rotation as Rot

class HOVSimulator:
    def __init__(self, scenario="Dam-HoveringImagingSonar", force=25, max_intensity=0.8, auto_control=True, sonar_image=False):
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

    def coord_translate(self, raio, azimuth, elev):
        x = raio * np.cos(azimuth) * np.sin(elev)
        y = raio * np.sin(azimuth) * np.sin(elev)
        z = raio * np.cos(elev)
        return x, y, z

    def parse_keys(self, val):
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
    
    def pose_sensor_update(self, state):
        pose = state['PoseSensor']
        self.rotation_matrix = pose[:3, :3]
        rotation = Rot.from_matrix(self.rotation_matrix)
        self.rotation = rotation.as_euler('xyz', degrees=True)
        self.position = pose[:3, 3]
        
    def create_pointcloud(self, all_coords, name=str, visualize_pcd=True):
        if self.sonar_image:
            visualize_pcd = False
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_coords)
        o3d.io.write_point_cloud("simulation_cloud/" + name + ".xyz", pcd)
        if visualize_pcd:
            o3d.visualization.draw_geometries([pcd])

    def visualize_image_sonar(self):
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

    def update_sonar_image(self, state):
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

    def on_press(self, key):
        if hasattr(key, 'char'):
            self.pressed_keys.append(key.char)
            self.pressed_keys = list(set(self.pressed_keys))

    def on_release(self, key):
        if hasattr(key, 'char'):
            self.pressed_keys.remove(key.char)

    def calculateVelocities(self,point,rot)->None:
        position_error = point - self.position
        desired_linear_velocity = self.pid_controller_linear.update(np.linalg.norm(position_error), self.dt)
        linear_velocity = position_error / np.linalg.norm(position_error) * desired_linear_velocity
        erro_orientacao = rot - self.rotation
        erro_orientacao[erro_orientacao > 180] -= 360
        erro_orientacao[erro_orientacao < -180] += 360
        desired_angular_velocity = self.pid_controller_angular.update(np.linalg.norm(erro_orientacao), self.dt)
        angular_velocity = erro_orientacao / np.linalg.norm(erro_orientacao) * desired_angular_velocity
        angular_velocity=[0,0,angular_velocity[2]]
        self.command = np.concatenate(([0,0,0], angular_velocity), axis=None)
        if np.isnan(self.command[5]):
            self.env.reset()
        print(self.command)

    def get_coord_from_sonar(self, state):
        sonar_data = state['ImagingSonar']
        dist_vals, self.angul_vals = np.meshgrid(self.r, self.theta, indexing="ij")
        x, y, z = self.coord_translate(dist_vals, self.angul_vals, np.radians(90 - self.rotation[1]))
        mask_points = sonar_data > (np.max(sonar_data) * self.max_intensity)
        mask_distances = np.sqrt(x**2 + y**2 + z**2) < 30.0
        mask = mask_points & mask_distances
        coords = np.column_stack((x[mask] + self.position[0], 
                                 y[mask] + self.position[1], 
                                 z[mask] + self.position[2]))
        return coords
    
    def align_to_dense_region(self, state):
        #************* = pendencias aqui
        sonar_data = state['ImagingSonar']
        ponto_A = sonar_data[0][self.binsA//2]
        ponto_A = self.coord_translate(self.maxR, self.binsA/2, np.radians(90 - self.rotation[1]))
        vetor_v = [ponto_A[0]-self.position[0], ponto_A[1]-self.position[1], ponto_A[2]-self.position[2]]
        coords = self.get_coord_from_sonar(state)
        self.mean_point = np.mean(coords, axis=0)
        vetor_u = [self.mean_point[0]-self.position[0], self.mean_point[1]-self.position[1], self.mean_point[2]-self.position[2]]
        cos_alfa = ((vetor_v[0]*vetor_u[0]) + (vetor_v[1]*vetor_u[1]) + (vetor_v[2]*vetor_u[2])) / (np.linalg.norm(vetor_u)*np.linalg.norm(vetor_v))
        angulo = math.degrees(math.acos(cos_alfa))
        self.calculateVelocities(self.position,[0,0,self.rotation[2]-angulo])

    def run_simulation(self):
        all_coords = []
        if self.sonar_image:
            self.visualize_image_sonar()
        while True:
            if 'q' in self.pressed_keys:
                plt.close()
                break
            state = self.env.tick()
            self.pose_sensor_update(state)
            if 'ImagingSonar' in state: 
                coords = self.get_coord_from_sonar(state)
                all_coords.append(coords)
                self.align_to_dense_region(state)
                self.env.step(self.command*0.0001)
                if self.sonar_image:
                    self.update_sonar_image(state)
                    self.a.remove()
                
        all_coords = np.vstack(all_coords)
        self.create_pointcloud(all_coords, 'cloud', visualize_pcd=True)

hov_simulator = HOVSimulator(auto_control=True,sonar_image=True)
hov_simulator.run_simulation()
#rotacoes estão desativas = pcd desalinhada
#falta refinar quando a rotação ta certa
""" -- Reunião sensores - 02/12/24
parte de alinhar o hov ao local mais denso = done
refinar oq fazer depois de alinhado
ir pra parte de orientar ele autonomamente
fazer curvas e trajetorias predefinidas
2
contribuição com sensores e autonomia
separação dos códigos para melhor modularização
posteriormente ajudar a criar novos módulos para o utils do holoocean
"""