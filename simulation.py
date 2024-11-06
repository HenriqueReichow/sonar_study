import holoocean
import numpy as np
import open3d as o3d
from pynput import keyboard
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt

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

    def on_press(self, key):
        if hasattr(key, 'char'):
            self.pressed_keys.append(key.char)
            self.pressed_keys = list(set(self.pressed_keys))

    def on_release(self, key):
        if hasattr(key, 'char'):
            self.pressed_keys.remove(key.char)

    def sonar_data_(self, state):
        sonar_data = state['ImagingSonar']
        dist_vals, angul_vals = np.meshgrid(self.r, self.theta, indexing="ij")
        x, y, z = self.coord_translate(dist_vals, angul_vals, np.radians(90 - self.rotation[1]))
        mask = sonar_data > (np.max(sonar_data) * self.max_intensity)

        coords = np.column_stack((x[mask] + self.position[0], 
                                  y[mask] + self.position[1], 
                                  z[mask] + self.position[2]))

        coords = coords @ self.rotation_matrix
        return coords

    def pose_sensor_update(self, state):
        pose = state['PoseSensor']
        self.rotation_matrix = pose[:3, :3]
        rotation = Rot.from_matrix(self.rotation_matrix)
        self.rotation = rotation.as_euler('xyz', degrees=True)
        self.position = pose[:3, 3]

    def visualize_image_sonar(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8,5))
        self.ax.set_theta_zero_location("N")
        self.ax.set_thetamin(-self.azi/2)
        self.ax.set_thetamax(self.azi/2)

        theta = np.linspace(-self.azi/2, self.azi/2, self.binsA)*np.pi/180
        r = np.linspace(self.minR, self.maxR, self.binsR)
        T, R = np.meshgrid(theta, r)
        z = np.zeros_like(T)

        plt.grid(False)
        self.plot = self.ax.pcolormesh(T, R, z, cmap='gray', shading='auto', vmin=0, vmax=1)
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_sonar_image(self,state):
        s = state['ImagingSonar']
        self.plot.set_array(s.ravel())
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def create_pointcloud(self, all_coords, name=str, visualize_pcd=True):
        if self.sonar_image:
            visualize_pcd = False
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_coords)
        o3d.io.write_point_cloud("simulation_cloud/"+name+".xyz", pcd)
        if visualize_pcd:
            o3d.visualization.draw_geometries([pcd])

    def automatic_control(self, state):
        x = []
        np.linspace(0,1,100)
    
    def run_simulation(self):
        if self.sonar_image:
            self.visualize_image_sonar()
        all_coords = []

        while True:
            if 'q' in self.pressed_keys:
                plt.close()
                break
            command = self.parse_keys(self.force)
            self.env.act("auv0", command)
            state = self.env.tick()

            self.pose_sensor_update(state)

            if 'ImagingSonar' in state:
                coords = self.sonar_data_(state)
                all_coords.append(coords)
                if self.sonar_image:
                    self.update_sonar_image(state)

        all_coords = np.vstack(all_coords)
        self.create_pointcloud(all_coords,'cloud',visualize_pcd=True)

hov_simulator = HOVSimulator(auto_control=False)
hov_simulator.run_simulation()

#detectar qual região do hov tem uma concentracao maior de points
#e assim fazer ele se locomover até ela 
#centralizar o ponto mais denso dos dados com o centro de visão do hov
#manter as rotações estaveis (está desgovernado)
#python robotics
#curva de bezier