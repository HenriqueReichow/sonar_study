import holoocean
import numpy as np
import open3d as o3d
from pynput import keyboard
from scipy.spatial.transform import Rotation as Rot

#OPP
#definir o control_scheme da maneira correta
#entender o pressed_keys

class HOVcontroll:
    def __init__(self,control_scheme,pressed_keys=list(),force=25):
        self.pressed_keys = pressed_keys
        self.force = force
        self.control_scheme = control_scheme
    
    def parse_keys(keys, val):
        command = np.zeros(8)
        if 'i' in keys:
            command[0:4] += val
        if 'k' in keys:
            command[0:4] -= val
        if 'j' in keys:
            command[[4,7]] += val
            command[[5,6]] -= val
        if 'l' in keys:
            command[[4,7]] -= val
            command[[5,6]] += val

        if 'w' in keys:
            command[4:8] += val
        if 's' in keys:
            command[4:8] -= val
        if 'a' in keys:
            command[[4,6]] += val
            command[[5,7]] -= val
        if 'd' in keys:
            command[[4,6]] -= val
            command[[5,7]] += val

        return command

    def on_press(key):
        global pressed_keys
        if hasattr(key, 'char'):
            pressed_keys.append(key.char)
            pressed_keys = list(set(pressed_keys))

    def on_release(key):
        global pressed_keys
        if hasattr(key, 'char'):
            pressed_keys.remove(key.char)
    
    def calculate_density(sonar_data):
        max = np.max(sonar_data)
        data_max = sonar_data >= max
        return data_max
    
    def calculate_new_position(coords,sonar_data):
        new = coords[HOVcontroll.calculate_density(sonar_data)]
        new[0] = new[0] - 25
        return new
    
    def automatic_command(scenario, coords, sonar_data, control_scheme=1):
        config = holoocean.packagemanager.get_scenario(scenario)
        config['agents'][0]['sensors']["control_scheme"] = control_scheme
        if control_scheme == 1:
            new = HOVcontroll.calculate_new_position(coords,sonar_data)
            return np.concatenate((new,rot),axis=0)

class Config:
    def __init__(self, scenario):
        self.scenario = scenario

    def traducao(raio, azimuth, elev):
        x = raio * np.cos(azimuth) * np.sin(elev)
        y = raio * np.sin(azimuth) * np.sin(elev)
        z = raio * np.cos(elev)
        return x,y,z

    def config_scenario(scenario):
        config = holoocean.packagemanager.get_scenario(scenario)
        sonar_config = config['agents'][0]['sensors'][-1]["configuration"]
        azi = sonar_config['Azimuth']
        minR = sonar_config['RangeMin']
        maxR = sonar_config['RangeMax']
        binsR = sonar_config['RangeBins']
        binsA = sonar_config['AzimuthBins']
        return azi,minR,maxR,binsR,binsA

class Cloud_use:
    def __init__(self,points,pointcloud,name):
        self.points = points
        self.pointcloud = pointcloud
        self.name = name
    def read_cloud():
        pass
    def add_points():
        pass
    def write_cloud(name=str):
        pass
    def visualize(pcd):
        pass
#main 

listener = keyboard.Listener(
            on_press=HOVcontroll.on_press,
            on_release=HOVcontroll.on_release)
listener.start()

pressed_keys = list()
force = 25

scenario = "Dam-HoveringImagingSonar"
azi,minR,maxR,binsR,binsA = Config.config_scenario(scenario)

theta = np.linspace(-azi/2, azi/2, binsA) * np.pi/180
r = np.linspace(minR, maxR, binsR)
T, R = np.meshgrid(theta, r)

with holoocean.make(scenario) as env:
    all_coords = []
    while True:
        if 'q' in pressed_keys:
            break
        command = HOVcontroll.parse_keys(pressed_keys, force)
        
        env.act("auv0", command)
        state = env.tick()
        
        pose = state['PoseSensor']

        rotation = pose[:3,:3]
        rot = Rot.from_matrix(rotation)
        rot = rot.as_euler('xyz', degrees=True)
        pos = pose[:3,3]

        if 'ImagingSonar' in state:
            sonar_data = state['ImagingSonar']
            
            dist_vals, angul_vals = np.meshgrid(r, theta, indexing="ij")
            x, y, z = Config.traducao(dist_vals, angul_vals, np.radians(90 - rot[1]))

            mask = sonar_data > np.max(sonar_data) * 0.8 
            max = sonar_data == np.max(sonar_data)

            coord_max = np.column_stack((x[max] + pos[0], 
                                      y[max] + pos[1], 
                                      z[max] + pos[2]))
            print(coord_max)
            coords = np.column_stack((x[mask] + pos[0], 
                                      y[mask] + pos[1], 
                                      z[mask] + pos[2]))
            
            coords = coords @ rotation 
            all_coords.append(coords)

    all_coords = np.vstack(all_coords)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_coords)
    
    o3d.io.write_point_cloud(f"simulation_cloud/cloud.xyz", pcd)
    pcd_load = o3d.io.read_point_cloud(f"simulation_cloud/cloud.xyz")
    o3d.visualization.draw_geometries([pcd])

    #detectar qual região do hov tem uma concentracao maior de points
    #e assim fazer ele se locomover até ela 
    #centralizar o ponto mais denso dos dados com o centro de visão do hov
    #env.act resultou algo
    #manter as rotações estaveis (está desgovernado)
    #melhorar o código, está muito confuso
