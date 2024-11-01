import holoocean
import holoocean.sensors
import numpy as np
import open3d as o3d
from pynput import keyboard
from scipy.spatial.transform import Rotation as Rot

#usar o posesensor da maneira certa
#transformar tudo em uma class
#refinar o código
#definir o control_scheme da maneira correta

def traducao(raio, azimuth, elev):
    x = raio * np.cos(azimuth) * np.sin(elev)
    y = raio * np.sin(azimuth) * np.sin(elev)
    z = raio * np.cos(elev)
    return x,y,z

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

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

pressed_keys = list()
force = 25

scenario = "Dam-HoveringImagingSonar"
config = holoocean.packagemanager.get_scenario(scenario)
sonar_config = config['agents'][0]['sensors'][-1]["configuration"]
azi = sonar_config['Azimuth']
minR = sonar_config['RangeMin']
maxR = sonar_config['RangeMax']
binsR = sonar_config['RangeBins']
binsA = sonar_config['AzimuthBins']

theta = np.linspace(-azi/2, azi/2, binsA) * np.pi/180
r = np.linspace(minR, maxR, binsR)
T, R = np.meshgrid(theta, r)

with holoocean.make(scenario) as env:

    all_coords = []
    while True:
        if 'q' in pressed_keys:
            break
        command = parse_keys(pressed_keys, force)
        
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
            x, y, z = traducao(dist_vals, angul_vals, np.radians(90 - rot[1]))

            mask = sonar_data > np.max(sonar_data) * 0.8 

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