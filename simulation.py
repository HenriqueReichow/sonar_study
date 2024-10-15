import holoocean
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from pynput import keyboard
from scipy.spatial.transform import Rotation as Rot

def traducao(raio,azimuth,elev): 
    x = raio * np.cos(azimuth) * np.sin(elev)
    y = raio * np.sin(azimuth) * np.sin(elev)
    z = raio * np.cos(elev)
    return x,y,z

def rotation(roll, pitch, yaw):
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])

    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0], 
                        [-np.sin(pitch), 0, np.cos(pitch)]])

    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
    
    R = (R_yaw@(R_pitch@R_roll)).T
    return R

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

pressed_keys = list()
force = 25

#### GET SONAR CONFIG
scenario = "OpenWater-HoveringImagingSonar"
config = holoocean.packagemanager.get_scenario(scenario)
sonar_config = config['agents'][0]['sensors'][-1]["configuration"]
azi = sonar_config['Azimuth']
minR = sonar_config['RangeMin']
maxR = sonar_config['RangeMax']
binsR = sonar_config['RangeBins']
binsA = sonar_config['AzimuthBins']
elev = sonar_config['Elevation']

plt.ion()
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
ax.set_theta_zero_location("N")
ax.set_thetamin(-azi/2)
ax.set_thetamax(azi/2)

theta = np.linspace(-azi/2, azi/2, binsA)*np.pi/180
r = np.linspace(minR, maxR, binsR)
T, R = np.meshgrid(theta, r)
z = np.zeros_like(T)

plt.grid(False)
plot = ax.pcolormesh(T, R, z, cmap='gray', shading='auto', vmin=0, vmax=1)
plt.tight_layout()
fig.canvas.draw()
fig.canvas.flush_events()

command = [0,0,0,0,0,0,0,5]

with holoocean.make(scenario) as env:

    x_coord,y_coord,z_coord = [],[],[]

    while True:
        if 'q' in pressed_keys:
            break
        #command = parse_keys(pressed_keys, force)

        env.act("auv0", command)
        state = env.tick()
        location = config['agents'][0]['location']
        if 'ImagingSonar' in state:
            sonar_data = state['ImagingSonar']

            plot.set_array(sonar_data.ravel())
            fig.canvas.draw()
            fig.canvas.flush_events()
                
            dist_vals,angul_vals = np.meshgrid(r,theta,indexing="ij")
            x,y,z = traducao(dist_vals, angul_vals, np.radians(90 - elev))
            mask = sonar_data > np.max(sonar_data) * 0.8 

            x_coord.append(x[mask] + location[0])
            y_coord.append(y[mask] + location[1])
            z_coord.append(z[mask] + location[2])


    '''x_coord = np.concatenate(x_coord)
    y_coord = np.concatenate(y_coord)
    z_coord = np.concatenate(z_coord)'''

    cloud = np.column_stack((x_coord,y_coord,z_coord)) 
    #r = Rot.from_euler('xyz', location)
    #cloud = cloud @ r
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    o3d.io.write_point_cloud(f"clouds/crop_clouds/mission-1/cloud.xyz", pcd)
    pcd_load = o3d.io.read_point_cloud(f"clouds/crop_clouds/mission-1/cloud.xyz")
    o3d.visualization.draw_geometries([pcd])

    plt.ioff()
    plt.show()