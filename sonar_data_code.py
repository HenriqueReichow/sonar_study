#import holoocean
import numpy as np
import json
import open3d as o3d

def traducao(raio,azimuth,elev): 
    x = raio * np.cos(azimuth) * np.sin(elev)
    y = raio * np.sin(azimuth) * np.sin(elev)
    z = raio * np.cos(elev)

    y_roll = y * np.cos(roll) - z * np.sin(roll)
    z_roll = y * np.sin(roll) + z * np.cos(roll)
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
    
    R = (np.dot(R_yaw, np.dot(R_pitch, R_roll))).T

    return R

for n in range(40):
    x_coord,y_coord,z_coord = [],[],[]
    for k in range(1,200):
            try:
                dados = np.load(f"/home/lh/Documents/mission-1/auv-{n}-data/{n}-raw-sonar-data-{k}.npy")

            except FileNotFoundError as error:
                  break
            else:
                with open(f'/home/lh/Documents/mission-1/auv-{n}-data/{n}-sonar_meta_data-{k}.json','r') as arquivo:
                        dados_json = json.load(arquivo)

                azi = dados_json["sonar_azimuth"]
                minR = dados_json["sonar_range_min"]
                maxR = dados_json["sonar_range_max"]
                sonar_pos_x = dados_json["x"]
                sonar_pos_y = dados_json["y"]
                sonar_pos_z = dados_json["z"]
                pitch = dados_json["pitch"]
                yaw = dados_json["yaw"]
                roll = dados_json["roll"]

                #tentando transformar em pointcloud
                distances = np.linspace(minR, maxR, len(dados))
                angulos = np.radians(np.linspace(-azi/2 + yaw, azi/2 + yaw, len(dados[:][0])))
                dist_vals,angul_vals = np.meshgrid(distances,angulos,indexing="ij")
                x,y,z = traducao(dist_vals, angul_vals, np.radians(90 - pitch))

                mask_points = dados > np.max(dados) * 0.8 
                mask_distances = np.sqrt(x**2 + y**2 + z**2) < 3.5
                mask = mask_points & mask_distances

                x_coord.append((x[mask] + sonar_pos_x).flatten())
                y_coord.append((y[mask] + sonar_pos_y).flatten())
                z_coord.append((z[mask] + sonar_pos_z).flatten())

    x_coord = np.concatenate(x_coord)
    y_coord = np.concatenate(y_coord)
    z_coord = np.concatenate(z_coord)   

    cloud = np.column_stack((x_coord,y_coord,z_coord)) 

    cloud = np.dot(cloud, rotation(roll, pitch, yaw))
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    o3d.io.write_point_cloud(f"clouds/crop_clouds/mission-1/cloud_crop_sonar_mission-1_{n}.xyz", pcd)
    pcd_load = o3d.io.read_point_cloud(f"clouds/crop_clouds/mission-1/cloud_crop_sonar_mission-1_{n}.xyz")
    o3d.visualization.draw_geometries([pcd_load])