#import holoocean
import numpy as np
import json
import open3d as o3d

"""class Sonar_data_pcd:

    def __init__(self, data_sonar, data_json):
          self.data_sonar = data_sonar
          self.data_json = data_json
"""
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
    
    R = (np.dot(R_yaw, np.dot(R_pitch, R_roll))).T
    return R

for n in range(40):
    all_coords = []
    for k in range(1,500):
            try:
                dados = np.load(f"/home/lh/Documents/all_missions/mission-1/auv-{n}-data/{n}-raw-sonar-data-{k}.npy")

            except FileNotFoundError as error:
                break
            else:
                with open(f'/home/lh/Documents/all_missions/mission-1/auv-{n}-data/{n}-sonar_meta_data-{k}.json','r') as arquivo:
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

                distances = np.linspace(minR, maxR, len(dados))
                angulos = np.radians(np.linspace(-azi/2 + yaw, azi/2 + yaw, len(dados[:][0])))
                dist_vals,angul_vals = np.meshgrid(distances,angulos,indexing="ij")
                x,y,z = traducao(dist_vals, angul_vals, np.radians(90 - pitch))
                    
                mask_points = dados > np.max(dados) * 0.8 
                mask_distances = np.sqrt(x**2 + y**2 + z**2) < 3.5
                mask = mask_points & mask_distances

                coords = np.column_stack((x[mask] + sonar_pos_x, 
                                    y[mask] + sonar_pos_y,
                                    z[mask] + sonar_pos_z))
                all_coords.append(coords)

    all_coords = np.vstack(all_coords)

    cloud = all_coords @ rotation(roll, pitch, yaw)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    o3d.io.write_point_cloud(f"clouds/crop_clouds/mission-1/cloud_crop_sonar_mission-1_{n}.xyz", pcd)
    pcd_load = o3d.io.read_point_cloud(f"clouds/crop_clouds/mission-1/cloud_crop_sonar_mission-1_{n}.xyz")
    o3d.visualization.draw_geometries([pcd_load])