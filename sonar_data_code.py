#import holoocean
import numpy as np
import json
import open3d as o3d

def traducao(raio,azimuth,elev,roll): 
    x = raio * np.cos(azimuth) * np.sin(elev)
    y = raio * np.sin(azimuth) * np.sin(elev)
    z = raio * np.cos(elev)

    #y_roll = y * np.cos(roll) - z * np.sin(roll)
    #z_roll = y * np.sin(roll) + z * np.cos(roll)
    return x,y,z

for n in range(40):
    x_coord,y_coord,z_coord = [],[],[]
    for k in range(1,200):
            try:
                dados = np.load(f"/home/lh/Documents/mission-3/auv-{n}-data/{n}-raw-sonar-data-{k}.npy")

            except FileNotFoundError as error:
                  break
            else:
                with open(f'/home/lh/Documents/mission-3/auv-{n}-data/{n}-sonar_meta_data-{k}.json','r') as arquivo:
                        dados_json = json.load(arquivo)

                azi = dados_json["sonar_azimuth"]
                minR = dados_json["sonar_range_min"]
                maxR = dados_json["sonar_range_max"]
                binsR = dados_json["range_bins"]
                binsA = dados_json["azimuth_bins"]
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
                x,y,z = traducao(dist_vals, angul_vals, np.radians(90 - pitch), np.radians(roll))
                mask_points = dados > np.max(dados) * 0.8 
                print("nro linhas ",len(dados[:][0]))
                print("nro col ",len(dados))
                print(dados.shape)
                

                x_coord.append((x[mask_points] + sonar_pos_x).flatten())
                y_coord.append((y[mask_points] + sonar_pos_y).flatten())
                z_coord.append((z[mask_points] + sonar_pos_z).flatten())

    x_coord = np.concatenate(x_coord)
    y_coord = np.concatenate(y_coord)
    z_coord = np.concatenate(z_coord)   

    cloud = np.column_stack((x_coord,y_coord,z_coord)) 
    cloud = np.asarray(cloud)
    cloud = cloud@(np.array([[ 1, 0, 0],[ 0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])).T
    

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    o3d.io.write_point_cloud(f"/home/lh/Documents/clouds/mission-1/nuvem_sonar_mission-1_{n}.xyz", pcd)
    pcd_load = o3d.io.read_point_cloud(f"/home/lh/Documents/clouds/mission-1/nuvem_sonar_mission-1_{n}.xyz")
    o3d.visualization.draw_geometries([pcd_load])