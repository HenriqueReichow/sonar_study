import numpy as np
import json
import open3d as o3d
import os
import sys

class DatasetToPointcloud:
    def __init__(self, visualize_pcd = True, mission_dataset_path = "/home/lh/Documents/all_missions/mission-1", distance_filter = True, intensity_filter=0.8):
        self.visualize_pcd = visualize_pcd
        self.mission_dataset_path = mission_dataset_path
        self.distance_filter = distance_filter
        self.intensity_filter = intensity_filter

    def coord_translate(self, raio, azimuth, elev): 
        x = raio * np.cos(azimuth) * np.sin(elev)
        y = raio * np.sin(azimuth) * np.sin(elev)
        z = raio * np.cos(elev)
        return x,y,z

    def rotation(self, roll, pitch, yaw):
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
    
    def check_mission(self):
        if "mission-1" in self.mission_dataset_path:
            return 1
        if "mission-2" in self.mission_dataset_path:
            return 
        if "mission-3" in self.mission_dataset_path:
            return 3 
        else:
            return 4
        
    def sonardata_to_pcd(self):
        for n in range(len(os.listdir(self.mission_dataset_path))):
            all_coords = []
            for k in range(1,len(os.listdir(os.path.join(self.mission_dataset_path,f"auv-{n}-data")))+1):
                    try:
                        dados = np.load(self.mission_dataset_path + f"/auv-{n}-data/{n}-raw-sonar-data-{k}.npy")

                    except FileNotFoundError as error:
                        break
                    else:
                        with open(self.mission_dataset_path + f'/auv-{n}-data/{n}-sonar_meta_data-{k}.json','r') as arquivo:
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
                        x,y,z = self.coord_translate(dist_vals, angul_vals, np.radians(90 - pitch))
                            
                        mask_points = dados > np.max(dados) * self.intensity_filter 

                        if self.distance_filter:
                            mask_distances = np.sqrt(x**2 + y**2 + z**2) < 3.5
                            mask = mask_points & mask_distances
                        else:
                            mask = mask_points

                        coords = np.column_stack((x[mask] + sonar_pos_x, 
                                            y[mask] + sonar_pos_y,
                                            z[mask] + sonar_pos_z))
                        all_coords.append(coords)

            all_coords = np.vstack(all_coords)
            cloud = all_coords @ self.rotation(roll, pitch, yaw)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud)

            #verificar se existem as pastas, caso não criar 
            caminho = os.path.join(os.getcwd(), "data")
            if not os.path.exists(caminho):
                os.makedirs(caminho)

            caminho = os.path.join(caminho, "clouds")
            if not os.path.exists(caminho):
                os.makedirs(caminho)

            if self.distance_filter:
                caminho_crop = os.path.join(caminho, "crop_clouds")
                if not os.path.exists(caminho_crop):
                    os.makedirs(caminho_crop)

                if not os.path.exists(os.path.join(caminho_crop, f"mission-{self.check_mission()}")):
                    os.makedirs(os.path.join(caminho_crop, f"mission-{self.check_mission()}"))

                o3d.io.write_point_cloud(f"data/clouds/crop_clouds/mission-{self.check_mission()}/cloud_crop_sonar_mission-{self.check_mission()}_{n}.xyz", pcd)
                pcd_load = o3d.io.read_point_cloud(f"data/clouds/crop_clouds/mission-{self.check_mission()}/cloud_crop_sonar_mission-{self.check_mission()}_{n}.xyz")

            else:
                if not os.path.exists(os.path.join(caminho, f"mission-{self.check_mission()}")):
                    os.makedirs(os.path.join(caminho, f"mission-{self.check_mission()}"))

                o3d.io.write_point_cloud(f"data/clouds/mission-{self.check_mission()}/nuvem_sonar_mission-{self.check_mission()}_{n}.xyz", pcd)
                pcd_load = o3d.io.read_point_cloud(f"data/clouds/mission-{self.check_mission()}/nuvem_sonar_mission-{self.check_mission()}_{n}.xyz")
            
            if self.visualize_pcd:
                o3d.visualization.draw_geometries([pcd_load])

a = DatasetToPointcloud(intensity_filter=0.8)
a.sonardata_to_pcd()