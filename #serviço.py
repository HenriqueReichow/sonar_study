import numpy as np
import json
import open3d as o3d

def traducao(raio, azimuth, elev):
    x = raio * np.cos(azimuth) * np.sin(np.radians(elev))
    y = raio * np.sin(azimuth) * np.sin(np.radians(elev))
    z = raio * np.cos(np.radians(elev))
    return x, y, z

x_coord, y_coord, z_coord = [], [], []

for k in range(1, 144):
    dados = np.load(f"C:\\Users\\reich\\OneDrive\\Documentos\\auv-0-data\\0-raw-sonar-data-{k}.npy")
    with open(f'C:\\Users\\reich\\OneDrive\\Documentos\\auv-0-data\\0-sonar_meta_data-{k}.json', 'r') as arquivo:
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

    distances = np.linspace(minR, maxR, binsR)
    angulos = np.radians(np.linspace(-azi/2 + yaw, azi/2 + yaw, binsA))

    mask = dados > np.max(dados) * 0.8

    raios_vals, angulos_vals = np.meshgrid(distances, angulos, indexing='ij')
    x, y, z = traducao(raios_vals, angulos_vals, 90 - pitch)

    x_coord.append((x[mask] + sonar_pos_x).flatten())
    y_coord.append((y[mask] + sonar_pos_y).flatten())
    z_coord.append((z[mask] + sonar_pos_z).flatten())

x_coord = np.concatenate(x_coord)
y_coord = np.concatenate(y_coord)
z_coord = np.concatenate(z_coord)


cloud = np.column_stack((x_coord, y_coord, z_coord))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cloud)

o3d.io.write_point_cloud("nuvem_sonar_mission_1.xyz", pcd)

pcd_load = o3d.io.read_point_cloud("nuvem_sonar_mission_1.xyz")
o3d.visualization.draw_geometries([pcd_load])
