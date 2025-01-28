import teaserpp_python
import open3d as o3d
import numpy as np


def alinhar_clouds( source_cloud, target_cloud): #usa o teaser++ para alinhar as nuvens

        if type(source_cloud) == str:
            src_pcd = o3d.io.read_point_cloud(source_cloud)
        else:
            src_pcd = source_cloud

        if type(target_cloud) == str:
            dst_pcd = o3d.io.read_point_cloud(target_cloud)
        else:
            dst_pcd = target_cloud

        src_pcd.paint_uniform_color([1, 0.5, 0])
        dst_pcd.paint_uniform_color([1, 0.8, 0])

        src = np.asarray(src_pcd.points).T
        dst = np.asarray(dst_pcd.points).T

        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1
        solver_params.noise_bound = 0.01
        solver_params.estimate_scaling =False
        solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 100
        solver_params.rotation_cost_threshold = 1e-12
        solver = teaserpp_python.RobustRegistrationSolver(solver_params)
        solver.solve(src, dst)

        solution = solver.getSolution()
        #scale = solution.scale
        translation = solution.translation
        rotation = solution.rotation

        array = np.eye(4)
        array[:3, :3] = rotation
        array[:3, 3] = translation

        aligned_cloud = o3d.geometry.PointCloud()
        aligned_cloud.points = o3d.utility.Vector3dVector(np.dot(np.asarray(src_pcd.points), array[:3, :3].T) + array[:3, 3])

        #o3d.visualization.draw_geometries([aligned_cloud, dst_pcd])

        return aligned_cloud + dst_pcd