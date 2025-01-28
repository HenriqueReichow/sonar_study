import numpy as np
from Bezier import Bezier
import holoocean
import holoocean.environments
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from pynput import keyboard
import sys
sys.path.append('/home/lh/Desktop/HoloOceanUtils')
import holoOceanUtils
import teaserpp_python
from scipy.spatial.transform import Rotation as Rot
import os

# t_points  = np.arange(0, 1, 0.01)
# points1 = np.array([[-245.0, -26.0, -50.0], [-215.0, 4.0, -50.0], [-185.0, -26.0, -50.0]])
# points2 = np.array([[-185.0, -26.0, -50.0],[-215.0, -56.0, -50.0],[-245.0, -26.0, -50.0]])
# curve2 = Bezier.Curve(t_points, points2)
# curve1 = Bezier.Curve(t_points, points1)

# plt.figure()
# plt.plot(
# 	curve1[:, 0],   # x-coordinates.
# 	curve1[:, 1]    # y-coordinates.
# )
# plt.plot(
# 	points1[:, 0],  # x-coordinates.
# 	points1[:, 1],  # y-coordinates.
# 	'ro:'           # Styling (red, circles, dotted).
# )
# plt.plot(
#     curve2[:, 0],   # x-coordinates.
# 	curve2[:, 1]    # y-coordinates.
#     )
# plt.plot(
# 	points2[:, 0],  # x-coordinates.
# 	points2[:, 1],  # y-coordinates.
# 	'ro:'           # Styling (red, circles, dotted).
# )
# plt.grid()
# plt.show()

class teste:
    def __init__(self,waypoints):
        self.number_of_waypoints = len(waypoints)
        self.waypoints = waypoints
        self.reached_waypoints: int=0
        self.actual_waypoint=waypoints[self.reached_waypoints]

        self.pid_controller_linear = holoOceanUtils.PIDController(kp=0.5,ki=0.0,kd=0.01)
        #self.pid_controller_linear = PIDController(kp=0.25,ki=0.025,kd=0.0)
        #self.pid_controller_angular = PIDController(kp=0.125,ki=0.0125,kd=0.0)
        self.pid_controller_angular = holoOceanUtils.PIDController(kp=0.01,ki=0.0,kd=0.01)
        self.dt=1/200
     
    def reachedWaypoint(self)->bool:
        tresh_hold_distance=0.1
        tresh_hold_angle=1.0
        x_rov=self.actual_location[0]
        y_rov=self.actual_location[1]
        z_rov=self.actual_location[2]
        roll_rov=self.actual_rotation[0]
        pitch_rov=self.actual_rotation[1]
        yaw_rov=self.actual_rotation[2]
        
        if roll_rov>180:
            roll_rov-=360
        if pitch_rov>180:
            pitch-=360
        if yaw_rov>180:
            yaw_rov-=360

        x=self.actual_waypoint[0]
        y=self.actual_waypoint[1]
        z=self.actual_waypoint[2]
        roll=self.actual_waypoint[3]
        pitch=self.actual_waypoint[4]
        yaw=self.actual_waypoint[5]

        if roll>180:
            roll-=360
        if pitch>180:
            pitch-=360
        if yaw>180:
            yaw-=360

        if np.linalg.norm(np.array([x,y,z])-np.array([x_rov,y_rov,z_rov]))<=tresh_hold_distance and np.linalg.norm(roll-roll_rov)<=tresh_hold_angle and np.linalg.norm(pitch-pitch_rov)<=tresh_hold_angle and np.linalg.norm(yaw-yaw_rov)<=tresh_hold_angle:
            print('aqui')
            self.reached_waypoints+=1
            if self.reached_waypoints<self.number_of_waypoints:
                self.actual_waypoint=self.waypoints[self.reached_waypoints]
                return True
            else:
                return False
        return False
    
    def calculateVelocities(self)->None:
        position_error = self.actual_waypoint[:3] - self.actual_location
        
        desired_linear_velocity = self.pid_controller_linear.update(np.linalg.norm(position_error), self.dt)
        linear_velocity = position_error / np.linalg.norm(position_error) * desired_linear_velocity

        erro_orientacao = self.actual_waypoint[3:] - self.actual_rotation

        erro_orientacao[erro_orientacao > 180] -= 360
        erro_orientacao[erro_orientacao < -180] += 360

        desired_angular_velocity = self.pid_controller_angular.update(np.linalg.norm(erro_orientacao), self.dt)
        angular_velocity = erro_orientacao / np.linalg.norm(erro_orientacao) * desired_angular_velocity
        angular_velocity=[0,0,angular_velocity[2]]
        self.command = np.concatenate((linear_velocity, angular_velocity), axis=None)
        #print(self.command)

    def fineshedMission(self)->bool:
        if self.reached_waypoints-1>self.number_of_waypoints:
            self.command=[0,0,0,0,0,0]
            plt.close('all')
            return True
        else:
            return False

    def pose_sensor_update(self, state):  # Atualiza a rotação e posição do ROV
        pose = state['PoseSensor']
        self.rotation_matrix = pose[:3, :3]
        rotation = Rot.from_matrix(self.rotation_matrix)
        self.actual_rotation = rotation.as_euler('xyz', degrees=True)
        self.actual_location = pose[:3, 3]
    
    def run_simulation(self):
        env = holoocean.make('Dam-HoveringImagingSonar')
        env.draw_point([-215.0, 4.0, -50.0],lifetime=0,color=[0,0,255])
        env.draw_point([-185.0, -26.0, -50.0],lifetime=0,color=[0,255,0])
        env.draw_point([-215.0, -56.0, -50.0],lifetime=0)
        env.draw_point([-215, -26 ,-50],lifetime=0)#centro
        state = env.tick()
        while True:
            self.pose_sensor_update(state)
            self.calculateVelocities()
            self.reachedWaypoint()
            self.fineshedMission()
            state=env.step(self.command*0.005)

waypoints = [[-215.0, 4.0, -50.0, 0, 0, 0], [-185.0, -26.0, -50.0, 0, 0, 0], [-215.0, -56.0, -50.0, 0, 0, 0]]

a = teste(waypoints)
a.run_simulation()