import casadi as ca
import numpy as np
import math
import matplotlib.pyplot as plt

from Plotting import Plotting

class TripleOmniRobot:
    def __init__(self, pos=np.array([0,0,0]), vel=np.array([0,0,0])):
        self.pos = pos  # [x,y,theta]
        self.vel = vel  # [vx, vy, omega]

        # Store the path
        self.path = [np.append(self.pos, self.vel)]

        # The constants of the Omni robot
        self.d = 0.5        # Robot diagram [m]
        self.r = 0.06       # Wheel radius [m]

        # Forward kinematic
        self.f_np = lambda x_, u_: np.array([
            u_[0]*ca.cos(x_[2]) - u_[1]*ca.sin(x_[2]),
            u_[0]*ca.sin(x_[2]) + u_[1]*ca.cos(x_[2]),
            u_[2]
        ])

        # The constraint of Robot
        self.max_vx = 3.0; self.min_vx = -self.max_vx
        self.max_vy = 3.0; self.min_vy = -self.max_vy
        self.max_omega = math.pi/3; self.min_omega = -self.max_omega

    def correct_velocity(self, vx, vy, omega):
        vx = min(max(vx, self.min_vx), self.max_vx)
        vy = min(max(vy, self.min_vx), self.max_vy)
        omega = min(max(omega, self.min_omega), self.max_omega)
        # print(f1, f2, f3)
        return vx, vy, omega

    def update_configuration(self, vx, vy, omega, dt):
        vx, vy, omega= self.correct_velocity(vx, vy, omega)
        self.vel  = np.array([vx, vy, omega])

        dpos = self.f_np(self.pos, self.vel)
        self.pos = self.pos + dpos*dt

        # Add current configuration to paths
        self.path.append(np.append(self.pos, self.vel))

# omni = TripleOmniRobot()
# omni.update_configuration(3,0,1,0.5)
# omni.update_configuration(3,0,1,0.5)
# omni.update_configuration(3,0,1,0.5)

# print(np.array(omni.path))
# plot = Plotting("Haha")
# plot.plot_path(omni.path)
# plt.show()