import numpy as np
import math
import time
import matplotlib.pyplot as plt
from Plotting import Plotting

from TripleOmniRobot import TripleOmniRobot
from MPCController import MPCController
    
def desired_trajectory(omni, t, T, N_):
    # initial state / last state
    x_ = np.zeros((N_+1, 3))
    x_[0] = omni.pos
    u_ = np.zeros((N_, 3))

    for i in range(N_):
        t_predict = t + T*i
        x_ref_ = 4*math.cos(2*math.pi/12*t_predict)
        y_ref_ = 4*math.sin(2*math.pi/12*t_predict)
        theta_ref_ = 2*math.pi/12*t_predict
        
        dotx_ref_ = -2*math.pi/12*y_ref_
        doty_ref_ =  2*math.pi/12*x_ref_
        dotq_ref_ =  2*math.pi/12

        vx_ref_ = dotx_ref_*math.cos(dotq_ref_) + doty_ref_*math.sin(dotq_ref_)
        vy_ref_ = -dotx_ref_*math.sin(dotq_ref_) + doty_ref_*math.cos(dotq_ref_)
        omega_ref_ = dotq_ref_

        x_[i+1] = np.array([x_ref_, y_ref_, theta_ref_])
        u_[i] = np.array([vx_ref_, vy_ref_, omega_ref_])

    return x_, u_

def desired_trajectory1(omni, t, T, N_):
    # initial state / last state
    x_ = np.zeros((N_+1, 3))
    x_[0] = omni.pos
    u_ = np.zeros((N_, 3))

    for i in range(N_):
        t_predict = t + T*i
        x_ref_ = 4*math.sin(2*math.pi/12*t_predict)
        y_ref_ = 4*math.cos(2*math.pi/12*t_predict)
        theta_ref_ = 2*math.pi/12*t_predict
        
        dotx_ref_ =  2*math.pi/12*4*math.cos(2*math.pi/12*t_predict)
        doty_ref_ = -2*math.pi/12*4*math.sin(2*math.pi/12*t_predict)
        dotq_ref_ =  2*math.pi/12

        vx_ref_ = dotx_ref_*math.cos(dotq_ref_) + doty_ref_*math.sin(dotq_ref_)
        vy_ref_ = -dotx_ref_*math.sin(dotq_ref_) + doty_ref_*math.cos(dotq_ref_)
        omega_ref_ = dotq_ref_

        x_[i+1] = np.array([x_ref_, y_ref_, theta_ref_])
        u_[i] = np.array([vx_ref_, vy_ref_, omega_ref_])

    return x_, u_

if __name__ == "__main__":
    T = 0.02
    N = 30
    sim_time = 12

    omni1 = TripleOmniRobot(pos=[0,4,0])
    mpc1 = MPCController(omni1, T, N)

    omni2 = TripleOmniRobot(pos=[4,0,0])
    mpc2 = MPCController(omni2, T, N)

    iner = 0
    ref1 = []
    ref2 = []
    while iner - sim_time/T < 0.0:
        t = iner*T
        next_trajectories1, next_controls1 = desired_trajectory1(omni1, t, T, N)
        ref1.append(next_trajectories1[1,:])

        next_trajectories2, next_controls2 = desired_trajectory(omni2, t, T, N)
        ref2.append(next_trajectories2[1,:])

        vel1 = mpc1.solve(next_trajectories1, next_controls1, mpc2.next_states)
        omni1.update_configuration(vel1[0,0], vel1[0,1], vel1[0,2], T)

        vel2 = mpc2.solve(next_trajectories2, next_controls2, mpc1.next_states)
        omni2.update_configuration(vel2[0,0], vel2[0,1], vel2[0,2], T)

        iner += 1

    plot = Plotting("MPC")
    # plot.plot_path(ref1, label="reference 1")
    # plot.plot_path(omni1.path, label="omni 1")
    # plot.plot_path(ref2, label="reference 2")
    # plot.plot_path(omni2.path, label="omni 2")
    plot.plot_animation(omni1.path, omni2.path, ref1, ref2)
    plt.show()