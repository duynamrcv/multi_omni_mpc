import casadi as ca
import numpy as np
import math

# from TripleOmniRobot import TripleOmniRobot

class MPCController:
    def __init__(self, omni, T=0.02, N=30, Q=np.diag([30.0, 30.0, 30.0]),
                R=np.diag([1.0, 1.0, 1.0])):
        self.omni = omni    # Model
        self.T = T          # time step
        self.N = N          # horizon length

        self.Q = Q          # Weight matrix for states
        self.R = R          # Weight matrix for controls
    
        # The history states and controls
        self.next_states = np.ones((self.N+1, 3))*omni.pos
        self.u0 = np.zeros((self.N, 3))

        self.setup_controller()
    
    def setup_controller(self):
        self.opti = ca.Opti()

        # state variable: position and velocity
        self.opt_states = self.opti.variable(self.N+1, 3)
        x = self.opt_states[:,0]
        y = self.opt_states[:,1]
        theta = self.opt_states[:,2]

        # the velocity
        self.opt_controls = self.opti.variable(self.N, 3)
        vx = self.opt_controls[0]
        vy = self.opt_controls[1]
        omega = self.opt_controls[2]

        # create model
        f = lambda x_, u_: ca.vertcat(*[
            ca.cos(x_[2])*u_[0] - ca.sin(x_[2])*u_[1],  # dx
            ca.sin(x_[2])*u_[0] + ca.cos(x_[2])*u_[1],  # dy
            u_[2],                                      # dtheta
        ])

        # parameters, these parameters are the reference trajectories of the pose and inputs
        self.opt_u_ref = self.opti.parameter(self.N, 3)
        self.opt_x_ref = self.opti.parameter(self.N+1, 3)

        # initial condition
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x_ref[0, :])
        for i in range(self.N):
            x_next = self.opt_states[i, :] + f(self.opt_states[i, :], self.opt_controls[i, :]).T*self.T
            self.opti.subject_to(self.opt_states[i+1, :] == x_next)
        
        # parameters of another robot
        self.opt_o = self.opti.parameter(self.N+1, 3)

        # add constraints to obstacle
        for i in range(self.N+1):
            temp_constraints_ = (self.opt_states[i,0] - self.opt_o[i,0])**2 \
                                +(self.opt_states[i,1] - self.opt_o[i,1])**2 - self.omni.d**2
            self.opti.subject_to(self.opti.bounded(self.omni.d**2, temp_constraints_, math.inf))

        # cost function
        obj = 0
        for i in range(self.N):
            state_error_ = self.opt_states[i, :] - self.opt_x_ref[i+1, :]
            control_error_ = self.opt_controls[i, :] - self.opt_u_ref[i, :]
            obj = obj + ca.mtimes([state_error_, self.Q, state_error_.T]) \
                        + ca.mtimes([control_error_, self.R, control_error_.T])
        self.opti.minimize(obj)

        # boundary and control conditions
        self.opti.subject_to(self.opti.bounded(self.omni.min_vx, vx, self.omni.max_vx))
        self.opti.subject_to(self.opti.bounded(self.omni.min_vy, vy, self.omni.max_vy))
        self.opti.subject_to(self.opti.bounded(self.omni.min_omega, omega, self.omni.max_omega))

        opts_setting = {'ipopt.max_iter':2000,
                        'ipopt.print_level':0,
                        'print_time':0,
                        'ipopt.acceptable_tol':1e-8,
                        'ipopt.acceptable_obj_change_tol':1e-6}

        self.opti.solver('ipopt', opts_setting)
    
    def solve(self, next_trajectories, next_controls, sense_pos):
        ## set parameter, here only update initial state of x (x0)
        self.opti.set_value(self.opt_x_ref, next_trajectories)
        self.opti.set_value(self.opt_u_ref, next_controls)
        self.opti.set_value(self.opt_o, sense_pos)
        
        ## provide the initial guess of the optimization targets
        self.opti.set_initial(self.opt_states, self.next_states)
        self.opti.set_initial(self.opt_controls, self.u0)
        
        ## solve the problem
        sol = self.opti.solve()
        
        ## obtain the control input
        self.u0 = sol.value(self.opt_controls)
        self.next_states = sol.value(self.opt_states)
        return self.u0

# omni = TripleOmniRobot()
# mpc = MPCController(omni)