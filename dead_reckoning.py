from math import sin,cos
import numpy as np
import math

class dead_reckoning:

    def __init__(self,df_mag):
        self.df_mag = df_mag

    '''
    vel:  list of velocity, with length 3,[v_x,v_y,v_z] observed at time t_n from IMU
    pos:  list of coordinates,with length 3, [p_x,p_y,p_z]  at time t_n-1
    acc:  list of acceleration, with length 3,[a_x,a_y,a_z]  observed at time t_n from IMU
    time_period : time interval between t_n-1 and t_n
    '''

    def predict(self,vel,pos,acc,eular,time_period):
        v_x,v_y,v_z = vel[0],vel[1],vel[2]
        p_x,p_y,p_z = pos[0],pos[1],pos[2]
        alpha,beta,gama = math.radians(eular[0]),math.radians(eular[1]),math.radians(eular[2])
        a_x = acc[0]*0.01
        a_y = acc[1]*0.01
        a_z = acc[2]*0.01

        trans = np.array([
            [1, 0, 0, time_period, 0, 0],
            [0, 1, 0, 0, time_period, 0],
            [0, 0, 1, 0, 0, time_period],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]])

        D = np.array([
            [0.5 * time_period * time_period, 0, 0],
            [0, 0.5 * time_period * time_period, 0],
            [0, 0, 0.5 * time_period * time_period],
            [time_period, 0, 0],
            [0, time_period, 0],
            [0, 0, time_period]])

        Rot = np.array([
            [cos(alpha) * cos(gama), sin(gama) * cos(alpha) * sin(alpha) - sin(alpha) * cos(beta),
             cos(alpha) * sin(gama) * cos(beta) + sin(alpha) * sin(beta)],
            [sin(alpha) * cos(gama), sin(alpha) * sin(beta) * sin(gama) + cos(alpha) * cos(beta),
             sin(alpha) * cos(beta) * cos(gama) - cos(alpha) * sin(beta)],
            [-sin(gama), cos(gama) * sin(beta), cos(gama) * cos(beta)]])

        U = [a_x, a_y, a_z]

        x_old = np.array([p_x, p_y, p_z, v_x, v_y, v_z])
        x_new = np.dot(trans, x_old) + np.dot(np.dot(D, Rot), U)
        return x_new



