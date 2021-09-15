# Pratik Chaudhari (pratikac@seas.upenn.edu)

import os, sys, pickle, math
from copy import deepcopy
from scipy.spatial.transform import Rotation

from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import time

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20, 20
        s.ymin, s.ymax = -20, 20
        s.szx = int(np.ceil((s.xmax-s.xmin)/s.resolution+1))
        s.szy = int(np.ceil((s.ymax-s.ymin)/s.resolution+1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh/(1-s.occupied_prob_thresh))

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        #### TODO: XXXXXXXXXXX
        # raise NotImplementedError
        i_row = np.floor((x - s.xmin) / s.resolution)
        i_row = np.clip(i_row, 0, 2 * s.xmax / s.resolution)
        j_col = np.floor((y - s.ymin) / s.resolution)
        j_col = np.clip(j_col, 0, 2 * s.ymax / s.resolution)
        toReturn = np.vstack((i_row, j_col)).astype(int)    # occu_coor should be 2X1081 shape
        return toReturn


class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.05, Q=1e-3*np.eye(3),
                 resampling_threshold=0.3):
        s.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        # s.Q = 1e-8*np.eye(3)      # this Q is for running observation only
        s.Q = Q     # this Q is for running slam

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

        s.current_particle = np.array([0, 0, 0])
        s.path = np.array([0, 0])
        s.show_free = np.array([0, 0])
        s.all_free = np.array([[0], [0]])

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir,
                                               'data/%s/%s_lidar%d'%(split,split,idx)))
        s.joint = load_joint_data(os.path.join(src_dir,
                                               'data/%s/%s_joint%d'%(split,split,idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t']-t))

    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135,135+s.lidar_angular_resolution,
                                   s.lidar_angular_resolution)*np.pi/180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3, s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n)/float(s.n)

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """
        #### TODO: XXXXXXXXXXX

        part_num = np.shape(p)[1]   # number of particles
        weight_sum = np.cumsum(w)   # create a roulette
        weight_sum /= weight_sum[-1]    # normalize

        n_rand_num = (np.linspace(0, part_num - 1, part_num) + np.random.uniform(size=part_num)) / part_num

        new_p = np.zeros(np.shape(p))
        new_w = np.ones(part_num) / part_num    # new weight array should all have the same weight

        # use roulette method, making clones of larger particles and discarding smaller particles
        # while keeping the weights of new particles the uniform
        new_idx = 0
        old_idx = 0
        while new_idx < part_num:
            while weight_sum[old_idx] < n_rand_num[new_idx]:
                old_idx += 1
            new_p[:, new_idx] = p[:, old_idx]
            new_idx += 1

        # new_p, new_w should both be (n,)
        return new_p, new_w

    @staticmethod
    def log_sum_exp(w):
        # log-sum-exp trick to deal with overflow and underflow when computing np.exp()
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def rays2world(s, p, d, rpy, head_angle=0, neck_angle=0, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be
        equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """
        #### TODO: XXXXXXXXXXX
        # raise NotImplementedError

        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data
        d = np.clip(d, s.lidar_dmin, s.lidar_dmax)

        points = np.zeros((3, len(d)))
        # free_points = ()

        # 1. from lidar distances to points in the LiDAR frame
        points[0, :] = np.cos(angles) * d
        points[1, :] = np.sin(angles) * d

        # 2. transformation matrices from LiDAR frame to world frame
        T_lidar_to_head = euler_to_se3(0, 0, 0, np.array([0, 0, s.lidar_height]))
        T_head_to_body = euler_to_se3(0, head_angle, neck_angle, np.array([0, 0, 0]))
        T_body_to_world = euler_to_se3(rpy[0], rpy[1], rpy[2],
                                       np.array([p[0], p[1], s.head_height]))

        # multiply transformation matrices to convert from lidar frame to world frame
        points = np.vstack((points, np.ones((1, len(d)))))      # format it to multiply with transformation matrices
        ray_in_world = T_body_to_world @ T_head_to_body @ T_lidar_to_head @ points

        # if the ray is too low, then it is hitting the floor not the wall, so we only include the hits
        # that are at least 0.1m off the ground
        true_wall = ray_in_world[2] > 0.1
        ray_in_world = ray_in_world[:, true_wall]

        # ray_in_world should be (4, n), but only returning x, y position, so it is (2, n)
        return ray_in_world[:2, :]

    def get_free_coor(s, occu_coor, particle):  #particle is in pixel coordinate

        free_coor = np.empty([2, 1], dtype=int)

        # iterate through all laser ray hits, find all the free cells between the robot estimated position and the cell
        # where laser scan hit something
        for i in range(np.shape(occu_coor)[1]):

            # create a line between current particle to the laser hit locations in grid space in both x and y axis
            x_idx = np.linspace(particle[0], occu_coor[0, i],
                                num=int(np.linalg.norm(occu_coor[:, i] - particle)), dtype=int, endpoint=False)
            y_idx = np.linspace(particle[1], occu_coor[1, i],
                                num=int(np.linalg.norm(occu_coor[:, i] - particle)), dtype=int, endpoint=False)
            if i == 0:
                free_coor[0, 0] = x_idx[0]
                free_coor[1, 0] = y_idx[0]

            free_coor = np.hstack((free_coor, np.vstack((np.reshape(x_idx[1:], (1, len(x_idx) - 1)),
                                                         np.reshape(y_idx[1:], (1, len(y_idx) - 1))))))

        # extract unique coordinates from the list
        free_coor = np.unique(free_coor, return_index=False, axis=1)
        # s.all_free = np.hstack((s.all_free , free_coor))
        # s.all_free = np.unique(s.all_free, return_index=False, axis=1)

        # should be in shape (2, a) where a is however many free coordinates it finds
        return free_coor

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two
        poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        #### TODO: XXXXXXXXXXX

        if t == 0:
            return np.zeros(3)

        xyth = s.lidar[t]['xyth']
        xyth_last = s.lidar[t - 1]['xyth']
        return smart_minus_2d(xyth, xyth_last)

    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of
        the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """
        #### TODO: XXXXXXXXXXX

        control = s.get_control(t)

        for i in range(s.n):
            s.p[:, i] = smart_plus_2d(s.p[:, i], control)
            s.p[:, i] = smart_plus_2d(s.p[:, i], np.random.multivariate_normal([0, 0, 0], s.Q))


    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        #### TODO: XXXXXXXXXXX

        # w_new = w_old * P(x | m) where P is probability that robot is at this particle
        w_new = np.zeros(np.shape(w))
        for i in range(len(w)):
            w_new[i] = w[i] * np.exp(obs_logp[i])

        # normalize the new weight array to sum up to 1
        w_new = w_new / np.sum(w_new)

        # w_new should be in shape (n,)
        return w_new

    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update
            the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the
        binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """
        #### TODO: XXXXXXXXXXX
        # raise NotImplementedError

        # start_time = time.time()

        # 1. find the head, neck angle at t
        rpy = s.lidar[t]['rpy']
        neck_angles = s.joint['head_angles'][0]
        head_angles = s.joint['head_angles'][1]

        # for initial timestep, it takes only 1 particle and sets its laserscan to be the true environment
        if t == 0:
            xyscan = s.rays2world(s.p[:, 0], s.lidar[t]['scan'], rpy, head_angles[s.find_joint_t_idx_from_lidar(t)],
                                  neck_angles[s.find_joint_t_idx_from_lidar(t)], s.lidar_angles)
            occu_coor = s.map.grid_cell_from_xy(xyscan[0, :], xyscan[1, :])
            for j in range(np.shape(occu_coor)[1]):
                s.map.cells[occu_coor[0, j], occu_coor[1, j]] = 1
            s.current_particle = s.p[:, 0]

            # add current_particle to path, which will be plotted
            s.path = s.map.grid_cell_from_xy(s.current_particle[0], s.current_particle[1])
            s.path = np.reshape(s.path, (2, 1))

        # iterate through all particles and localize to one particle, then map the environment
        else:

            # print("--- %s seconds ---" % (time.time() - start_time))

            # 2. project lidar scan into the world frame, update each particle weight
            log_p_obs = np.zeros(np.shape(s.p)[1])      # keep track of the log(P) for each particles

            # iterate through all particles and see how many of its readings match with known map
            for i in range(np.shape(s.p)[1]):

                # convert lidar readings into fixed world frame
                xyscan = s.rays2world(s.p[:, i], s.lidar[t]['scan'], rpy, head_angles[s.find_joint_t_idx_from_lidar(t)],
                                      neck_angles[s.find_joint_t_idx_from_lidar(t)], s.lidar_angles)

                # print("rays2world  --- %s seconds ---" % (time.time() - start_time))

                # convert xyscan's world frame coordinate (which is in meters) into grid world frame (which is pixels)
                occu_coor = s.map.grid_cell_from_xy(xyscan[0, :], xyscan[1, :])

                # print("grid from xy  --- %s seconds ---" % (time.time() - start_time))

                # count the number of lidar detected cells that are already occupied in the previous map.cell
                # log(P) = count
                # but map is either 1 or 0, so adding every map.cells' occupied coordinates is fine
                for j in range(np.shape(occu_coor)[1]):
                    log_p_obs[i] += s.map.cells[occu_coor[0, j], occu_coor[1, j]]

                # print("count each particle --- %s seconds ---" % (time.time() - start_time))

            # print("count all particles  --- %s seconds ---" % (time.time() - start_time))

            # 3. update the weights for each particles, used log sum exp trick for log_p_obs
            s.w = s.update_weights(s.w, log_p_obs - s.log_sum_exp(log_p_obs))

            # 4. find the current particle with largest weight, set that as the location of the robot
            # update the occupancy map from the lidar reading at this particle location
            s.current_particle = s.p[:, np.argmax(s.w)]
            s.path = np.hstack((s.path, np.reshape(s.map.grid_cell_from_xy(s.current_particle[0],
                                                                           s.current_particle[1]), (2, 1))))

            # recompute the readings for current_particle, convert to pixel world frame and map it
            # also compute the coordinates of the free cells from this lidar reading
            xyscan = s.rays2world(s.current_particle, s.lidar[t]['scan'], rpy,
                                  head_angles[s.find_joint_t_idx_from_lidar(t)],
                                  neck_angles[s.find_joint_t_idx_from_lidar(t)], s.lidar_angles)
            occu_coor = s.map.grid_cell_from_xy(xyscan[0, :], xyscan[1, :])
            free_coor = s.get_free_coor(occu_coor, s.map.grid_cell_from_xy(s.current_particle[0],
                                                                           s.current_particle[1]))
            s.show_free = free_coor

            # print("--- %s seconds ---" % (time.time() - start_time))

            # add free or occ to current log_odds map, which is kept throughout the whole process
            # add log_odds_occ if grid is measured to be occupied
            for i in range(np.shape(occu_coor)[1]):
                s.map.log_odds[occu_coor[0, i], occu_coor[1, i]] += s.lidar_log_odds_occ

            # subtract log_odds_free if it is free cell according to current_particle's lidar scan
            for i in range(np.shape(free_coor)[1]):
                s.map.log_odds[free_coor[0, i], free_coor[1, i]] += s.lidar_log_odds_free

            # constraint log_odds values between -max and max
            s.map.log_odds = np.clip(s.map.log_odds, -s.map.log_odds_max, s.map.log_odds_max)

            # 5. generate new map based on current log_odds value and threshold
            # set map.cells occupancy grid to 1 if log_odds at that point is larger than threshold
            s.map.cells = np.zeros(np.shape(s.map.cells))
            s.map.cells[np.argwhere(s.map.log_odds > s.map.log_odds_thresh)[:, 0],
                        np.argwhere(s.map.log_odds > s.map.log_odds_thresh)[:, 1]] = 1

            # print("--- %s seconds ---" % (time.time() - start_time))
            # print("one observation step done")

            # resample particles to avoid particle degradation
            s.resample_particles()

    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')
