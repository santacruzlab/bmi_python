import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from pylab import colorbar, cm, imshow
import time
 
from utils.util_fns import *
from utils.constants import *
import tables
from ismore import common_state_lists
import pandas as pd
import pickle
try:
    from ismore.invasive.make_global_armassist_hull import global_hull
except:
    from make_global_armassist_hull import global_hull

ips = common_state_lists.ismore_pos_states

class SafetyGrid(object):
    '''
    A class that discretizes the workspace into a grid. Each square in the
    grid contains information that can be used during an experiment to help
    implement safety measures related to the ArmAssist xy-position, ArmAssist 
    psi (orientation) angle, and ReHand pronosupination angle.
    '''
 
    # each square in the grid is of this dtype
    dtype = np.dtype([('is_valid',  np.bool_),
                      ('min_psi',   np.float64),
                      ('max_psi',   np.float64),
                      ('min_prono', np.float64),
                      ('max_prono', np.float64)])
 
    def __init__(self, mat_size, delta):
        self.mat_size = mat_size  # in cm (e.g., see settings.py)
        self.delta = float(delta)  # size of each square (e.g., width in cm)
 
        self.grid_shape = (
            int(np.ceil(mat_size[1] / self.delta)),  # nrows <--> y-size of mat
            int(np.ceil(mat_size[0] / self.delta)),  # ncols <--> x-size of mat
        )
        self._grid = np.zeros(self.grid_shape, dtype=self.dtype)
 
        self._grid['min_psi']   = np.nan
        self._grid['max_psi']   = np.nan
        self._grid['min_prono'] = np.nan
        self._grid['max_prono'] = np.nan
        self.global_hull = pickle.load(open('/home/tecnalia/code/ismore/invasive/armassist_hull.pkl'))
         
    def _pos_to_square(self, pos):
        return (int(pos[1] / self.delta), int(pos[0] / self.delta))
 
    def _square_to_pos(self, square):
        return (square[1] * self.delta, square[0] * self.delta)
 
    def _is_square_on_grid(self, square):
        return (0 <= square[0] < self._grid.shape[0]) and \
               (0 <= square[1] < self._grid.shape[1])
 
    def is_valid_pos(self, pos):
        try:
            tmp = self._grid[self._pos_to_square(pos)]['is_valid']
        except:
            tmp = False

        #global_valid = self.global_hull.hull_xy.find_simplex(pos) >=0
        
        # False if globally invalid
        #if global_valid == False:
        #    return False
        #else:
        # False if locally invalid
        if tmp == False:
            return False
        # True if both local and global valid
        else:
            return True
 
    def dist_to_interior(self, pos):
        return np.linalg.norm(self.interior_pos - pos)

    def dist_to_valid_point(self, pos):
        ''' Find distance to valid point on the safety grid '''
        
        pos_list = []
        dist_list = []
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                if self._grid[i, j]['is_valid']:
                    dist_list.append(np.linalg.norm(pos - self._square_to_pos([i, j])))
                    pos_list.append([i, j])

        min_ix = np.argmin(np.hstack((dist_list)))
        return dist_list[min_ix], self._square_to_pos(pos_list[min_ix])

    def get_minmax_psi(self, pos):
        '''Return a tuple with the min and max psi angle for a given xy-position.'''
         
        try:
            square = self._pos_to_square(pos)
            return self._grid[square]['min_psi'], self._grid[square]['max_psi']
        except:
            return 0., 0.

    def get_minmax_prono(self, pos):
        '''Return a tuple with the min and max prono angle for a given xy-position.'''
        try:
            square = self._pos_to_square(pos)
            return self._grid[square]['min_prono'], self._grid[square]['max_prono']
        except:
            return 0., 0.
            
    def set_valid_boundary(self, positions):
        '''Given a list of positions defining the outer boundary, mark the
        corresponding squares as valid in the underlying grid representation.
        Positions should have shape (n_positions, 2). The first and last 
        position should be roughly the same.
        '''
 
        # iterate through each position in the list of positions
        for idx in range(positions.shape[0]):
            pos = positions[idx, :]
 
            if idx == positions.shape[0] - 1:
                next_pos_idx = 0
            else:
                next_pos_idx = idx + 1
            next_pos = positions[next_pos_idx, :]
 
            n_pts = max(3, int(10 * (dist(pos, next_pos) / self.delta)))
 
            # consider equally-spaced pts on a virtual line from pos to next_pos
            for weight in np.linspace(0, 1, n_pts):
                pos_on_line = (1-weight) * pos + weight * next_pos
                square = self._pos_to_square(pos_on_line)
 
                # for each pt on this line, mark the corresponding square in 
                # the safety grid as a valid position
                self._grid[square]['is_valid'] = True
 
    def update_minmax_psi(self, pos, psi, local_dist):
        '''Given an xy-position and a psi angle value, update the min/max 
        psi value for all squares within a radius of local_dist cm.
        '''
 
        square = self._pos_to_square(pos)
        n_squares = int(np.ceil(local_dist / self.delta))
 
        row_start = max(square[0] - n_squares, 0)
        row_end   = min(square[0] + n_squares, self._grid.shape[0] - 1)
        col_start = max(square[1] - n_squares, 0)
        col_end   = min(square[1] + n_squares, self._grid.shape[1] - 1)
 
        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                square = (row, col)
                if self._grid[square]['is_valid'] and dist(pos, self._square_to_pos(square)) <= local_dist:
                    # min() and max() functions will overwrite np.nan values
                    self._grid[square]['min_psi'] = min(psi, self._grid[square]['min_psi'])
                    self._grid[square]['max_psi'] = max(psi, self._grid[square]['max_psi'])
 
    def update_minmax_prono(self, pos, prono, local_dist):
        '''Given an xy-position and a prono angle value, update the min/max
        prono value for all squares within a radius of local_dist cm.
        '''
 
        square = self._pos_to_square(pos)
        n_squares = int(np.ceil(local_dist / self.delta))
 
        row_start = max(square[0] - n_squares, 0)
        row_end   = min(square[0] + n_squares, self._grid.shape[0] - 1)
        col_start = max(square[1] - n_squares, 0)
        col_end   = min(square[1] + n_squares, self._grid.shape[1] - 1)
 
        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                square = (row, col)
                if self._grid[square]['is_valid'] and dist(pos, self._square_to_pos(square)) <= local_dist:
                    # min() and max() functions will overwrite np.nan values
                    self._grid[square]['min_prono'] = min(prono, self._grid[square]['min_prono'])
                    self._grid[square]['max_prono'] = max(prono, self._grid[square]['max_prono'])
 
    def mark_interior_as_valid(self, interior_pos):
        '''Assuming that the boundary of valid positions has already been set,
        marks all the squares in the interior as valid too. The argument 
        interior_pos should be a xy-position that is known to be in the 
        interior.
        '''
 
        # starting at interior_pos, do a breadth-first traversal of squares
        #   (using a queue) and use a set to keep track of which squares have
        #   already been visited
        self.interior_pos = interior_pos
        starting_square = self._pos_to_square(interior_pos)
        queue = deque([starting_square])
        visited = set()
 
        while len(queue) > 0:  # while the queue is not empty
            # get the next square to be processed (i.e., to be marked as valid)
            square = queue.popleft()
 
            self._grid[square]['is_valid'] = True
            visited.add(square)
 
            row = square[0]
            col = square[1]
            # if a neighboring square is:
            #   1) on the grid,
            #   2) hasn't already been visited,
            #   3) isn't already in the queue, and
            #   4) isn't already marked as valid
            # then mark add it to the queue 
            for neighbor in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
                if neighbor not in visited and neighbor not in queue:
                    if self._is_square_on_grid(neighbor):
                        if not self._grid[neighbor]['is_valid']:
                            queue.append(neighbor)
 
    def is_psi_minmax_set(self):
        '''Return true if the min/max psi value is set for all squares that 
        are marked as valid.
        '''
 
        for row in range(self._grid.shape[0]):
            for col in range(self._grid.shape[1]):
                if self._grid[row][col]['is_valid'] and np.isnan(self._grid[row][col]['min_psi']):
                        return False
        return True
 
    def is_prono_minmax_set(self):
        '''Return true if the min/max prono value is set for all squares that 
        are marked as valid.
        '''
 
        for row in range(self._grid.shape[0]):
            for col in range(self._grid.shape[1]):
                if self._grid[row][col]['is_valid']:
                    if np.isnan(self._grid[row][col]['min_prono']) or np.isnan(self._grid[row][col]['max_prono']):
                        return False
        return True
 
    def plot_valid_area(self):
        '''Plot the valid xy area of the workspace.'''
 
        extent = [0, self.mat_size[0], 0, self.mat_size[1]]
 
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(self._grid[:]['is_valid'], 
            interpolation='none', extent=extent, origin='lower')
        ax.set_title('Valid ArmAssist positions')
        ax.set_xlabel('cm')
        ax.set_ylabel('cm')
 
    def plot_minmax_psi(self):
        '''Plot the min/max psi for each position.'''
 
        extent = [0, self.mat_size[0], 0, self.mat_size[1]]
 
        global_min = rad_to_deg * np.nanmin(self._grid[:]['min_psi'])
        global_max = rad_to_deg * np.nanmax(self._grid[:]['max_psi'])
 
        fig, axes = plt.subplots(nrows=1, ncols=2)
        variables = ['min_psi', 'max_psi']
        titles = ['Min psi angle', 'Max psi angle']
        for ax, var, title in zip(axes.flat, variables, titles):
            matrix = rad_to_deg * self._grid[:][var]
            for row in range(self._grid.shape[0]):
                for col in range(self._grid.shape[1]):
                    if self._grid[row][col]['is_valid'] and np.isnan(self._grid[row][col][var]):
                        matrix[row][col] = 1e6
 
            im = ax.imshow(matrix, 
                interpolation='none', origin='lower', extent=extent, vmin=global_min, vmax=global_max)
            ax.set_title(title + '\n(black = no value)')
            ax.set_xlabel('cm')
            ax.set_ylabel('cm')
         
        im.cmap.set_over('k')
        im.set_clim(global_min, global_max)
 
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar_ax.set_title('degrees')
        fig.colorbar(im, cax=cbar_ax)
 
    def plot_minmax_prono(self):
        '''Plot the min/max prono angle for each position.'''
 
        extent = [0, self.mat_size[0], 0, self.mat_size[1]]
 
        global_min = rad_to_deg * np.nanmin(self._grid[:]['min_prono'])
        global_max = rad_to_deg * np.nanmax(self._grid[:]['max_prono'])
        if np.isnan(global_min):
            global_min = 0
            global_max = 1
 
        fig, axes = plt.subplots(nrows=1, ncols=2)
        variables = ['min_prono', 'max_prono']
        titles = ['Min prono angle', 'Max prono angle']
        for ax, var, title in zip(axes.flat, variables, titles):
            matrix = rad_to_deg * self._grid[:][var]
            for row in range(self._grid.shape[0]):
                for col in range(self._grid.shape[1]):
                    if self._grid[row][col]['is_valid'] and np.isnan(self._grid[row][col][var]):
                        matrix[row][col] = 1e6
 
            im = ax.imshow(matrix, 
                interpolation='none', origin='lower', extent=extent, vmin=global_min, vmax=global_max)
            ax.set_title(title + '\n(black = no value)')
            ax.set_xlabel('cm')
            ax.set_ylabel('cm')
 
        im.cmap.set_over('k')
        im.set_clim(global_min, global_max)
         
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar_ax.set_title('degrees')
        fig.colorbar(im, cax=cbar_ax)
 
    def calculate_valid_area(self):
        '''Return estimate of the range of motion area in cm^2.'''
         
        n_valid_squares = 0
        for row in range(self._grid.shape[0]):
            for col in range(self._grid.shape[1]):
                if self._grid[row][col]['is_valid']:
                    n_valid_squares += 1
 
        return n_valid_squares * self.delta**2
 
    def define_finger_minmax(self, rh_pos):
        ''' 
        Accepts npoints x 3, where 3 columns are rh_thumb, rh_index, rh_3finger
        Finds min / max of each of these angles: 
        '''
        self.rh_pthumb = [np.min(rh_pos[:]['rh_pthumb']), np.max(rh_pos[:]['rh_pthumb'])]
        self.rh_pindex = [np.min(rh_pos[:]['rh_pindex']), np.max(rh_pos[:]['rh_pindex'])]
        self.rh_pfing3 = [np.min(rh_pos[:]['rh_pfing3']), np.max(rh_pos[:]['rh_pfing3'])]
 
    def get_rh_minmax(self, attr):
        return getattr(self, attr)

class SafetyGridTargetsMatrix(SafetyGrid):
    ''' 
    Class to create a simpler safety environment based on a calibration file that goes
    to 3 distinct X/Y targets + Rest. At each target, the arm is moved to test the comfortable
    range of psi, and the hand is opened and closed to test the comfortable range of hand opening
    and closing. A similar grid as SafetyGrid is created, but now the min / max psi and prono values
    are interpolated from the three targets + rest

    During the calibration phase (similar to create_targets_matrix.py) , the user needs to click "start"
    when they are at the desired location. After that, they can test psi / prono / open + close of hand. Then 
    they should click "accept" to signify the end of that targets' calibration time. 

    The data used for the angular calibrations is all data in between start and accepts. 
    '''

    def __init__(self, mat_size, delta, boundary_tolerence, angle_tolerence, hdf_file_list, 
        local_dist, local_dist_gte_45x):
        super(SafetyGridTargetsMatrix, self).__init__(mat_size, delta)
        
        self.hdf_file_list = hdf_file_list #tables.openFile(hdf_file)
        self.parse_hdf_file(hdf_file_list)

        self.boundary_tolerence = boundary_tolerence
        self.angle_tolerence = angle_tolerence
        self.local_dist = local_dist
        self.local_dist_gte_45x = local_dist_gte_45x
 
    def parse_hdf_file(self, hdf_file_list):
        ''' Extract target calibration epochs from HDF file -- starting w/ 'start' --> 'accept'''

        dt = []
        for d in common_state_lists.rh_pos_states+common_state_lists.rh_vel_states:
            dt.append((d, np.float64))

        self.dt = dt
        self.rh_data = np.zeros((1, ), dtype=dt)
        self.epoch_ix = {}
        self.targets_xy = {}
        self.psi_pos = {}
        self.prono_pos = {}
        self.trial_types = []
        cnt = 0

        for hdf_f in hdf_file_list:
            hdf = tables.openFile(hdf_f)
        
            self.rh_data = np.hstack((hdf.root.rehand[:]['data'], self.rh_data)) 

            # Find the first start ix: 
            start_ix_all = np.nonzero(hdf.root.task[:]['trial_accept_reject'] == 'start')[0]
            dsi = np.diff(start_ix_all)
            start_ix_filt = [start_ix_all[0]]
            for i, j in enumerate(dsi):
                if j > 1:
                    start_ix_filt.append(start_ix_all[i+1])
            start_ix = np.array(start_ix_filt)
             
            # Extract calibration epochs:

            for i, ia in enumerate(start_ix):
                acc_ix = ia + np.nonzero(hdf.root.task[ia:]['trial_accept_reject'] == 'accept')[0]

                if i + 1 < len(start_ix):
                    # Accept index is less than the next start index
                    assert acc_ix[0] < start_ix[i+1]
                else:
                    # If last start index, make sure accept index is not empty
                    assert len(acc_ix) > 0
                
                # If pass tests, get next accept index
                acc_ix = acc_ix[0]
                trial_type = 'trial_'+str(i+cnt)#hdf.root.task[acc_ix - 1]['trial_type']
                self.epoch_ix[trial_type] = [ia, acc_ix]
                self.targets_xy[trial_type] = pd.Series(hdf.root.task[ia]['plant_pos'], ips)[common_state_lists.aa_xy_states]
                self.psi_pos[trial_type] = pd.DataFrame(hdf.root.task[ia:acc_ix]['plant_pos'], columns=ips)['aa_ppsi']
                self.prono_pos[trial_type] = pd.DataFrame(hdf.root.task[ia:acc_ix]['plant_pos'], columns=ips)['rh_pprono']
                self.trial_types.append(trial_type)
            cnt += i+1
 
    def set_valid_boundary(self, positions=None):
        if positions is None:
            positions = np.vstack(([v for i, (k, v) in enumerate(self.targets_xy.items())]))
        super(SafetyGridTargetsMatrix, self).set_valid_boundary(positions)
        self.mean_pos = np.mean(positions, axis=0)
        self.mark_interior_as_valid(self.mean_pos)

    def update_minmax_psi_prono(self, psi_or_prono = 'psi'):
        max_val = {}
        min_val = {}
        #local_dist = self.local_dist
        local_dist_dict = {}

        for i, trl in enumerate(self.trial_types):
            if psi_or_prono == 'psi':
                val = self.psi_pos[trl]
                max_key = 'max_psi'
                min_key = 'min_psi'

            elif psi_or_prono == 'prono':
                val = self.prono_pos[trl]
                max_key = 'max_prono'
                min_key = 'min_prono'
                
            pos = self._pos_to_square(self.targets_xy[trl])

            # Set min/ max psi for later interpolation
            max_val[trl] = np.max(val)
            min_val[trl] = np.min(val)

            if self.targets_xy[trl]['aa_px'] > 45 and psi_or_prono == 'psi':
                max_local_dist = self.local_dist_gte_45x
                min_local_dist = self.local_dist
            else:
                max_local_dist = self.local_dist
                min_local_dist = self.local_dist

            local_dist_dict[trl, 'min'] = min_local_dist
            local_dist_dict[trl, 'max'] = max_local_dist

            for im, (nm, local_dist) in enumerate(zip(['min', 'max'], [min_local_dist, max_local_dist])):
                print nm, local_dist, trl, psi_or_prono

                n_squares = int(np.ceil(local_dist / self.delta))

                row_start = max(pos[0] - n_squares, 0)
                row_end   = min(pos[0] + n_squares, self._grid.shape[0] - 1)
                col_start = max(pos[1] - n_squares, 0)
                col_end   = min(pos[1] + n_squares, self._grid.shape[1] - 1)
     
                for row in range(row_start, row_end + 1):
                    for col in range(col_start, col_end + 1):
                        square = (row, col)
                        if self._grid[square]['is_valid'] and dist(self.targets_xy[trl], self._square_to_pos(square)) <= local_dist:
                            # min() and max() functions will overwrite np.nan values
                            if nm == 'min':
                                self._grid[square][min_key] = np.nanmin(list(val) + [self._grid[square][min_key]])
                            elif nm == 'max':
                                self._grid[square][max_key] = np.nanmax(list(val) + [self._grid[square][max_key]])

        # Now interpolate remainder of valid squares: 
        is_valid = np.nonzero(self._grid['is_valid']==True)

        for i, (x, y) in enumerate(zip(is_valid[0], is_valid[1])):
            square = (x, y)
            grid_sq = self._grid[square]
            if np.logical_or(np.isnan(grid_sq[min_key]), np.isnan(grid_sq[max_key])):

                wt_max_psi = 0
                wt_min_psi = 0

                for im, nm in enumerate(['min', 'max']):
                    wts = []
                    for trl in self.trial_types:
                        d = np.max([self.delta, dist(self._square_to_pos(square), self.targets_xy[trl]) - local_dist_dict[trl, nm]])
                        wts.append(1./d)
                        if nm == 'min':
                            wt_min_psi += (1./d)*min_val[trl]
                        elif nm == 'max':
                            wt_max_psi += (1./d)*max_val[trl]

                    norm = float(np.sum(wts))
                    if nm == 'min':
                        self._grid[square][min_key] = wt_min_psi/norm
                    elif nm == 'max':
                        self._grid[square][max_key] = wt_max_psi/norm
                

        # Do rehand: 
        self.define_finger_minmax(self.rh_data)

    def update_targ_pos(self, target_matrix, trials_to_add):
        for t in trials_to_add:
            #N = target_matrix[t].shape[1]
            #for n in range(N):
            targ = target_matrix[t[0]][t[1]]
            k = t[0]+'_'+str(t[1])
            self.targets_xy[k] = targ[common_state_lists.aa_xy_states]
            self.psi_pos[k] = np.array([targ['aa_ppsi']])
            self.prono_pos[k] = np.array([targ['rh_pprono']])
            self.trial_types.append(k)

            rh = np.hstack((targ[common_state_lists.rh_pos_states], np.zeros((4,)) ))
            self.rh_data = np.hstack((self.rh_data, np.array(rh, dtype=self.dt)))

    def define_attractor(self, x, y, dist):
        ''' XY point that is desired attractor point '''
        mn, mx = self.get_minmax_psi((x, y+dist))
        psi = np.mean([mn, mx])

        mn, mx = self.get_minmax_prono((x, y+dist))
        pron = np.mean([mn, mx])

        mn, mx = self.get_rh_minmax('rh_pindex')
        idx = np.mean([mn, mx]) 
          
        mn, mx = self.get_rh_minmax('rh_pthumb')
        thb = np.mean([mn, mx])

        mn, mx = self.get_rh_minmax('rh_pfing3')
        fing3 = np.mean([mn, mx])

        self.attractor_point = np.array([x, y+dist, psi, thb, idx, fing3, pron])
if __name__ == '__main__':
    # below is a simple example of how a SafetyGrid can be created
    # see the scripts:
    #   define_safety_boundary.py
    #   define psi_prono_safety_range.py
    # for full usage
 
    safety_grid = SafetyGrid([95, 85], 0.5)
 
    interior_pos = (50, 50)
    angles = np.linspace(0, 2 * np.pi, 100)
    radius = 10
    x = radius*np.cos(angles) + 0.1*np.random.randn(len(angles)) + interior_pos[0]
    y = radius*np.sin(angles) + 0.1*np.random.randn(len(angles)) + interior_pos[1]
    boundary_positions = np.array([x, y]).T
 
    safety_grid.set_valid_boundary(boundary_positions)
    safety_grid.mark_interior_as_valid(interior_pos)
    safety_grid.plot_valid_area()
