from __future__ import division
import numpy as np
import pandas as pd
import random
import cv2

# ---------- script configuration ----------------------------------------------
print_size = 'A1'
file_name = 'maze21.png'
cell_size = 10 # cell size in mm, this is the distance between walls
dpi = 300 # print quality in dots per inch
# the following parameters control the complexity of the maze
walker_every_x_steps = 10
# ------------------------------------------------------------------------------

def main():
    # setup the image object for the maze
    image = MazeImage(print_size, cell_size, dpi) 
    # create the maze
    maze_dimensions = {'rows': image.rows, 'columns': image.columns}
    maze_entry = {'row': 0, 'column': 0}
    maze_exit = {'row': image.rows-1, 'column': image.columns-1}
    maze = Maze(maze_dimensions, maze_entry, maze_exit,
                walker_every_x_steps)
    maze.create_maze(start_row = maze_entry['row'],
                     start_column = maze_entry['column'])
    # save the image in an image file
    image.maze_to_image(maze, file_name, paths = False, path_through = False,
                        walkers_through_maze = True)
    maze.print_stats()
    print('Done!')

class MazeImage:
    
    def __init__(self, print_size, cell_size, dpi):
        paper_sizes = {"4A0": {"width": 1682, "height": 2378},
                       "2A0": {"width": 1189, "height": 1682},
                       "A0": {"width": 841, "height": 1189},
                       "A1": {"width": 594, "height": 841},
                       "A2": {"width": 420, "height": 594},
                       "A3": {"width": 297, "height": 420},
                       "A4": {"width": 210, "height": 297},
                       "A5": {"width": 148, "height": 210},
                       "A6": {"width": 105, "height": 148},
                       "A7": {"width": 74, "height": 105},
                       "A8": {"width": 52, "height": 74},
                       "A9": {"width": 37, "height": 52},
                       "A10": {"width": 26, "height": 37}}
        self.colours = {'black': (0,0,0),
                        'red': (0,0,255),
                        'blue': (255,0,0),
                        'green': (0,255,0)}
        inch = 25.4 # inch to mm
        self.dpmm = dpi/inch # dots per mm
        self.rows = int(paper_sizes[print_size]['height']/cell_size)
        self.columns = int(paper_sizes[print_size]['width']/cell_size)
        self.border_up = int(cell_size * self.dpmm)
        self.border_down = int(cell_size * self.dpmm)
        self.border_right = int(cell_size * self.dpmm)
        self.border_left = int(cell_size * self.dpmm)

    def maze_to_image(self, maze, image_file_name, 
                      walls = True, paths = False, path_through = False, 
                      walkers_through_maze = False):
        img_width = self.border_left + \
                    int(maze.maze_dimensions['columns'] * cell_size * self.dpmm) + \
                    self.border_right
        img_height = self.border_up + \
                     int(maze.maze_dimensions['rows'] * cell_size * self.dpmm) + \
                     self.border_down
        self.img = np.ones((img_height,img_width,3), np.uint8)*255
        # draw all walls
        if walls:
            for _, row in maze.maze.iterrows():
                for cell in row:
                    self.draw_walls(cell, self.colours['black'])
        # draw all paths
        if paths:
            for walker in maze.walker_list:
                path = walker.path_steps
                for i, _ in enumerate(path[:-1]):
                    self.draw_line(path[i], path[i+1], self.colours['blue'])
        # draw the path through the maze
        if path_through:
            for i, _ in enumerate(maze.path_through_maze[:-1]):
                self.draw_line(maze.path_through_maze[i],
                               maze.path_through_maze[i+1],
                               self.colours['green'])
        # show the walker paths that made up the path through the maze
        if walkers_through_maze:
            for walker in maze.walkers_through_maze:
                path = walker['walker'].path_steps
                for i, _ in enumerate(path[:-1]):
                    self.draw_line(path[i], path[i+1], self.colours['green'])
        cv2.imwrite(image_file_name, self.img)

    def draw_walls(self, cell, colour):
        row = cell['position']['row']
        column = cell['position']['column']
        pt = 5 # line thickness
        if cell['walls']['up']:
            startx = int((column * cell_size) * self.dpmm)
            starty = int((row * cell_size) * self.dpmm)
            endx   = int((column * cell_size + cell_size) * self.dpmm)
            endy   = starty
            self.cv2_line(startx, starty, endx, endy, colour, pt)
        if cell['walls']['down']:
            startx = int((column * cell_size) * self.dpmm)
            starty = int((row * cell_size + cell_size) * self.dpmm)
            endx   = int((column * cell_size + cell_size) * self.dpmm)
            endy   = starty
            self.cv2_line(startx, starty, endx, endy, colour, pt)
        if cell['walls']['right']:
            startx = int((column * cell_size + cell_size) * self.dpmm)
            starty = int((row * cell_size) * self.dpmm)
            endx   = startx
            endy   = int((row * cell_size + cell_size) * self.dpmm)
            self.cv2_line(startx, starty, endx, endy, colour, pt)
        if cell['walls']['left']:
            startx = int((column * cell_size) * self.dpmm)
            starty = int((row * cell_size) * self.dpmm)
            endx   = startx
            endy   = int((row * cell_size + cell_size) * self.dpmm)
            self.cv2_line(startx, starty, endx, endy, colour, pt)

    def draw_line(self, start_cell, end_cell, colour):
        startx = int((start_cell[1] * cell_size + cell_size/2) * self.dpmm)
        starty = int((start_cell[0] * cell_size + cell_size/2) * self.dpmm)
        endx   = int((end_cell[1] * cell_size + cell_size/2) * self.dpmm)
        endy   = int((end_cell[0] * cell_size + cell_size/2) * self.dpmm)
        self.cv2_line(startx, starty, endx, endy, colour, 1)
        
    def cv2_line(self, startx, starty , endx, endy , colour, pt):
        self.img = cv2.line(self.img,
                            (startx + self.border_left, starty + self.border_up),
                            (endx + self.border_left, endy + self.border_up), colour, pt)

class Maze:
    
    def __init__(self, maze_dimensions, maze_entry, maze_exit,
                 walker_every_x_steps = 10):
        self.maze = None
        self.maze_dimensions = maze_dimensions
        self.maze_entry = maze_entry
        self.maze_exit = maze_exit
        self.walker_every_x_steps = walker_every_x_steps
        self.active_walkers = []
        self.walker_tree = None # store all walkers in a tree structure
        self.walker_list = [] # store all walkers in a list
        self.path_through_maze = []
        self.walkers_through_maze = []
        self.initialise_maze()

    def initialise_maze(self):
        self.maze = pd.DataFrame(index = range(self.maze_dimensions['rows']),
                                 columns = range(self.maze_dimensions['columns']))
        for column in range(self.maze.shape[1]):
            for row in range(self.maze.shape[0]):
                cell = {'position': {'row': row, 'column': column},
                        'walls': {'up': True, 'down': True, 'right': True, 'left': True},
                        'isWayPoint': False}
                self.maze.set_value(row, column, cell.copy())

    def create_maze(self, start_row = 0, start_column = 0):
        self.create_paths(start_row, start_column)
        self.set_walls()
        self.open_maze(self.maze_entry['row'], self.maze_entry['column'])
        self.open_maze(self.maze_exit['row'], self.maze_exit['column'])
        self.get_through_maze(self.maze_exit['row'], self.maze_exit['column'])

    def create_paths(self, row = 0, column = 0):
        if not(self.active_walkers):
            # initialise the first walker
            self.walker_tree = self.create_walker(row, column)
        while self.active_walkers:
            for walker in self.active_walkers:
                walker.move_one_step()
                self.remove_stuck_walkers(walker)
                self.start_new_walkers(walker)
                self.remove_finished_walker(walker)

    def create_walker(self, row, column, parent_walker = None):
        walker = Walker(self.maze, row, column, parent_walker)
        self.active_walkers.append(walker)
        self.walker_list.append(walker)
        if parent_walker != None:
            # Only if the walker is not the root of the walker tree
            # add the new walker as a child node to its parent.
            parent_walker.walker_childs.append(walker)
        return walker

    def start_new_walkers(self, walker):
        if (walker.step_count % self.walker_every_x_steps) == 0:
            # Start a new walker at the same position of the current walker.
            self.create_walker(walker.row, walker.column, walker)
        if walker.direction == 'trace_back' and \
           not(walker.is_paused) and \
           len(walker.breadcrumbs) > 0:
            # Start a new walker at the current trace back point.
            self.create_walker(walker.row, walker.column, walker)
            walker.is_paused = True

    def remove_stuck_walkers(self, walker):
        # Remove walkers that couldn't make more than one step from the 
        # walker tree.
        if walker.is_finished and len(walker.path_steps) <= 1:
            walker.parent_walker.walker_childs.remove(walker)
            self.walker_list.remove(walker)

    def remove_finished_walker(self, walker):
        if walker.is_finished:
            if not(walker.parent_walker == None):
                # parent walker to continue trace back
                walker.parent_walker.is_paused = False
            self.active_walkers.remove(walker)

    def set_walls(self):
        for w in self.walker_list:
            path = w.path_steps
            for i, _ in enumerate(path[:-1]):
                self.remove_wall(path[i], path[i+1])

    def remove_wall(self, start_cell, end_cell):
        lookup_table = {(-1, 0): {'start': 'up', 'end': 'down'},
                        (1, 0): {'start': 'down', 'end': 'up'},
                        (0, 1): {'start': 'right', 'end': 'left'},
                        (0, -1): {'start': 'left', 'end': 'right'}}
        dr = end_cell[0] - start_cell[0]
        dc = end_cell[1] - start_cell[1]
        # remove wall in start cell
        cell = self.maze.loc[start_cell[0], start_cell[1]]
        start_cell_wall = lookup_table[(dr, dc)]['start']
        cell['walls'][start_cell_wall] = False
        self.maze.set_value(start_cell[0], start_cell[1], cell)
        # remove wall in end cell
        cell = self.maze.loc[end_cell[0], end_cell[1]]
        end_cell_wall = lookup_table[(dr, dc)]['end']
        cell['walls'][end_cell_wall] = False
        self.maze.set_value(end_cell[0], end_cell[1], cell)

    def open_maze(self, row, column):
        cell = self.maze.loc[row, column]
        if row == 0:
            cell['walls']['up'] = False
            self.maze.set_value(row, column, cell)
        if row == self.maze.index[-1]:
            cell['walls']['down'] = False
            self.maze.set_value(row, column, cell)
        if column == 0:
            cell['walls']['left'] = False
            self.maze.set_value(row, column, cell)
        if column == self.maze.columns[-1]:
            cell['walls']['right'] = False
            self.maze.set_value(row, column, cell)

    def get_through_maze(self, exit_row, exit_column):
        # find the paths that get to the exit
        walker, path_index = self.find_exit_path(exit_row, exit_column)
        self.trace_back_to_start(walker, path_index)

    def find_exit_path(self, exit_row, exit_column):
        for w in self.walker_list:
            pos = [i for i, x in enumerate(w.path_steps) 
                   if x == [exit_row, exit_column]]
            if pos:
                index = pos[0]
                break
        return w, index

    def trace_back_to_start(self, walker, index):
        self.path_through_maze = walker.path_steps[:index+1]
        self.walkers_through_maze = [{'walker': walker, 'index': index}]
        while not(walker.parent_walker == None):
            # get the beginning of the child path
            row, column = walker.path_steps[0]
            walker = walker.parent_walker
            # find the position where the child path connects to the parent path
            pos = [i for i, x in enumerate(walker.path_steps) 
                   if x == [row, column]]
            if pos:
                index = pos[0]
                self.path_through_maze = walker.path_steps[:index] + self.path_through_maze
                self.walkers_through_maze = [{'walker': walker, 'index': index}] + self.walkers_through_maze
    
#     def calculate_complexity(self):
#         for step in self.path_through_maze:
#             row, column = step
#             walker = Walker(self.maze, row, column, None)
            
            #TODO: to be completed
    
    def print_stats(self):
        print('\nmaze statistics\n---------------\n')
        print('maze dimensions: {} rows by {} columns'.format(self.maze_dimensions['rows'],
                                                              self.maze_dimensions['columns']))
        print('length of path through maze: {} steps'.format(len(self.path_through_maze)))
        print('number of paths: {}'.format(len(self.walker_list)))
        print('number of walkers that created path through maze: {}'.format(len(self.walkers_through_maze)))

class Walker:
    
    def __init__(self, maze, row, column, parent_walker):
        # deltas for each step direction
        self.pos = {'up': {'c': 0, 'r': -1},
                    'right': {'c': 1, 'r': 0}, 
                    'left': {'c': -1, 'r': 0}, 
                    'down': {'c': 0, 'r': 1}}
        self.maze = maze
        self.row = row
        self.column = column
        self.parent_walker = parent_walker # which walker generated this walker?
        self.walker_childs = [] # list of child walkers generated by this walker
        self.direction = 'forward'
        self.is_paused = False
        self.is_finished = False
        self.step_count = 0
        self.path_steps = []
        self.breadcrumbs = []
        self.set_way_point(self.row, self.column)

    def move_one_step(self):
        if self.direction == 'forward':
            directions = self.get_possible_directions()
            if directions:
                # when not in a dead end choose the direction of the next step
                next_step = random.choice(directions)
                self.row = self.row + self.pos[next_step]['r']
                self.column = self.column + self.pos[next_step]['c']
                self.set_way_point(self.row, self.column)
            else:
                # when in a dead end switch to trace back mode
                self.direction = 'trace_back'
        if self.direction == 'trace_back' and not(self.is_paused):
            self.trace_back()

    def set_way_point(self, row, column):
        self.breadcrumbs.append([row, column])
        self.path_steps.append([row, column])
        self.step_count += 1
        cell = self.maze.loc[row, column]
        cell['isWayPoint'] = True
        self.maze.set_value(row, column, cell)

    def get_possible_directions(self):
        cell = self.maze.loc[self.row, self.column]
        directions = self.get_directions(cell)
        directions = self.avoid_walls(cell)
        directions = self.avoid_outside_maze(directions, cell)
        directions = self.avoid_set_way_points(directions, cell)
        return directions

    def get_directions(self, cell):
        directions = [w for w in cell['walls']]
        return directions

    def avoid_walls(self, cell): # not used but kept for future use
        directions = [w for w in cell['walls'] 
                      if cell['walls'][w] == False]
        return directions
    
    def avoid_outside_maze(self, directions, cell):
        r = cell['position']['row']
        c = cell['position']['column']
        directions = [d for d in directions
                      if r + self.pos[d]['r'] in self.maze.index and c + self.pos[d]['c'] in self.maze.columns]
        return directions
    
    def avoid_set_way_points(self, directions, cell):
        r = cell['position']['row']
        c = cell['position']['column']
        directions = [d for d in directions 
                      if self.maze.loc[r + self.pos[d]['r'], c + self.pos[d]['c']]['isWayPoint'] == False]
        return directions
    
    def trace_back(self):
        if self.breadcrumbs:
            self.breadcrumbs.pop() # go back one step
            if self.breadcrumbs:
                # use current position as starting point for new path
                self.row, self.column = self.breadcrumbs.pop()
            else:
                # The walker can't move anywhere.
                self.is_finished = True
        else:
            # The walker has traced back his whole path and is now back at his
            # starting point.
            self.is_finished = True

# ------------------- main -----------------------------------------------------

if __name__ == "__main__":
    main()
