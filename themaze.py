from __future__ import division
import numpy as np
import pandas as pd
import random
import cv2

# ---------- script configuration ----------------------------------------------
print_size = 'A4'
cell_size = 10 # cell size in mm
dpi = 300 # print quality in dots per inch
# ------------------------------------------------------------------------------

def main():
    paper_sizes = {
        "4A0": {"width": 1682, "height": 2378},
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
        "A10": {"width": 26, "height": 37}
        }
    inch = 25.4 # inch to mm
    dpmm = dpi/inch # dots per mm
    rows = int(paper_sizes[print_size]['height']/cell_size)
    columns = int(paper_sizes[print_size]['width']/cell_size)

    # create the maze
    maze_dimensions = {'rows': rows, 'columns': columns}
    maze_entry = {'row': 0, 'column': 0}
    maze_exit = {'row': rows-1, 'column': columns-1}
    maze = Maze(maze_dimensions, maze_entry, maze_exit)
    maze.create_maze(start_row = maze_entry['row'],
                     start_column = maze_entry['column'])
    maze.save_maze_image(dpmm)
    print('Done!')

class Maze:
    
    def __init__(self, maze_dimensions, maze_entry, maze_exit):
        # deltas for each step direction
        self.pos = {'up': {'c': 0, 'r': -1},
                    'right': {'c': 1, 'r': 0}, 
                    'left': {'c': -1, 'r': 0}, 
                    'down': {'c': 0, 'r': 1}}
        self.maze_dimensions = maze_dimensions
        self.maze_entry = maze_entry
        self.maze_exit = maze_exit
        self.step_count = 0
        self.path_steps = []
        self.paths = []
        self.breadcrumbs = []
        self.path_through_maze = []
        self.initialise_maze()

    def initialise_maze(self):
        self.maze = pd.DataFrame(index = range(self.maze_dimensions['rows']),
                                 columns = range(self.maze_dimensions['columns']))
        for column in range(self.maze.shape[1]):
            for row in range(self.maze.shape[0]):
                self.maze.set_value(row, column, {'position': {'row': row,
                                                               'column': column},
                                                  'walls': {'up': True,
                                                            'down': True,
                                                            'right': True,
                                                            'left': True},
                                                  'isWayPoint': False})
    def create_maze(self, start_row = 0, start_column = 0):
        self.create_paths(start_row, start_column)
        self.set_walls()
        return self.paths

    def create_paths(self, row = 0, column = 0):
        while True: # change recursion to a while loop
            cell = self.set_way_point(row, column)
            if cell['position'] == self.maze_exit:
                # We found the exit! Save the path through the maze.
                self.path_through_maze = self.breadcrumbs[:]
            directions = self.get_possible_directions(cell)
            if directions:
                # when not in a dead end choose the direction of the next step
                next_step = random.choice(directions)
                row = row + self.pos[next_step]['r']
                column = column + self.pos[next_step]['c']
            else:
                # when in a dead end save current path end trace back
                if len(self.path_steps) > 1:
                    # a path has at least two way points
                    self.paths.append(self.path_steps)
                self.path_steps = []
                if self.breadcrumbs:
                    self.breadcrumbs.pop() # go back one step
                    if len(self.breadcrumbs) > 0:
                        # use current position as starting point for new path
                        row, column = self.breadcrumbs.pop()
                    else:
                        # The maze is complete when there are no more way points in the 
                        # breadcrumbs list. All possible paths have been created.
                        return

    def set_way_point(self, row, column):
        self.breadcrumbs.append([row, column])
        self.path_steps.append([row, column])
        cell = self.maze.loc[row, column]
        cell['isWayPoint'] = True
        self.maze.set_value(row, column, cell)
        return cell

    def get_possible_directions(self, cell):
        directions = self.get_directions(cell)
        directions = self.avoid_outside_maze(directions, cell)
        directions = self.avoid_set_way_points(directions, cell)
        return directions

    def get_directions(self, cell):
        directions = [w for w in cell['walls']]
        return directions

    def avoid_walls(self, cell):
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
    
    def set_walls(self):
        for path in self.paths:
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

    def save_maze_image(self, dpmm):
        img_width = int(self.maze_dimensions['columns'] * cell_size * dpmm)
        img_height = int(self.maze_dimensions['rows'] * cell_size * dpmm)
        img = np.ones((img_height,img_width,3), np.uint8)*255
        # draw all walls
        if True:
            for index, row in self.maze.iterrows():
                for cell in row:
                    self.draw_walls(img, cell, (0,0,0), dpmm)
        # draw all paths
        if False:
            for path in self.paths:
                for i, _ in enumerate(path[:-1]):
                    self.draw_line(img, path[i], path[i+1], (0,0,255), dpmm)
        # draw the path through the maze
        if True:
            for i, _ in enumerate(self.path_through_maze[:-1]):
                self.draw_line(img,
                               self.path_through_maze[i],
                               self.path_through_maze[i+1],
                               (255,0,0),
                               dpmm)
        cv2.imwrite('maze.png', img)

    def draw_walls(self, img, cell, colour, dpmm):
        row = cell['position']['row']
        column = cell['position']['column']
        if cell['walls']['up']:
            startx = int((column * cell_size) * dpmm)
            starty = int((row * cell_size) * dpmm)
            endx   = int((column * cell_size + cell_size) * dpmm)
            endy   = starty
            img = cv2.line(img, (startx, starty), (endx, endy), colour, 3)
        if cell['walls']['down']:
            startx = int((column * cell_size) * dpmm)
            starty = int((row * cell_size + cell_size) * dpmm)
            endx   = int((column * cell_size + cell_size) * dpmm)
            endy   = starty
            img = cv2.line(img, (startx, starty), (endx, endy), colour, 3)
        if cell['walls']['right']:
            startx = int((column * cell_size + cell_size) * dpmm)
            starty = int((row * cell_size) * dpmm)
            endx   = startx
            endy   = int((row * cell_size + cell_size) * dpmm)
            img = cv2.line(img, (startx, starty), (endx, endy), colour, 3)
        if cell['walls']['left']:
            startx = int((column * cell_size) * dpmm)
            starty = int((row * cell_size) * dpmm)
            endx   = startx
            endy   = int((row * cell_size + cell_size) * dpmm)
            img = cv2.line(img, (startx, starty), (endx, endy), colour, 3)

    def draw_line(self, img, start_cell, end_cell, colour, dpmm):
        startx = int((start_cell[1] * cell_size + cell_size/2) * dpmm)
        starty = int((start_cell[0] * cell_size + cell_size/2) * dpmm)
        endx   = int((end_cell[1] * cell_size + cell_size/2) * dpmm)
        endy   = int((end_cell[0] * cell_size + cell_size/2) * dpmm)
        img = cv2.line(img, (startx, starty), (endx, endy), colour, 1)

# ------------------- main -----------------------------------------------------

if __name__ == "__main__":
    main()
