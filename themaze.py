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
    max_steps = 2000
    maze = Maze(maze_dimensions, maze_entry, maze_exit, max_steps)
    paths = maze.create_maze(row = maze_entry['row'],
                             column = maze_entry['column'])

    # create image, white background
    save_maze_image(paths, rows, columns, dpmm)
    print('Done!')

class Maze:
    
    def __init__(self, maze_dimensions, maze_entry, maze_exit, max_steps):
        # deltas for each step direction
        self.pos = {'up': {'c': 0, 'r': -1},
                    'right': {'c': 1, 'r': 0}, 
                    'left': {'c': -1, 'r': 0}, 
                    'down': {'c': 0, 'r': 1}}
        self.maze_dimensions = maze_dimensions
        self.maze_entry = maze_entry
        self.maze_exit = maze_exit
        self.step_count = 0
        self.max_steps = max_steps
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
    def create_maze(self, row = 0, column = 0):
        self.create_paths(row, column)
        self.set_walls()
        return self.paths

    def create_paths(self, row = 0, column = 0):
        cell = self.set_way_point(row, column)
        if cell['position'] == self.maze_exit:
            # We found the exit! Save the path through the maze.
            self.path_through_maze = self.breadcrumbs
        directions = self.get_possible_directions(cell)
        if directions:
            # when not in a dead end choose the direction of the next step
            next_step = random.choice(directions)
            self.step_count = self.step_count + 1
            if self.step_count < self.max_steps:
                self.create_paths(row + self.pos[next_step]['r'], 
                                  column + self.pos[next_step]['c'])
            else:
                # save last path
                self.paths.append(self.path_steps)
        else:
            # when in a dead end save current path end trace back
            if len(self.path_steps) > 1:
                # a proper path has at least two steps
                self.paths.append(self.path_steps)
            self.path_steps = []
            if self.breadcrumbs:
                self.breadcrumbs.pop() # go back one step
                if len(self.breadcrumbs) > 0:
                    # use current position as starting point for new path
                    row, column = self.breadcrumbs.pop()
                    self.create_paths(row, column)
                # The maze is complte when there are no more way points in the 
                # breadcrumbs list. All possible paths have been created. 

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
        pass

def save_maze_image(paths, rows, columns, dpmm):
    img_width = int(columns * cell_size * dpmm)
    img_height = int(rows * cell_size * dpmm)
    img = np.ones((img_height,img_width,3), np.uint8)*255
    # draw the create_paths path
    for path in paths:
        for i, _ in enumerate(path[:-1]):
            start_cell = path[i]
            end_cell = path[i+1]
            startx = int((start_cell[1] * cell_size + cell_size/2) * dpmm)
            starty = int((start_cell[0] * cell_size + cell_size/2) * dpmm)
            endx   = int((end_cell[1] * cell_size + cell_size/2) * dpmm)
            endy   = int((end_cell[0] * cell_size + cell_size/2) * dpmm)
            img = cv2.line(img,(startx, starty),(endx, endy),(0,0,255),1)
     
    # Draw a diagonal lines with thickness of 1 px
    img = cv2.line(img,(0,0),(img_width,img_height),(0,0,0),1)
    img = cv2.line(img,(img_width,0),(0,img_height),(0,0,0),1)
    # Draw border lines with thickness of 1 px
    img = cv2.line(img,(0,0),(img_width,0),(0,0,0),1)
    img = cv2.line(img,(0,0),(0,img_height),(0,0,0),1)
    img = cv2.line(img,(img_width,0),(img_width,img_height),(0,0,0),1)
    img = cv2.line(img,(0,img_height),(img_width,img_height),(0,0,0),1)
    cv2.imwrite('maze.png',img)

# ------------------- main -----------------------------------------------------

if __name__ == "__main__":
    main()
