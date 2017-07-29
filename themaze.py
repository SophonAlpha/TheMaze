from __future__ import division
import numpy as np
import pandas as pd
import random
import cv2

print_size = 'A4'
cell_size = 20 # cell size in mm

# deltas for each step direction
pos = {'top': {'x': 0, 'y': -1},
       'right': {'x': 1, 'y': 0}, 
       'left': {'x': -1, 'y': 0}, 
       'bottom': {'x': 0, 'y': 1}}

def walk(row = 0, column = 0, step_count = 0, max_steps = 10):
    # mark current cell as way point
    cell = maze.loc[row, column]
    cell['onpath'] = True
    maze.set_value(row, column, cell)
    # get all directions where there is no wall
    directions = [w for w in cell['walls'] if cell['walls'][w] == False]
    # remove directions that would lead outside the maze
    directions = [d for d in directions
                  if row+pos[d]['y'] in maze.index and column+pos[d]['x'] in maze.columns]
    # remove directions that would lead to a cell where we have been before
    directions = [d for d in directions 
                  if maze.loc[row+pos[d]['y'], column+pos[d]['x']]['onpath'] == True]
    #choose the direction of the next step
    next_step = random.choice(directions)
    # make the next step
    step_count = step_count + 1
    if step_count < max_steps:
        walk(maze, row + pos[next_step]['y'], 
             column + pos[next_step]['x'], 
             step_count, 
             max_steps)
    # TODO: add code to detect if walker got stuck
    return

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
dpi = 300 # print quality in dots per inch
inch = 25.4 # inch to mm
dpmm = dpi/inch # dots per mm
cells_width = int(paper_sizes[print_size]['width']/cell_size)
cells_height = int(paper_sizes[print_size]['height']/cell_size)
img_width = int(cells_width * cell_size * dpmm)
img_height = int(cells_height * cell_size * dpmm)

# create empty maze
cell = {'walls': {'top': False,'bottom': False, 'right': False, 'left': False},
        'onpath': False}
maze = pd.DataFrame(index = range(cells_height),
                    columns = range(cells_width))
for w in range(maze.shape[1]):
    for h in range(maze.shape[0]):
        maze.set_value(h, w, cell)

walk(row = 0, column = 0)

# create image, white background
img = np.ones((img_height,img_width,3), np.uint8)*255

# draw the walk path

for r in maze.index:
    for c in maze.columns:
        neighbours = [ [r+pos[p]['y'], c+pos[p]['x']] for p in pos # TODO: refactor to function 
                      if c+pos[p]['x'] in maze.columns and r+pos[p]['y'] in maze.index]
        lines = [n for n in neighbours if maze.loc[n[0], n[1]]['onpath'] == True]
        for l in lines:
            startx = int((c * cell_size + cell_size/2) * dpmm)
            starty = int((r * cell_size + cell_size/2) * dpmm)
            endx   = int((l[1] * cell_size + cell_size/2) * dpmm)
            endy   = int((l[0] * cell_size + cell_size/2) * dpmm)
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
print('Done!')

