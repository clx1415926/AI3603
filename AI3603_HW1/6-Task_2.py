import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import heapq
from collections import deque 

MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '3-map/map.npy')


### START CODE HERE ###
# This code block is optional. You can define your utility function and class in this block if necessary.

class Node:
    def __init__(self, parent_node=None, position=None):
        self.parent_node = parent_node
        self.position = position
        self.g = 0  
        self.h = 0     
        self.total_cost = 0       

    def __eq__(self, other):
        return self.position == other.position
    
    def __lt__(self, other):
        if self.total_cost == other.total_cost:
            return self.h < other.h
        return self.total_cost < other.total_cost

#计算地图上每个点离障碍物有多远
def distance(world_map):
    map_height, map_width = world_map.shape

    # 初始化距离图，所有非障碍物点距离为无穷大
    dist_map = np.full(world_map.shape, float('inf'))
    queue = deque()

    for r in range(map_height):
        for c in range(map_width):
            if world_map[r][c] == 1:
                dist_map[r, c] = 0
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                nr, nc = r + dr, c + dc

                if 0 <= nr < map_height and 0 <= nc < map_width:
                    if dist_map[nr, nc] > dist_map[r, c] + 1:
                        dist_map[nr, nc] = dist_map[r, c] + 1
                        queue.append((nr, nc))
    return dist_map

###  END CODE HERE  ###


def Improved_A_star(world_map, start_pos, goal_pos):
    """
    Given map of the world, start position of the robot and the position of the goal, 
    plan a path from start position to the goal using A* algorithm.

    Arguments:
    world_map -- A 120*120 array indicating current map, where 0 indicating traversable and 1 indicating obstacles.
    start_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    path -- A N*2 array representing the planned path by A* algorithm.
    """
    ### START CODE HERE ###
    start_point = Node(None, tuple(start_pos))
    goal_point = Node(None, tuple(goal_pos))
    path = []
    nodes, explored = [], set()
    dist_map = distance(world_map)
    
    #两个惩罚权重，避障和转向
    OBSTACLE_WEIGHT = 10.0  
    STEERING_WEIGHT = 0.8   
    heapq.heappush(nodes, (start_point.total_cost, start_point))

    
    while nodes:
        _, current_point = heapq.heappop(nodes)
        
        if current_point.position in explored:
            continue
        explored.add(current_point.position)
        
        if current_point == goal_point:
            final_path = []
            while current_point is not None:
                final_path.append(list(current_point.position))
                current_point = current_point.parent_node
            path = final_path[::-1]
            break

        possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for move in possible_moves:
            neighbour_pos = (current_point.position[0] + move[0], current_point.position[1] + move[1])

            map_height, map_width = world_map.shape
            if not (0 <= neighbour_pos[0] < map_height and 0 <= neighbour_pos[1] < map_width) or world_map[neighbour_pos[0]][neighbour_pos[1]] != 0 or neighbour_pos in explored:
                continue
            
            neighbour_node = Node(current_point, neighbour_pos)
            
            move_cost = 1.0 if abs(move[0]) + abs(move[1]) == 1 else 1.414
            
            steering_cost = 0
            if current_point.parent_node is not None:
                prev_move = (current_point.position[0] - current_point.parent_node.position[0],
                             current_point.position[1] - current_point.parent_node.position[1])
                if move != prev_move:
                    steering_cost = STEERING_WEIGHT
            
            neighbour_node.g = current_point.g + move_cost + steering_cost

            #和task1不同的移动方式，导致启发函数不同
            dx = abs(neighbour_node.position[0] - goal_point.position[0])
            dy = abs(neighbour_node.position[1] - goal_point.position[1])
            neighbour_node.h = np.sqrt(dx**2 + dy**2)
            
            dist_to_obstacle = dist_map[neighbour_pos[0], neighbour_pos[1]]
            obstacle_cost = 0

            if dist_to_obstacle <= 3 and dist_to_obstacle > 0:
                obstacle_cost = OBSTACLE_WEIGHT / dist_to_obstacle

            neighbour_node.total_cost = (neighbour_node.g + 
                                         neighbour_node.h + 
                                         obstacle_cost)
            
            heapq.heappush(nodes, (neighbour_node.total_cost, neighbour_node))

    if not path:
        print("Path not found!")
        return []

    ###  END CODE HERE  ###
    return path


if __name__ == '__main__':

    # Get the map of the world representing in a 120*120 array, where 0 indicating traversable and 1 indicating obstacles.
    map = np.load(MAP_PATH)

    # Define goal position of the exploration
    goal_pos = [100, 100]

    # Define start position of the robot.
    start_pos = [10, 10]

    # Plan a path based on map from start position of the robot to the goal.
    path = Improved_A_star(map, start_pos, goal_pos)

    # Visualize the map and path.
    obstacles_x, obstacles_y = [], []
    for i in range(120):
        for j in range(120):
            if map[i][j] == 1:
                obstacles_x.append(i)
                obstacles_y.append(j)

    path_x, path_y = [], []
    for path_node in path:
        path_x.append(path_node[0])
        path_y.append(path_node[1])

    plt.plot(path_x, path_y, "-r")
    plt.plot(start_pos[0], start_pos[1], "xr")
    plt.plot(goal_pos[0], goal_pos[1], "xb")
    plt.plot(obstacles_x, obstacles_y, ".k")
    plt.grid(True)
    plt.axis("equal")
    plt.show()