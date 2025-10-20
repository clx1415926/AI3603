import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import heapq

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
        #如果总成本相同，则优先选择启发式成本更低的,使其更快到达终点（似乎加了这两行会减少锯齿，使得路径更平滑，但同时也会变慢）
        if self.total_cost == other.total_cost:
            return self.h < other.h
        return self.total_cost < other.total_cost

###  END CODE HERE  ###


def A_star(world_map, start_pos, goal_pos):
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
    
    # 最小堆实现将被探索的节点，以方便拿到最小cost的节点
    nodes = []
    heapq.heappush(nodes, (start_point.total_cost, start_point))
    
    explored = set()
    
    while nodes:
        _, current_point = heapq.heappop(nodes)
        
        if current_point.position in explored:
            continue
        
        if current_point == goal_point:
            final_path = []
            while current_point is not None:
                final_path.append(list(current_point.position))
                current_point = current_point.parent_node
            path = final_path[::-1]
            break


        possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for move in possible_moves:
            
            neighbour = (current_point.position[0] + move[0],current_point.position[1] + move[1])

            if neighbour in explored:
                continue

            #检查坐标是否越界
            map_height, map_width = world_map.shape
            if not (0 <= neighbour[0] < map_height and 0 <= neighbour[1] < map_width):
                continue
            
            #检查该位置是否为障碍物
            if world_map[neighbour[0]][neighbour[1]] != 0:
                continue
            
            neighbour_node = Node(current_point, neighbour)
            neighbour_node.g = current_point.g + 1
            neighbour_node.h = (abs(neighbour_node.position[0] - goal_point.position[0]) + abs(neighbour_node.position[1] - goal_point.position[1]))
            neighbour_node.total_cost = neighbour_node.g + neighbour_node.h
            heapq.heappush(nodes, (neighbour_node.total_cost, neighbour_node))

            explored.add(current_point.position)

    if path == []:
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
    path = A_star(map, start_pos, goal_pos)

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