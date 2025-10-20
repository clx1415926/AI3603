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
    distance_map = np.full(world_map.shape, float('inf'))
    queue = deque()

    for r in range(map_height):
        for c in range(map_width):
            if world_map[r][c] == 1:
                distance_map[r, c] = 0
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                nr, nc = r + dr, c + dc

                if 0 <= nr < map_height and 0 <= nc < map_width:
                    if distance_map[nr, nc] > distance_map[r, c] + 1:
                        distance_map[nr, nc] = distance_map[r, c] + 1
                        queue.append((nr, nc))
    return distance_map

###  END CODE HERE  ###


def Self_driving_path_planner(world_map, start_pos, goal_pos):
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
            neighbor_pos = (current_point.position[0] + move[0], current_point.position[1] + move[1])
            
            map_height, map_width = world_map.shape
            if not (0 <= neighbor_pos[0] < map_height and 0 <= neighbor_pos[1] < map_width) or world_map[int(neighbor_pos[0])][int(neighbor_pos[1])] != 0 or neighbor_pos in explored:
                continue
            
            neighbor_node = Node(current_point, neighbor_pos)

            move_cost = 1.0 if abs(move[0]) + abs(move[1]) == 1 else 1.414
            neighbor_node.g = current_point.g + move_cost

            dx = neighbor_node.position[0] - goal_point.position[0]
            dy = neighbor_node.position[1] - goal_point.position[1]
            neighbor_node.h = np.sqrt(dx**2 + dy**2)

            neighbor_node.total_cost = neighbor_node.g + neighbor_node.h

            heapq.heappush(nodes, (neighbor_node.total_cost, neighbor_node))

    if not path:
        print("Path not found by A*!")
        return []

    # 参数 (可调节)
    alpha = 0.15  # 平滑权重
    beta = 0.2    # 障碍物排斥权重
    influence_radius = 5.0 # 障碍物排斥力的影响半径
    iterations = 100 # 迭代次数

    smoothed_path = np.array(path, dtype=float)
    map_height, map_width = world_map.shape
    dist_map = distance(world_map)

    for _ in range(iterations):
        for i in range(1, len(smoothed_path) - 1):
            current_point = smoothed_path[i]

            # 计算平滑力
            smoothing_force = alpha * (smoothed_path[i - 1] + smoothed_path[i + 1] - 2 * current_point)

            # 计算障碍物排斥力
            obstacle_force = np.zeros(2)
            x, y = int(current_point[0]), int(current_point[1])
            
            if 0 <= x < map_height and 0 <= y < map_width:
                dist = dist_map[x, y]
                
                if dist < influence_radius:
                    # 使用有限差分法近似距离场的梯度
                    if 0 < x < map_height - 1 and 0 < y < map_width - 1:
                        grad_x = dist_map[x + 1, y] - dist_map[x - 1, y]
                        grad_y = dist_map[x, y + 1] - dist_map[x, y - 1]
                        grad = np.array([grad_x, grad_y])
                        
                        grad_norm = np.linalg.norm(grad)
                        if grad_norm > 1e-6:
                            # 力的方向是梯度方向，大小与beta和距离成反比
                            # 离障碍物越近，(influence_radius - dist)越大，力也越大
                            force_magnitude = beta * (influence_radius - dist) / influence_radius
                            obstacle_force = force_magnitude * (grad / grad_norm)
            
            new_pos = current_point + smoothing_force + obstacle_force
            
            # 碰撞检测
            new_x, new_y = int(new_pos[0]), int(new_pos[1])
            if not (0 <= new_x < map_height and 0 <= new_y < map_width and world_map[new_x, new_y] == 1):
                smoothed_path[i] = new_pos

    path = smoothed_path.tolist()

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
    path = Self_driving_path_planner(map, start_pos, goal_pos)

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