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
    """
    A* 算法的节点类，用于存储路径信息。
    """
    def __init__(self, parent_node=None, position=None):
        self.parent_node = parent_node
        self.position = position
        self.cost_from_start = 0  # g: 从起点到当前节点的代价
        self.cost_to_goal = 0     # h: 启发式函数，从当前节点到目标的估计代价
        self.total_cost = 0       # f = g + h

    def __eq__(self, other):
        return self.position == other.position
    
    def __lt__(self, other):
        return self.total_cost < other.total_cost

def calculate_distance_map(world_map):
    """
    计算地图上每个点到最近障碍物的距离。
    """
    map_height, map_width = world_map.shape
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
                if dr == 0 and dc == 0: continue
                
                nr, nc = r + dr, c + dc

                if 0 <= nr < map_height and 0 <= nc < map_width:
                    # 使用欧几里得距离传播，结果更精确
                    new_dist = distance_map[r, c] + np.sqrt(dr**2 + dc**2)
                    if distance_map[nr, nc] > new_dist:
                        distance_map[nr, nc] = new_dist
                        queue.append((nr, nc))
    return distance_map

###  END CODE HERE  ###


def Self_driving_path_planner(world_map, start_pos, goal_pos):
    """
    给定世界地图、机器人起始位置和目标位置，
    规划一条从起点到目标的、更优化的平滑路径。

    Arguments:
    world_map -- 一个 120*120 的数组，表示当前地图，0表示可通过，1表示障碍物。
    start_pos -- 一个 2D 向量，表示机器人的当前位置。
    goal_pos -- 一个 2D 向量，表示目标位置。

    Return:
    path -- 一个 N*2 的数组，表示规划出的平滑路径。
    """

    ### START CODE HERE ###

    # --- 步骤 1: 使用 A* 算法找到一条初始路径 ---
    start_node = Node(None, tuple(start_pos))
    goal_node = Node(None, tuple(goal_pos))
    open_list, closed_set = [], set()
    heapq.heappush(open_list, (start_node.total_cost, start_node))
    initial_path = []

    while open_list:
        _, current_node = heapq.heappop(open_list)
        if current_node.position in closed_set: continue
        closed_set.add(current_node.position)
        
        if current_node == goal_node:
            path_segment = []
            current = current_node
            while current is not None:
                path_segment.append(list(current.position))
                current = current.parent_node
            initial_path = path_segment[::-1]
            break

        possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for move in possible_moves:
            neighbor_pos = (current_node.position[0] + move[0], current_node.position[1] + move[1])
            map_height, map_width = world_map.shape
            if not (0 <= neighbor_pos[0] < map_height and 0 <= neighbor_pos[1] < map_width) or \
               world_map[int(neighbor_pos[0])][int(neighbor_pos[1])] != 0 or \
               neighbor_pos in closed_set:
                continue
            
            neighbor_node = Node(current_node, neighbor_pos)
            move_cost = 1.414 if abs(move[0]) + abs(move[1]) == 2 else 1.0
            neighbor_node.cost_from_start = current_node.cost_from_start + move_cost
            dx = neighbor_node.position[0] - goal_node.position[0]
            dy = neighbor_node.position[1] - goal_node.position[1]
            neighbor_node.cost_to_goal = np.sqrt(dx**2 + dy**2)
            neighbor_node.total_cost = neighbor_node.cost_from_start + neighbor_node.cost_to_goal
            heapq.heappush(open_list, (neighbor_node.total_cost, neighbor_node))

    if not initial_path:
        print("Path not found by A*!")
        return []

    # --- 步骤 2: 对初始路径进行平滑处理 (优化版) ---
    
    # 平滑参数 (可调节)
    alpha = 0.15  # 平滑权重
    beta = 0.2    # 障碍物排斥权重
    influence_radius = 5.0 # <--- 优化点 1: 障碍物排斥力的影响半径
    iterations = 100 # 迭代次数

    smoothed_path = np.array(initial_path, dtype=float)
    dist_map = calculate_distance_map(world_map)
    map_height, map_width = world_map.shape

    for _ in range(iterations):
        for i in range(1, len(smoothed_path) - 1):
            p_curr = smoothed_path[i]

            # 1. 计算平滑力
            smoothing_force = alpha * (smoothed_path[i - 1] + smoothed_path[i + 1] - 2 * p_curr)

            # 2. 计算障碍物排斥力 (优化版)
            obstacle_force = np.zeros(2)
            x, y = int(p_curr[0]), int(p_curr[1])
            
            # 确保坐标在距离图内
            if 0 <= x < map_height and 0 <= y < map_width:
                dist = dist_map[x, y]
                
                # <--- 优化点 2: 只有在影响半径内才计算排斥力
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
            
            # 更新点的位置
            new_pos = p_curr + smoothing_force + obstacle_force
            
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