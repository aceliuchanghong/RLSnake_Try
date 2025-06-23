import numpy as np
import random
import os
import time
from termcolor import colored


class DynamicVRPEnv:
    def __init__(
        self, num_customers=8, num_depots=2, vehicle_capacities=[5, 8], max_trips=5
    ):
        """
        初始化动态车辆路径问题环境
        :param num_customers: 客户数量
        :param num_depots: 仓库数量
        :param vehicle_capacities: 一个包含每辆车容量的列表
        :param max_trips: 每辆车的最大回家次数
        """
        self.num_depots = num_depots
        self.num_customers = num_customers
        self.num_nodes = num_depots + num_customers
        self.num_vehicles = len(vehicle_capacities)
        self.vehicle_capacities = np.array(vehicle_capacities)
        self.max_trips = max_trips
        self.max_demand = 3  # 假设客户最大需求为3

        # 动作空间大小: 每个车辆都可以选择去任何一个节点
        # action = vehicle_id * num_nodes + node_id
        self.action_space_size = self.num_vehicles * self.num_nodes

        # 状态空间维度
        # state: [veh_pos(one-hot), veh_capacity(norm), veh_trips(norm), order_demands(norm)]
        state_dim = (
            (self.num_vehicles * self.num_nodes)
            + self.num_vehicles
            + self.num_vehicles
            + self.num_customers
        )
        self.state_space_dim = state_dim

        self.coords = None
        self.distances = None
        self.reset()

    def reset(self):
        """
        重置环境到初始状态
        """
        # 1. 初始化节点坐标 (前 num_depots 个为仓库)
        self.coords = np.random.rand(self.num_nodes, 2) * 100
        # 计算距离矩阵
        self.distances = np.linalg.norm(
            self.coords[:, np.newaxis] - self.coords, axis=2
        )

        # 2. 初始化车辆状态
        # 每辆车从不同的仓库出发，如果仓库比车多，则从前几个仓库出发
        self.vehicle_positions = np.arange(min(self.num_vehicles, self.num_depots))
        self.vehicle_loads = np.zeros(self.num_vehicles)  # 当前载货量
        self.vehicle_trips = np.zeros(self.num_vehicles, dtype=int)  # 已完成的行程数
        self.vehicle_done = np.zeros(
            self.num_vehicles, dtype=bool
        )  # 车辆是否已完成任务回家

        # 3. 初始化订单 (随机生成3个初始订单)
        self.orders = {}
        initial_customers = random.sample(range(self.num_depots, self.num_nodes), 3)
        for customer_node in initial_customers:
            self.orders[customer_node] = random.randint(1, self.max_demand)

        self.served_customers = set()

        return self.get_state()

    def _decode_action(self, action):
        """将扁平化的动作值解码为 (vehicle_id, node_id)"""
        vehicle_id = action // self.num_nodes
        node_id = action % self.num_nodes
        return vehicle_id, node_id

    def get_state(self):
        """
        获取当前状态的向量表示
        """
        # 车辆位置 (one-hot, 扁平化)
        pos_state = np.zeros(self.num_vehicles * self.num_nodes, dtype=np.float32)
        for i in range(self.num_vehicles):
            pos_state[i * self.num_nodes + self.vehicle_positions[i]] = 1.0

        # 车辆剩余容量 (归一化)
        remaining_capacity = self.vehicle_capacities - self.vehicle_loads
        cap_state = remaining_capacity / self.vehicle_capacities

        # 车辆已完成行程 (归一化)
        trips_state = self.vehicle_trips / self.max_trips

        # 订单需求 (归一化)
        demand_state = np.zeros(self.num_customers, dtype=np.float32)
        for node, demand in self.orders.items():
            customer_idx = node - self.num_depots
            demand_state[customer_idx] = demand / self.max_demand

        # 拼接所有状态
        state = np.concatenate([pos_state, cap_state, trips_state, demand_state])
        return state

    def step(self, action):
        """
        执行一步动作，返回 next_state, reward, done, info
        """
        vehicle_id, target_node = self._decode_action(action)
        current_pos = self.vehicle_positions[vehicle_id]

        # --- 1. 检查动作合法性 ---
        # a) 车辆是否已经完成任务
        if self.vehicle_done[vehicle_id]:
            reward = -100  # 对已完成的车辆下达指令是无效的
            done = all(self.vehicle_done)
            return self.get_state(), reward, done, {}

        # b) 访问客户，但容量不足
        is_customer = target_node >= self.num_depots
        if is_customer:
            demand = self.orders.get(target_node, 0)
            if demand == 0:  # 尝试访问一个没有订单的客户
                reward = -100
                return self.get_state(), reward, False, {}
            if (
                self.vehicle_loads[vehicle_id] + demand
                > self.vehicle_capacities[vehicle_id]
            ):
                reward = -100  # 容量不足，无效动作
                return self.get_state(), reward, False, {}

        # --- 2. 更新状态 ---
        distance = self.distances[current_pos, target_node]
        reward = -distance  # 主要奖励：负距离

        self.vehicle_positions[vehicle_id] = target_node

        # 如果访问的是客户
        if is_customer:
            demand = self.orders.pop(target_node)
            self.vehicle_loads[vehicle_id] += demand
            self.served_customers.add(target_node)

        # 如果访问的是仓库 (回家)
        elif target_node < self.num_depots:
            self.vehicle_loads[vehicle_id] = 0  # 卸货并重新补给
            self.vehicle_trips[vehicle_id] += 1

        # --- 3. 随机生成新订单 (模拟动态性) ---
        # 确保总有订单可服务，除非所有客户都已被服务过
        if (
            random.random() < 0.2
            and len(self.orders) + len(self.served_customers) < self.num_nodes
        ):
            available_nodes = [
                i
                for i in range(self.num_depots, self.num_nodes)
                if i not in self.orders and i not in self.served_customers
            ]
            if available_nodes:
                new_node = random.choice(available_nodes)
                self.orders[new_node] = random.randint(1, self.max_demand)

        # --- 4. 检查终止条件 ---
        # a) 检查是否有车辆因为达到最大行程而结束任务
        if (
            self.vehicle_trips[vehicle_id] >= self.max_trips
            and self.vehicle_positions[vehicle_id] < self.num_depots
        ):
            self.vehicle_done[vehicle_id] = True

        # b) 判断整个episode是否结束
        done = False
        all_orders_done = not self.orders
        if all_orders_done:
            # 如果所有订单完成，检查是否所有车都回家了
            all_vehicles_at_depot = all(
                pos < self.num_depots for pos in self.vehicle_positions
            )
            if all_vehicles_at_depot:
                done = True
                reward += 200  # 完成所有任务的巨大奖励

        if all(self.vehicle_done):
            done = True  # 所有车都达到最大行程数回家
            if not all_orders_done:
                reward -= 100  # 如果还有订单没送完，则给予惩罚

        return self.get_state(), reward, done, {}

    ### 3. 参考贪吃蛇代码的render,帮我也加一下
    def render(self, mode="console"):
        """
        在命令行中可视化当前环境状态
        """
        os.system("clear" if os.name == "posix" else "cls")

        # 创建一个20x40的字符网格
        grid_height = 20
        grid_width = 40
        grid = [["." for _ in range(grid_width)] for _ in range(grid_height)]

        # 将连续坐标映射到离散网格
        scale_x = grid_width / 100
        scale_y = grid_height / 100

        def to_grid_coords(node_idx):
            x = int(self.coords[node_idx, 0] * scale_x)
            y = int(self.coords[node_idx, 1] * scale_y)
            return min(x, grid_width - 1), min(y, grid_height - 1)

        # 标记仓库
        for i in range(self.num_depots):
            gx, gy = to_grid_coords(i)
            grid[gy][gx] = colored(f"D{i}", "cyan", attrs=["bold"])

        # 标记有订单的客户
        for node in self.orders:
            gx, gy = to_grid_coords(node)
            customer_id = node - self.num_depots
            grid[gy][gx] = colored(f"C{customer_id}", "red")

        # 标记车辆
        vehicle_colors = ["yellow", "green", "magenta", "blue"]
        for i in range(self.num_vehicles):
            gx, gy = to_grid_coords(self.vehicle_positions[i])
            color = vehicle_colors[i % len(vehicle_colors)]
            if self.vehicle_done[i]:
                grid[gy][gx] = colored(f"V{i}", color, attrs=["dark"])
            else:
                grid[gy][gx] = colored(f"V{i}", color, attrs=["bold"])

        # 打印网格
        print("--- Dynamic VRP State ---")
        for row in grid:
            print(" ".join(row))
        print("-" * (grid_width * 2))

        # 打印状态信息
        print(colored("--- Status ---", "cyan"))
        for i in range(self.num_vehicles):
            color = vehicle_colors[i % len(vehicle_colors)]
            status = "DONE" if self.vehicle_done[i] else "ACTIVE"
            info = (
                f"Vehicle {i} ({status}): Pos: {self.vehicle_positions[i]}, "
                f"Load: {self.vehicle_loads[i]}/{self.vehicle_capacities[i]}, "
                f"Trips: {self.vehicle_trips[i]}/{self.max_trips}"
            )
            print(colored(info, color))

        order_info = "Active Orders (Node:Demand): " + str(self.orders)
        print(colored(order_info, "red"))
        print("-" * (grid_width * 2))

        time.sleep(0.1)


if __name__ == "__main__":
    # python snake/VehicleGameMain.py
    env = DynamicVRPEnv(vehicle_capacities=[5, 8])
    done = False
    total_reward = 0
    state = env.reset()
    env.render()

    max_steps = 300
    for step in range(max_steps):
        # 随机选择一个动作
        action = random.randint(0, env.action_space_size - 1)

        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        env.render()
        print(
            f"Step: {step+1}, Action: {env._decode_action(action)}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}"
        )

        if done:
            print("\n--- Episode Finished! ---")
            break

    if not done:
        print("\n--- Max steps reached. Episode terminated. ---")
