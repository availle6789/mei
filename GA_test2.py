# -*- coding: utf-8 -*-
"""
Created on Sun May 19 11:51:51 2024

@author: pc
"""

import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt

# Network Parameters
n = 84
demand = 50000
theta = 0.1
walk_link_cap = 9999999999  # Very large capacity for the walking link

class path_class:
    def __init__(self):
        self.flow = 0
        self.cost = 0
        self.links = []
        self.logit_prob = 0
        self.prob = 0

    def get_cost(self, _graph: nx.DiGraph):
        self.cost = sum(_graph.edges[e]['t0'] for e in self.links)

class my_network_class:
    def __init__(self, _graph):
        self.graph = _graph
        self.paths = []
        self.cycles = {}
        self.phase_times = {}

    def reset_edge_flows(self):
        """重置每个边缘的流量。"""
        for e in self.graph.edges:
            self.graph.edges[e]['xa'] = 0

    def update_edge_flow(self):
        """更新每个边缘的流量。"""
        self.reset_edge_flows()  # 在更新前重置流量
        for p in self.paths:
            for link in p.links:
                self.graph.edges[link]['xa'] += p.flow
        self.verify_edge_flows()  # 验证流量更新正确性

    def update_edge_cost(self, green_times=None):
        """更新每个边缘的成本。"""
        for e in self.graph.edges(data=True):
            if e[2]['type'] == 'BPR':
                cap = max(e[2]['cap'], 1)  # 避免除以0
                x = e[2]['xa'] / cap
                alpha, beta = 0.15, 4
                t0 = e[2]['t0']
                e[2]['t0'] = t0 * (1 + alpha * (x ** beta))

        # 信号控制交叉口的特殊处理
        for intersection_id in set(e[2].get('intersection_id') for e in self.graph.edges(data=True) if 'intersection_id' in e[2] and e[2]['type'] == 'signal'):
            intersection_edges = [(e[0], e[1], e[2]) for e in self.graph.edges(data=True) if e[2].get('intersection_id') == intersection_id]
            L = 10  # 损失时间
            total_intersection_flow = sum(e[2]['xa'] for e in intersection_edges)
            cycle = max(1.5 * L + 5 + 3600 / (total_intersection_flow / len(intersection_edges) if total_intersection_flow > 0 else 1), 90)  # 确保最小周期时间

            if green_times is not None:
                phases = [phase_id for phase_id in set(e[2]['phase_id'] for e in intersection_edges if 'phase_id' in e[2])]
                self.phase_times[intersection_id] = {phase_id: green_times[i] for i, phase_id in enumerate(phases)}
                cycle = sum(self.phase_times[intersection_id].values()) + L
            else:
                phase_times = {}
                for phase_id in set(e[2]['phase_id'] for e in intersection_edges if 'phase_id' in e[2]):
                    phase_edges = [e for e in intersection_edges if e[2]['phase_id'] == phase_id]
                    phase_flow = sum(e[2]['xa'] for e in phase_edges)
                    green_time = max(20, min(70, cycle * (phase_flow / total_intersection_flow if total_intersection_flow > 0 else 1)))
                    phase_times[phase_id] = green_time

                self.phase_times[intersection_id] = phase_times

            self.cycles[intersection_id] = cycle  # 存储周期时间

            for phase_id, green_time in self.phase_times[intersection_id].items():
                for e in [e for e in intersection_edges if e[2].get('phase_id') == phase_id]:
                    e[2]['green_time'] = round(green_time, 2)
                    cap = e[2]['cap'] if e[2]['cap'] > 0 else 1
                    x = e[2]['xa'] / cap
                    delay = (cycle * (1 - (green_time / cycle)) ** 2) / (2 * (1 - (green_time / cycle) * x)) + \
                            (x ** 2) / (2 * cap) / (1 - x) * e[2]['xa']
                    e[2]['t0'] = delay / 3600

    def update_path_cost(self, green_times=None):
        """更新每条路径的成本。"""
        self.update_edge_cost(green_times)
        for p in self.paths:
            p.get_cost(self.graph)

    def update_path_prob(self):
        """更新路径的选择概率。"""
        path_exp = [math.exp(-theta * p.cost) for p in self.paths]
        sum_path_exp = sum(path_exp)
        if sum_path_exp > 0:
            for p, exp_val in zip(self.paths, path_exp):
                p.logit_prob = exp_val / sum_path_exp
        else:
            for p in self.paths:
                p.logit_prob = 1 / len(self.paths) if len(self.paths) > 0 else 0

    def update_path_flow(self, _path_flow):
        """更新每条路径的流量。"""
        for p, flow in zip(self.paths, _path_flow):
            p.flow = flow

    def verify_edge_flows(self):
        """验证边的流量是否被正确分配。"""
        print("Verifying edge flows:")
        for u, v, data in self.graph.edges(data=True):
            print(f"Edge ({u}, {v}): Flow = {data['xa']}")

class GeneticAlgorithm:
    def __init__(self, graph, network, maxgen=50, sizepop=100, pcross=0.7, pmutation=0.1):
        self.graph = graph
        self.network = network
        self.maxgen = maxgen
        self.sizepop = sizepop
        self.pcross = pcross
        self.pmutation = pmutation

        self.num_phases = 32
        self.bounds = np.array([[20, 70] if i % 4 in {0, 2} else [20, 45] for i in range(self.num_phases)])

    def initialize_population(self):
        return np.array([np.random.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1]) for _ in range(self.sizepop)])

    def update_bounds(self):
        intersections = [
            {'intersection_id': 1, 'phases': {1: 0, 2: 1, 3: 2, 4: 3}},
            {'intersection_id': 2, 'phases': {1: 4, 2: 5, 3: 6, 4: 7}},
            {'intersection_id': 3, 'phases': {1: 8, 2: 9, 3: 10, 4: 11}},
            {'intersection_id': 4, 'phases': {1: 12, 2: 13, 3: 14, 4: 15}},
            {'intersection_id': 5, 'phases': {1: 16, 2: 17, 3: 18, 4: 19}},
            {'intersection_id': 6, 'phases': {1: 20, 2: 21, 3: 22, 4: 23}},
            {'intersection_id': 7, 'phases': {1: 24, 2: 25, 3: 26, 4: 27}},
            {'intersection_id': 8, 'phases': {1: 28, 2: 29, 3: 30, 4: 31}}
        ]

        bounds = []

        for intersection in intersections:
            intersection_id = intersection['intersection_id']
            phases = intersection['phases']
            C = self.network.cycles.get(intersection_id, 90)
            L = 10

            max_flow_ratios = {}

            for phase_id, index in phases.items():
                phase_edges = [e for e in self.graph.edges(data=True) if e[2].get('intersection_id') == intersection_id and e[2].get('phase_id') == phase_id]
                max_flow_ratio = max(e[2]['xa'] / e[2]['cap'] for e in phase_edges if e[2]['cap'] > 0) if phase_edges else 0
                max_flow_ratios[phase_id] = max_flow_ratio

            remaining_time = C - L

            for phase_id, index in phases.items():
                min_time = int((C * max_flow_ratios[phase_id]) / 0.85)
                max_time = remaining_time - sum([int((C * max_flow_ratios[p]) / 0.85) for p in phases if p != phase_id])
                bounds.append((min_time, max_time))
                remaining_time -= min_time

        self.bounds = np.array(bounds)

    def select(self, individuals, fitness):
        fitness1 = np.where(fitness == 0, np.inf, 1.0 / fitness)  # 防止除以0
        fitness1 = np.nan_to_num(fitness1, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN and inf
        total_fitness = np.sum(fitness1)
        if np.isnan(total_fitness) or total_fitness == 0:
            prob = np.ones(len(fitness)) / len(fitness)
        else:
            prob = fitness1 / total_fitness
        index = np.random.choice(range(len(individuals)), len(individuals), p=prob)
        return individuals[index, :]

    def crossover(self, chrom):
        sizepop = len(chrom)
        for _ in range(sizepop):
            if np.random.rand() > self.pcross:
                continue
            idx1, idx2 = np.random.choice(range(sizepop), 2, replace=False)
            point = np.random.randint(len(self.bounds))
            alpha = np.random.rand()
            child1 = alpha * chrom[idx1, point] + (1 - alpha) * chrom[idx2, point]
            child2 = alpha * chrom[idx2, point] + (1 - alpha) * chrom[idx1, point]
            chrom[idx1, point] = np.clip(child1, self.bounds[point, 0], self.bounds[point, 1])
            chrom[idx2, point] = np.clip(child2, self.bounds[point, 0], self.bounds[point, 1])
        return chrom

    def mutation(self, chrom, num):
        sizepop = len(chrom)
        for i in range(sizepop):
            if np.random.rand() > self.pmutation:
                continue
            point = np.random.randint(len(self.bounds))
            fg = (np.random.rand() * (1 - num / self.maxgen)) ** 2
            if np.random.rand() > 0.5:
                chrom[i, point] = chrom[i, point] + (self.bounds[point, 1] - chrom[i, point]) * fg
            else:
                chrom[i, point] = chrom[i, point] - (chrom[i, point] - self.bounds[point, 0]) * fg
            chrom[i, point] = np.clip(chrom[i, point], self.bounds[point, 0], self.bounds[point, 1])
        return chrom

    def test_solution(self, individual):
        for i, (bound_min, bound_max) in enumerate(self.bounds):
            if not (bound_min <= individual[i] <= bound_max):
                return False
        return True

    def fitness_function(self, green_times):
        intersections = [
            {'intersection_id': 1, 'phases': {1: 0, 2: 1, 3: 2, 4: 3}},
            {'intersection_id': 2, 'phases': {1: 4, 2: 5, 3: 6, 4: 7}},
            {'intersection_id': 3, 'phases': {1: 8, 2: 9, 3: 10, 4: 11}},
            {'intersection_id': 4, 'phases': {1: 12, 2: 13, 3: 14, 4: 15}},
            {'intersection_id': 5, 'phases': {1: 16, 2: 17, 3: 18, 4: 19}},
            {'intersection_id': 6, 'phases': {1: 20, 2: 21, 3: 22, 4: 23}},
            {'intersection_id': 7, 'phases': {1: 24, 2: 25, 3: 26, 4: 27}},
            {'intersection_id': 8, 'phases': {1: 28, 2: 29, 3: 30, 4: 31}}
        ]

        for intersection in intersections:
            intersection_id = intersection['intersection_id']
            phases = intersection['phases']

            for phase_id, index in phases.items():
                for e in [e for e in self.graph.edges(data=True) if e[2].get('intersection_id') == intersection_id and e[2].get('phase_id') == phase_id]:
                    e[2]['green_time'] = green_times[index]

        # 更新网络路径成本
        self.network.update_edge_cost(green_times)
        self.network.update_path_cost(green_times)
        self.network.update_edge_flow()  # 确保流量被更新

        # 计算总延误
        delay = sum(e[2]['t0'] * e[2]['xa'] for e in self.graph.edges(data=True))

        # 调试信息
        print("\n== Fitness Function Debug Information ==")
        for u, v, data in self.graph.edges(data=True):
            intersection_id = data.get('intersection_id')
            phase_id = data.get('phase_id')
            green_time = data.get('green_time')
            xa = data.get('xa')
            cap = data.get('cap')
            t0 = data.get('t0')
            print(f"Edge ({u}, {v}): Intersection {intersection_id}, Phase {phase_id}, Green Time = {green_time}, xa = {xa}, cap = {cap}, t0 = {t0}")

        print(f"Fitness function: Green Times = {green_times}, Delay = {delay}\n")

        if math.isnan(delay) or delay == float('inf') or delay == float('-inf'):
            print("Invalid delay calculation, returning inf.")
            return float('inf')

        return delay

    def print_results(self, bestchrom):
        intersections = [
            {'intersection_id': 1, 'phases': {1: 0, 2: 1, 3: 2, 4: 3}},
            {'intersection_id': 2, 'phases': {1: 4, 2: 5, 3: 6, 4: 7}},
            {'intersection_id': 3, 'phases': {1: 8, 2: 9, 3: 10, 4: 11}},
            {'intersection_id': 4, 'phases': {1: 12, 2: 13, 3: 14, 4: 15}},
            {'intersection_id': 5, 'phases': {1: 16, 2: 17, 3: 18, 4: 19}},
            {'intersection_id': 6, 'phases': {1: 20, 2: 21, 3: 22, 4: 23}},
            {'intersection_id': 7, 'phases': {1: 24, 2: 25, 3: 26, 4: 27}},
            {'intersection_id': 8, 'phases': {1: 28, 2: 29, 3: 30, 4: 31}}
        ]

        for intersection in intersections:
            intersection_id = intersection['intersection_id']
            phases = intersection['phases']
            cycle_time = sum(bestchrom[index] for index in phases.values()) + 10  # Loss time is 10 seconds

            print(f"Intersection {intersection_id} - Cycle Time: {cycle_time:.2f}s")
            for phase_id, index in phases.items():
                print(f"  Phase {phase_id}: Green Time = {bestchrom[index]:.2f}s")

    def run(self):
        individuals = self.initialize_population()
        self.network.update_edge_cost()  # Update cycle times
        self.update_bounds()  # Update the bounds based on the new cycles

        fitness = np.array([self.fitness_function(individual) for individual in individuals])

        bestfitness = np.min(fitness)
        bestindex = np.argmin(fitness)
        bestchrom = individuals[bestindex, :]
        trace = [bestfitness]

        for gen in range(self.maxgen):
            self.sync_graph_with_network()  # 同步网络数据到算法中使用的图
            print(f'Generation: {gen + 1}')
            individuals = self.select(individuals, fitness)
            individuals = self.crossover(individuals)
            individuals = self.mutation(individuals, gen)
            fitness = np.array([self.fitness_function(individual) for individual in individuals])

            newbestfitness = np.min(fitness)
            newbestindex = np.argmin(fitness)

            if bestfitness > newbestfitness:
                bestfitness = newbestfitness
                bestchrom = individuals[newbestindex, :]

            trace.append(bestfitness)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(trace)), trace, 'b--')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Total Delay)')
        plt.show()

        self.print_results(bestchrom)

    def sync_graph_with_network(self):
        for u, v in self.graph.edges():
            if (u, v) in self.network.graph.edges():
                self.graph.edges[u, v]['xa'] = self.network.graph.edges[u, v]['xa']

def set_network():
    Graph = nx.DiGraph()
    intersections = {
        1: {
            'edges': [
                (21, 26), (21, 24), (21, 28), (25, 22),
                (25, 28), (25, 24), (27, 24), (27, 22),
                (27, 26), (23, 28), (23, 26), (23, 22)
            ],
            'phases': {
                1: [(21, 26), (25, 22), (21, 28), (25, 24)],
                2: [(21, 24), (25, 28)],
                3: [(27, 24), (23, 28), (27, 26), (23, 22)],
                4: [(27, 22), (23, 26)]
            }
        },
        2: {
            'edges': [
                (29, 34), (29, 32), (29, 36), (33, 30),
                (33, 36), (33, 32), (35, 32), (35, 30),
                (35, 34), (31, 36), (31, 34), (31, 30)
            ],
            'phases': {
                1: [(29, 34), (33, 30), (29, 36), (33, 32)],
                2: [(29, 32), (33, 36)],
                3: [(35, 32), (31, 36), (35, 34), (31, 30)],
                4: [(35, 30), (31, 34)]
            }
        },
        3: {
            'edges': [
                (37, 42), (37, 40), (37, 44), (41, 38),
                (41, 44), (41, 40), (43, 40), (43, 38),
                (43, 42), (39, 44), (39, 42), (39, 38)
            ],
            'phases': {
                1: [(37, 42), (41, 38), (37, 44), (41, 40)],
                2: [(37, 40), (41, 44)],
                3: [(43, 40), (39, 44), (43, 42), (39, 38)],
                4: [(43, 38), (39, 42)]
            }
        },
        4: {
            'edges': [
                (45, 50), (45, 48), (45, 52), (49, 46),
                (49, 52), (49, 48), (51, 48), (51, 46),
                (51, 50), (47, 52), (47, 50), (47, 46)
            ],
            'phases': {
                1: [(45, 50), (49, 46), (45, 52), (49, 48)],
                2: [(45, 48), (49, 52)],
                3: [(51, 48), (47, 52), (51, 50), (47, 46)],
                4: [(51, 46), (47, 50)]
            }
        },
        5: {
            'edges': [
                (53, 58), (53, 56), (53, 60), (57, 54),
                (57, 60), (57, 56), (59, 56), (59, 54),
                (59, 58), (55, 60), (55, 58), (55, 54)
            ],
            'phases': {
                1: [(53, 58), (57, 54), (53, 60), (57, 56)],
                2: [(53, 56), (57, 60)],
                3: [(59, 56), (55, 60), (59, 58), (55, 54)],
                4: [(59, 54), (55, 58)]
            }
        },
        6: {
            'edges': [
                (61, 66), (61, 64), (61, 68), (65, 62),
                (65, 68), (65, 64), (67, 64), (67, 62),
                (67, 66), (63, 68), (63, 66), (63, 62)
            ],
            'phases': {
                1: [(61, 66), (65, 62), (61, 68), (65, 64)],
                2: [(61, 64), (65, 68)],
                3: [(67, 64), (63, 68), (67, 66), (63, 62)],
                4: [(67, 62), (63, 66)]
            }
        },
        7: {
            'edges': [
                (69, 74), (69, 72), (69, 76), (73, 70),
                (73, 76), (73, 72), (75, 72), (75, 70),
                (75, 74), (71, 76), (71, 74), (71, 70)
            ],
            'phases': {
                1: [(69, 74), (73, 70), (69, 76), (73, 72)],
                2: [(69, 72), (73, 76)],
                3: [(75, 72), (71, 76), (75, 74), (71, 70)],
                4: [(75, 70), (71, 74)]
            }
        },
        8: {
            'edges': [
                (77, 82), (77, 80), (77, 84), (81, 78),
                (81, 84), (81, 80), (83, 80), (83, 78),
                (83, 82), (79, 84), (79, 82), (79, 78)
            ],
            'phases': {
                1: [(77, 82), (81, 78), (77, 84), (81, 80)],
                2: [(77, 80), (81, 84)],
                3: [(83, 80), (79, 84), (83, 82), (79, 78)],
                4: [(83, 78), (79, 82)]
            }
        }
    }

    for i in range(1, n + 1):
        Graph.add_node(i)

    for intersection_id, data in intersections.items():
        for phase_id, edges in data['phases'].items():
            green_time_initial = 66 if phase_id in {1, 3} else 30
            for edge in edges:
                Graph.add_edge(edge[0], edge[1], xa=0, t0=0, cap=2000, type='signal', intersection_id=intersection_id, phase_id=phase_id, green_time=green_time_initial)

    initial_weights = {
        (1, 21): 0.4, (22, 1): 0.4, (2, 29): 0.4, (30, 2): 0.4,
        (3, 31): 0.4, (32, 3): 0.4, (4, 47): 0.4, (48, 4): 0.4,
        (5, 63): 0.4, (64, 5): 0.4, (6, 79): 0.4, (80, 6): 0.4,
        (7, 81): 0.4, (82, 7): 0.4, (8, 73): 0.4, (74, 8): 0.4,
        (9, 75): 0.4, (76, 9): 0.4, (10, 59): 0.4, (60, 10): 0.4,
        (11, 43): 0.4, (44, 11): 0.4, (12, 27): 0.4, (28, 12): 0.4,
        (24, 35): 1, (36, 23): 1, (40, 51): 1, (52, 39): 1,
        (56, 67): 1, (68, 55): 1, (72, 83): 1, (84, 71): 1,
        (26, 37): 1, (38, 25): 1, (42, 53): 1, (54, 41): 1,
        (58, 69): 1, (70, 57): 1, (34, 45): 1, (46, 33): 1,
        (50, 61): 1, (62, 49): 1, (66, 77): 1, (78, 65): 1
    }

    bpr_edges = {
        (1, 21), (22, 1), (2, 29), (30, 2),
        (3, 31), (32, 3), (4, 47), (48, 4),
        (5, 63), (64, 5), (6, 79), (80, 6),
        (7, 81), (82, 7), (8, 73), (74, 8),
        (9, 75), (76, 9), (10, 59), (60, 10),
        (11, 43), (44, 11), (12, 27), (28, 12),
        (24, 35), (36, 23), (40, 51), (52, 39),
        (56, 67), (68, 55), (72, 83), (84, 71),
        (26, 37), (38, 25), (42, 53), (54, 41),
        (58, 69), (70, 57), (34, 45), (46, 33),
        (50, 61), (62, 49), (66, 77), (78, 65)
    }

    for edge in bpr_edges:
        t0 = initial_weights.get(edge, 0)
        Graph.add_edge(edge[0], edge[1], xa=0, t0=t0, cap=2000, type='BPR', intersection_id=None, phase_id=None, green_time=None)

    return Graph

def find_paths_for_od_pairs(Graph, od_pairs, max_paths=3):
    od_paths = {}
    for start_node, end_node in od_pairs:
        if start_node != end_node:
            all_paths = list(nx.all_simple_paths(Graph, source=start_node, target=end_node, cutoff=15))
            sorted_paths = sorted(all_paths, key=len)[:max_paths]
            edge_paths = [[(path[i], path[i+1]) for i in range(len(path) - 1)] for path in sorted_paths]
            od_paths[(start_node, end_node)] = edge_paths

    return od_paths

def find_shortest_paths_dijkstra(graph, od_pairs):
    """
    使用 Dijkstra 算法找到每对 OD 对应的最短路径。
    """
    paths = []
    for start_node, end_node in od_pairs:
        if start_node != end_node:
            try:
                path = nx.dijkstra_path(graph, source=start_node, target=end_node, weight='t0')
                path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                temp_path = path_class()
                temp_path.links = path_edges
                paths.append(temp_path)
            except nx.NetworkXNoPath:
                continue
    return paths

def set_od_demand():
    od_pairs = [(i, j) for i in range(1, 13) for j in range(1, 13) if i != j]
    demand = {}
    primary_od_pairs = [(1, 7), (1, 8), (2, 7), (2, 8)]
    
    for od in od_pairs:
        if od in primary_od_pairs:
            demand[od] = 1200
        else:
            demand[od] = 550
    
    return demand

def MSA(graph, my_network, od_demand, fixed_percentage):
    """
    使用逐步平均法 (MSA) 进行交通分配。
    """
    maximum_iter = 100
    acceptable_gap = 0.001
    gap = 100

    od_pairs = list(od_demand.keys())
    my_network.paths = find_shortest_paths_dijkstra(graph, od_pairs)
   
    my_network.update_edge_cost()

    # 初始化路径流量
    I = 1
    x = [od_demand[od] / len(my_network.paths) for od in od_pairs]

    # 对主要 OD 对进行第一次路径分配后固定部分需求
    primary_od_pairs = [(1, 7), (1, 8), (2, 7), (2, 8)]
    fixed_flows = {}
    for od in primary_od_pairs:
        fixed_flows[od] = fixed_percentage * od_demand[od]

    while I < maximum_iter and gap > acceptable_gap:
        my_network.update_path_flow(x)
        my_network.update_edge_flow()
        my_network.update_path_cost()
        my_network.update_path_prob()

        my_network.paths = find_shortest_paths_dijkstra(graph, od_pairs)

        y = []
        for path in my_network.paths:
            od = (path.links[0][0], path.links[-1][1])
            if od in fixed_flows:
                y.append((od_demand[od] - fixed_flows[od]) * path.logit_prob + fixed_flows[od])
            else:
                y.append(od_demand[od] * path.logit_prob)

        for idx in range(len(x)):
            x[idx] = x[idx] + 1 / I * (y[idx] - x[idx])

        gap = max([abs(path.prob - path.logit_prob) for path in my_network.paths])

        I += 1

if __name__ == "__main__":
    graph = set_network()
    network = my_network_class(graph)
    ga = GeneticAlgorithm(graph, network)
    ga.run()
    
    od_demand = set_od_demand()
    MSA(graph, network, od_demand, fixed_percentage=0.1)


