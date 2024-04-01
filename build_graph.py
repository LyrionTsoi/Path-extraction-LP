import sys
from numpy import array, zeros, multiply, dot, ceil, where, mod
import argparse
import datetime
import time
import math
import subprocess
import copy
from collections import OrderedDict
# from gurobipy import *
from gurobipy import Model, GRB
import pulp


class Node:
    def __init__(self, nodeId, info, coverage=0):
        self.nodeId = nodeId
        self.info = info
        self.children = []
        self.coverage = coverage


# class Graph:
#     def __init__(self):
#         self.nodes = {}
#         self.g = {}  # 使用空字典保存，使用邻接矩阵来保存边权 方便后面的dijkstra算法

#     def add_node(self, nodeId, info):
#         if nodeId not in self.nodes:
#             self.nodes[nodeId] = Node(nodeId, info)

#     def add_edge(self, parentId, childId):
#         if parentId not in self.nodes or childId not in self.nodes:
#             raise ValueError("Both nodes must be in the graph")
#         self.nodes[parentId].children.append(childId)
#         # 初始化边权重
#         if parentId not in self.g:
#             self.g[parentId] = {}
#         self.g[parentId][childId] = 0  # 初始权重设置为0


class Graph:
    def __init__(self):
        self.nodes = {}
        self.g = {}  # 使用空字典保存边权，方便后面的算法

    def add_node(self, nodeId, info):
        if nodeId not in self.nodes:
            self.nodes[nodeId] = Node(nodeId, info)

    def add_edge(self, parentId, childId):
        if parentId not in self.nodes or childId not in self.nodes:
            raise ValueError("Both nodes must be in the graph")
        self.nodes[parentId].children.append(childId)
        # 初始化边权重
        if parentId not in self.g:
            self.g[parentId] = {}
        self.g[parentId][childId] = 0  # 初始权重设置为0

    def init_g(self):
        for i in self.nodes:
            self.g[i] = {}
            for j in self.nodes:
                self.g[i][j] = 0


def optimize_flows(graph):
    # 创建一个新的模型
    model = Model("MinimizeFlowDiscrepancy")

    # 为 graph 中的每条边创建流量变量
    flows = {}
    for i in graph.nodes:
        for child in graph.nodes[i].children:
            flows[(i, child)] = model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name=f"flow_{i}_{child}")

    # model.update()  # 确保所有变量都被添加到模型中

    # # 目标函数：最小化每个节点的流量与其丰度差异的总和
    # # 注意：实际实现可能需要根据丰度和流量计算差异
    # model.setObjective(sum(flows[(i, j)] for i in graph.nodes for j in graph.nodes[i].children) -
    #                    sum(graph.nodes[i].coverage for i in graph.nodes), GRB.MINIMIZE)

    # eps = 100

    # 为每个节点创建一个变量来表示流量与丰度之间差异的绝对值
    delta = {}
    for i in graph.nodes:
        delta[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"delta_{i}")

    # 更新模型以添加变量
    model.update()

    # 设置目标函数：最小化所有 delta 的总和
    model.setObjective(sum(delta[i] for i in graph.nodes), GRB.MINIMIZE)

    # 添加约束以确保 delta[i] 大于等于流量和丰度之间的差异
    for i in graph.nodes:
        in_flow = sum(flows[(j, i)]
                      for j in graph.nodes if i in graph.nodes[j].children)
        out_flow = sum(flows[(i, j)] for j in graph.nodes[i].children)
        total_flow = in_flow + out_flow  # 根据您的模型调整这里的计算方式
        model.addConstr(delta[i] >= total_flow -
                        graph.nodes[i].coverage, f"delta_pos_{i}")
        model.addConstr(delta[i] >= -(total_flow -
                        graph.nodes[i].coverage), f"delta_neg_{i}")

    # 添加流量守恒约束
    for i in graph.nodes:
        # 流入量 = 流出量
        in_flow = sum(flows[(j, i)]
                      for j in graph.nodes if i in graph.nodes[j].children)
        out_flow = sum(flows[(i, j)] for j in graph.nodes[i].children)
        model.addConstr(in_flow == out_flow, f"FlowConservation_{i}")

    # 求解问题前，检查目标函数和约束的设定
    print("Objective function before optimization:", model.getObjective())

    # 求解问题
    model.optimize()

    # 更新 graph 的 g[i][j] 流量信息，安全检查边是否存在
    for (i, j), flow_var in flows.items():
        if i in graph.g and j in graph.g[i]:
            # print(f"node u:{i} -> node v:{j}, weigth:{flow_var.X}")
            graph.g[i][j] = flow_var.X
        else:
            # 如果边不存在，可以选择添加新边，或者仅打印警告
            print(
                f"Warning: Edge ({i}, {j}) not found in the graph, creating new edge with flow {flow_var.X}.")
            # 创建边和流量
            graph.add_edge(i, j)  # 如果需要，可以在 add_edge 方法内部处理重复边的情况
            graph.g[i][j] = flow_var.X
            # print(f"node u:{i} -> node v:{j}, weigth:{flow_var.X}")

    return graph

# 解析FASTA文件并构建图


def build_graph_from_fasta(file_path):
    graph = Graph()

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            node_value = int(parts[0])
            node_info = parts[1]
            # 注意这里的map是做扁平化处理，不是和C++一样
            children_values = map(int, parts[2:])

            graph.add_node(node_value, node_info)

            for child_value in children_values:
                graph.nodes[node_value].children.append(child_value)

    return graph


def update_coverage(graph, file_path):
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' : ')
            node_value = int(parts[0])
            coverage = int(parts[1])

            # 检查节点是否存在于图中
            if node_value in graph.nodes:
                graph.nodes[node_value].coverage = coverage
            else:
                print(f"Warning: Node {node_value} not found in the graph.")


# 假设graph是已经初始化好的图实例，包含节点的丰度信息和孩子信息
# optimized_graph = optimize_flows(graph)


file_path = '6-graph.fasta'
coverage_file_path = '6-coverage.txt'
graph = build_graph_from_fasta(file_path)
update_coverage(graph, coverage_file_path)
graph.init_g()
optimized_graph = optimize_flows(graph)


# 打印图的节点和边
# for nodeId, node in graph.nodes.items():
#     children = ', '.join(str(child) for child in node.children)
#     print(f"Node {nodeId}: info={node.info}, children=[{children}]")

# 打印每个节点的丰度
# for nodeId, node in graph.nodes.items():
#     print(f"Node {nodeId} : coverage {node.coverage}")


# 遍历每个节点及其邻居
# for parent_id, neighbors in graph.g.items():
#     print(f"Parent Node {parent_id}:")
#     # 遍历该节点的邻居及其对应的边权重
#     for child_id, weight in neighbors.items():
#         print(f"  -> Child Node {child_id}, Weight: {weight}")

# for i in optimized_graph.g:
#     for j in optimized_graph.g[i]:
#         # 在这里执行对 g[i][j] 的操作
#         weight = optimized_graph.g[i][j]
#         print(f"Weight from node {i} to node {j}: {weight}")

# for parentId in graph.nodes:
#     print(f"parentId: {parentId}")
#     for childId in graph.nodes[parentId].children:
#         print(f"    childId: {childId}")

