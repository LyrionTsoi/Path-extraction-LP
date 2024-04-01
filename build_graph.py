import string
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
# from graph_functions import read_gfa, get_seq, is_cycle, rev_comp
from graph_tool.all import Graph
from graph_tool.topology import is_DAG, all_circuits, topological_sort, shortest_distance, shortest_path

class Node:
    def __init__(self, nodeId, info, coverage=0):
        self.nodeId = nodeId
        self.info = info
        self.children = []
        self.coverage = coverage

def optimize_flows(graph):
    # 创建一个新的模型
    m = Model("lp")

    # 为 graph 中的每条边创建流量变量
    x = m.addVars(list(graph.edges()), lb=0,
                  vtype=GRB.CONTINUOUS, name='x')
    # add additional variables to implement absolute values
    y = m.addVars(vg_to_contigs, lb=0, vtype=GRB.CONTINUOUS, name='y')

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
    graph = Graph(directed=True)

    info = graph.new_vertex_property("string")
    vertex_dict = {}  # 创建一个ID到顶点对象的映射

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            node_Id = int(parts[0])
            node_info = parts[1]  # 直接使用字符串
            children_values = map(int, parts[2:])

            # 检查顶点是否已存在，如果不存在，则添加
            if node_Id not in vertex_dict:
                vertex_dict[node_Id] = graph.add_vertex()
            info[vertex_dict[node_Id]] = node_info

            for child in children_values:
                if child not in vertex_dict:
                    vertex_dict[child] = graph.add_vertex()
                graph.add_edge(vertex_dict[node_Id], vertex_dict[child])

    graph.vertex_properties["info"] = info  # 将属性添加到图的外部

    return graph


def update_coverage(graph, file_path):
    coverage_property = graph.new_vertex_property("int")  # 避免与局部变量冲突
    id_to_vertex = {int(v): v for v in graph.vertices()}  # 创建ID到顶点的映射

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' : ')
            node_Id = int(parts[0])
            cov_value = int(parts[1])  # 使用不同的变量名来存储覆盖率值

            if node_Id in id_to_vertex:  # 使用映射来查找顶点
                coverage_property[id_to_vertex[node_Id]] = cov_value
            else:
                print(f"Warning: Node {node_Id} not found in the graph.")
    
    graph.vertex_properties["coverage"] = coverage_property  # 将属性添加到图中



# 使用函数来打印顶点和其属性
def print_vertex_properties(graph):
    info_property = graph.vertex_properties["info"]
    coverage_property = graph.vertex_properties["coverage"]

    for v in graph.vertices():
        v_id = int(v)  # 获取顶点ID（如果您维护了一个从顶点对象到ID的映射，这里可能需要调整）
        v_info = info_property[v]
        v_coverage = coverage_property[v]
        print(f"顶点 {v_id}: info = {v_info}, coverage = {v_coverage}")



file_path = '6-graph.fasta'
coverage_file_path = '6-coverage.txt'
graph = build_graph_from_fasta(file_path)
update_coverage(graph, coverage_file_path)
print_vertex_properties(graph)
# optimized_graph = optimize_flows(graph)



