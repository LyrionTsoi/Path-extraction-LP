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
from gurobipy import Model, GRB, LinExpr
import pulp
# from graph_functions import read_gfa, get_seq, is_cycle, rev_comp
from graph_tool.all import Graph
from graph_tool.topology import is_DAG, all_circuits, topological_sort, shortest_distance, shortest_path
import heapq

class Node:
    def __init__(self, nodeId, info, coverage=0):
        self.nodeId = nodeId
        self.info = info
        self.children = []
        self.coverage = coverage

from gurobipy import Model, GRB, quicksum

# 这里利用约束条件，使得其成为绝对值差异最小化（函数仍未完成，存在一些问题）
def optimize_flow_as_abs(graph):
    m = Model("lp")
    
    # 创建流量变量
    flow_vars = m.addVars([(e.source(), e.target()) for e in graph.edges()], name="flow", lb=0, vtype=GRB.CONTINUOUS)

    # 创建差异变量
    diff_vars = m.addVars(graph.vertices(), name="diff", lb=0, vtype=GRB.CONTINUOUS)

    # 目标函数
    m.setObjective(sum(diff_vars[v] for v in graph.vertices()), GRB.MINIMIZE)

    # 流量守恒约束，除了源点和汇点
    source = graph.vertex(0)
    sink = graph.vertex(graph.num_vertices() - 1)
    for v in graph.vertices():
        if v == source or v == sink:
            continue
        
        inflow = LinExpr()
        outflow = LinExpr()
        for e in v.in_edges():
            inflow += flow_vars[e.source(), e.target()]
        for e in v.out_edges():
            outflow += flow_vars[e.source(), e.target()]
        m.addConstr(inflow == outflow, f"conservation_{int(v)}")

    # 对于每个顶点，设置差异约束
    coverage_property = graph.vertex_properties["coverage"]
    for v in graph.vertices():
        inflow = LinExpr()
        for e in v.in_edges():
            inflow += flow_vars[(e.source(), e.target())]
        
        # coverage = LinExpr()
        coverage = coverage_property[v]
        total_flow_expr = LinExpr()
        total_flow_expr += inflow
        total_flow_expr -= coverage  # 将丰度作为常数从入流量中减去

        # 为了避免纯常数表达式，我们可以通过添加一个微小的变量项来确保 total_flow_expr 保持为线性表达式
        # 注意：以下代码假设至少有一个流量变量，如果完全没有流量变量，可能需要进一步调整
        m.addGenConstrAbs(diff_vars[v], total_flow_expr, name=f"abs_diff_{int(v)}")

    # 求解模型
    m.optimize()

    # 打印结果
    if m.status == GRB.OPTIMAL:
        print("Optimal solution found.")
        for e in graph.edges():
            print(f"Flow on edge {int(e.source())}->{int(e.target())}: {flow_vars[e.source(), e.target()].X}")
    else:
        print("Optimal solution not found.")


# 因为我们需要计算的是流量与丰度的差异的最小化，理论上这个差异是一个绝对值的
# 但是对应线性规划其无法处理绝对值最小化，所以这里使用了二次规划
def optimize_flow_as_qp(graph):
    m = Model("qp")
    
    # 创建流量变量
    flow_vars = m.addVars([(e.source(), e.target()) for e in graph.edges()], name="flow", lb=0, vtype=GRB.CONTINUOUS)

    # 创建差异的平方变量，不再直接使用差异变量
    # 注意：我们不需要单独的变量来表示差异的平方，它将直接在目标函数中计算

    # 修改目标函数：最小化差异平方之和
    coverage_property = graph.vertex_properties["coverage"]
    obj = quicksum((quicksum(flow_vars[(e.source(), e.target())] for e in v.in_edges()) - coverage_property[v]) * 
                   (quicksum(flow_vars[(e.source(), e.target())] for e in v.in_edges()) - coverage_property[v]) 
                   for v in graph.vertices())
    m.setObjective(obj, GRB.MINIMIZE)

    # 流量守恒约束，除了源点和汇点
    source = graph.vertex(0)
    sink = graph.vertex(graph.num_vertices() - 1)
    for v in graph.vertices():
        if v == source or v == sink:
            continue
        
        inflow = quicksum(flow_vars[e.source(), e.target()] for e in v.in_edges())
        outflow = quicksum(flow_vars[e.source(), e.target()] for e in v.out_edges())
        m.addConstr(inflow == outflow, f"conservation_{int(v)}")

    # 求解模型
    m.optimize()

    # 将流量存储进图中（因为这里为浮点数，所以就将其取整）
    if m.status == GRB.OPTIMAL:
        print("Optimal solution found.")

        # 创建一个边属性来存储流量值
        flow_property = graph.new_edge_property("int")

        # 遍历边，更新边的流量属性
        for e in graph.edges():
            source = int(e.source())
            target = int(e.target())
            flow_value = int(flow_vars[source, target].X)  # 取整数部分
            if flow_value != 0:
                flow_property[e] = flow_value
                print(f"Flow on edge {source}->{target}: {flow_property[e]}")

        # 将流量属性绑定到图中
        graph.edge_properties["flow"] = flow_property
    else:
        print("Optimal solution not found.")



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

    return graph,vertex_dict


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


# 使用dijkstra找权值的最小的路径
 def dijkstra(graph):
    flow_property = graph.edge_properties["flow"]
    # 最小堆
    heap = [] 
    # dist[i] 记录源点到i点的距离最小值
    dist = [sys.maxsize] * graph.num_vertices()

    for v in graph.vertices():
        heapq.heappush(heap)





file_path = '6-graph.fasta'
coverage_file_path = '6-coverage.txt'
graph,vertex_dict = build_graph_from_fasta(file_path)
update_coverage(graph, coverage_file_path)
# print_vertex_properties(graph)
# optimize_flow(graph)
optimize_flow_as_qp(graph)



