import functools
from multiprocessing import Pool, cpu_count
import random
import string
import sys
from numpy import array, zeros, multiply, dot, ceil, where, mod
import numpy as np
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
import validate_data
from gurobipy import Model, GRB, quicksum



# 解析FASTA文件并构建图
def build_graph_from_fasta(file_path):
    graph = Graph(directed=True)

    info = graph.new_vertex_property("string")  # 创建顶点属性
    vertex_dict = {}  # 创建一个ID到顶点对象的映射

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue  # 忽略无效行

            node_id = int(parts[0])
            node_info = parts[1]
            children_values = list(map(int, parts[2:]))

            if node_id not in vertex_dict:
                vertex_dict[node_id] = graph.add_vertex(node_id)
            info[node_id] = node_info

            for child in children_values:
                if child not in vertex_dict:
                    vertex_dict[child] = graph.add_vertex(child)
                graph.add_edge(node_id, child)

        graph.vertex_properties["info"] = info

    return graph, vertex_dict


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


# 调用函数导出流量信息到文件
def export_flow_info(graph, filename="flow_info.txt"):
    # 获取流量属性
    flow_property = graph.edge_properties["flow"]

    # 打开文件准备写入
    with open(filename, "w") as file:
        # 遍历所有边，获取流量信息并写入文件
        for e in graph.edges():
            source = int(e.source())
            target = int(e.target())
            flow = flow_property[e]
            # 格式化字符串：点u -> 点v + 流量信息
            line = f"{source} -> {target} with flow {flow}\n"
            file.write(line)

    print(f"Flow information has been exported to {filename}.")


def export_path_info(path, num, filename="Path_info.txt"):
    with open(filename, "a") as file:  # 使用 "a" 模式以追加的方式写入文件
        # 使用f-string格式化路径编号，确保num变量被正确解析
        line = f"path{num}: \n"
        # 将节点ID转换为字符串，并用逗号连接
        line += ", ".join(str(nodeId) for nodeId in path)
        line += "\n"  # 添加换行符以分隔不同的路径
        file.write(line)  # 写入整个路径信息

    print(f"Path information has been exported to {filename}.")
    
    
def export_pathExtraction_polio_fasta(graph, path, num, filename="validation-set.fasta"):

    if "info" not in graph.vertex_properties:
        print(f"The 'info' property does not exist on graph's vertices.")

    info_property = graph.vertex_properties["info"]

    with open(filename, "a") as file:
        line = f">seq_{num+1}\n"
        infos = [info_property[graph.vertex(nodeId)] for nodeId in path]
        line += "".join(infos)
        line += "\n"
        file.write(line)
    print(f"Path information with 'info' has been exported to {filename}.")


def export_pathExtraction_polio_txt(graph, path, num, filename="validation-set.txt"):
    if "info" not in graph.vertex_properties:
        print(f"The 'info' property does not exist on graph's vertices.")

    info_property = graph.vertex_properties["info"]

    with open(filename, "a") as file:
        line = f">seq_{num+1}\n"
        infos = [info_property[graph.vertex(nodeId)] for nodeId in path]
        line += "".join(infos)
        line += "\n"
        file.write(line)
    print(f"Path information with 'info' has been exported to {filename}.")


# 首先，我们需要一个函数来加载并缓存所有需要的序列数据
def load_fasta_data(filepaths):
    fasta_data = {}
    for filepath in filepaths:
        fasta_data[filepath] = validate_data.read_fasta_file(filepath)
    return fasta_data


# 滑动窗口的实现
def sliding_window(sequence, window_size=25):
    for i in range(len(sequence) - window_size + 1):
        yield sequence[i:i + window_size]


# 如果使用string中的string.count(query_seq)的话，这个算法是不重叠的算法
def count_overlapping_occurrences(ref_seq, query_seq):
    count = 0
    start = 0
    while True:
        start = ref_seq.find(query_seq, start)
        if start == -1: 
            break
        count += 1
        start += 1  # 移动一个字符后再次搜索，以允许重叠
    return count


# 这里要统计Polio-reads中序列出现的次数
def search_seq_inPolio(query_seq, fasta_data):
    cnt = 0
    kmercnt = 0
    if len(query_seq) < 25:
        print(f"error, len of query sequnce < 25")
        return 0

    for filepath in fasta_data:
        ref_seqs = fasta_data[filepath]
        for kmer in sliding_window(query_seq):
            kmercnt += 1
            for ref_seq in ref_seqs:
                string_ref_seq = str(ref_seq.seq)
                cnt += count_overlapping_occurrences(string_ref_seq, kmer)
                
    return int(cnt / kmercnt)



def find_seq(graph, prev, targetID):
    seq = []
    curSeq = {}
    info_property = graph.vertex_property["info"]

    while targetID != -1:
        curSeq += info_property[targetID]
        targetID = prev.get(targetID, -1)
    
    curSeq.reverse()
    seq.append(curSeq)

    return seq


# 拼接节点序列，使得节点总序列的长度大于25
# 注意这里我们都是采用的往前拼接的方法，理论上可能出现的是第一个点是源点，len(sourceInfo) < 25,要是这是往前拼接会失败（但是我们这里的几个图都不会出现这种情况，所以我们不考虑这种情况）
# 若要使用在更广泛的情况下，需要做一些修改
# attendtion
# Note that here we are using the method of forward concatenation. Theoretically, it is possible that the first point is the source point, and len(sourceInfo) < 25. 
# If we concatenate forward in this case, it will fail. (However, none of the graphs we are dealing with here will encounter this situation, so we do not need to consider it.)
def Concatenation_sequence(graph, curVertexID, curSeq):
    info_property = graph.vertex_properties["info"]
    curVertex = graph.vertex(curVertexID)

    # 如果当前序列长度超过25，立即返回这个序列
    if len(curSeq) > 25:
        return curSeq

    # 否则，继续递归查找
    for e in curVertex.in_edges():
        sourceID = int(e.source())
        sourceSeq = info_property[sourceID]
        nextSeq = sourceSeq + curSeq  # 向前拼接序列

        result_seq = Concatenation_sequence(graph, curVertexID=sourceID, curSeq=nextSeq)
        if result_seq:  # 如果找到了一个有效的序列，则返回
            return result_seq

    return None  # 如果没有找到任何有效序列，返回 None
    

# 单线程处理边权
def process_edge(graph, source_target_pair, info_property, fasta_data, weight_property):
    source_id, target_id = source_target_pair

    # 检查边权是否已存在
    edge = graph.edge(graph.vertex(source_id), graph.vertex(target_id))
    if edge is not None and weight_property[edge] != 0:
        print(f"Edge {source_id} -> {target_id} already has weight: {weight_property[edge]}")
        return (source_id, target_id, weight_property[edge])  # 如果已存在，直接返回现有权值


    seq_source_P = info_property[source_id]
    seq_target_P = info_property[target_id]

    seq = ''
    if len(seq_source_P) > 1:
        seq += seq_source_P[1:]
    else:
        seq += seq_source_P

    if len(seq_target_P) > 1:
        seq += seq_target_P[:-1]
    else:
        seq += seq_target_P


    if len(seq) >= 25:
        weight = search_seq_inPolio(seq, fasta_data)
    else:
        # 如果序列长度不够，尝试拼接
        seq = seq_source_P + seq_target_P
        if len(seq) >= 25:
            weight = search_seq_inPolio(seq, fasta_data)
        else:
            concatenated_seq = Concatenation_sequence(graph, curVertexID=source_id, curSeq=seq)
            weight = search_seq_inPolio(concatenated_seq, fasta_data)
    
    print(f"edge {source_id} -> {target_id} : {weight}")
    return (source_id, target_id, weight)


# 多线程处理
def edge_weight_multiprocess(graph, fasta_data):
    info_property = graph.vertex_properties["info"]

    # 确保有权值属性
    if 'weight' not in graph.edge_properties:
        weight_property = graph.new_edge_property("int")
        graph.edge_properties["weight"] = weight_property
    else:
        weight_property = graph.edge_properties["weight"]

    edges = [(int(e.source()), int(e.target())) for e in graph.edges()]

    print(f"cpu_count(): {cpu_count()-2}")
    with Pool(processes=cpu_count()-2) as pool:
        func = functools.partial(process_edge, graph, info_property=info_property, fasta_data=fasta_data,  weight_property=weight_property)
        results = pool.map(func, edges)

    weight_property = graph.new_edge_property("int")
    for source, target, weight in results:
        edge = graph.edge(graph.vertex(source), graph.vertex(target))
        weight_property[edge] = weight


def export_edge_weigth(graph, filename="edgeWeight-graph.txt"):
    if "weight" not in graph.edge_properties:
        print(f"The 'info' property does not exist on graph's edges.")

    wight_property = graph.edge_properties["weight"]
    with open(filename, "w") as file:  # 使用 "a" 模式以追加的方式写入文件
        for e in graph.edges():
            curWeight = wight_property[e]
            source = int(e.source())
            target = int(e.target())
            # 使用f-string格式化路径编号，确保num变量被正确解析
            line = f"edge {source} -> {target} : {curWeight}\n"
            file.write(line)  # 写入整个路径信息

    print(f"Edge's Weight information has been exported to {filename}.")


def read_edgeWeight(graph, filepath):
    if "weight" not in graph.edge_properties:
        weight_property = graph.new_edge_property("int")
    else:
        weight_property = graph.edge_properties['weight']


    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            sourceID = parts[1]
            targetID = parts[3]
            weight = parts[-1]
            edge = graph.edge(graph.vertex(sourceID), graph.vertex(targetID))
            weight_property[edge] = weight

    graph.edge_properties["weight"] = weight_property 