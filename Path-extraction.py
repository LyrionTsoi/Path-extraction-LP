import random
from numpy import array, zeros, multiply, dot, ceil, where, mod
import numpy as np
import copy
# from gurobipy import *
from gurobipy import Model, GRB, LinExpr
import pulp
from graph_tool.all import Graph
from graph_tool.topology import is_DAG, all_circuits, topological_sort, shortest_distance, shortest_path
import heapq
import validate_data
from gurobipy import Model, GRB, quicksum, abs_, max_
import graph_init
import test
import math

from gurobipy import Model, GRB, quicksum

def optimize_flow_as_min_paths(graph, vertex_to_id, flow_slack_tolerance, conservation_slack_tolerance):
    m = Model("network_flow")

    # Constants
    M = 10000  # Large constant for big-M constraints

    # Variables
    x = {}
    f = {}
    c = {}
    abs_diff = {}
    diff_pos = {}
    diff_neg = {}
    flow_slack = {}  # Slack variables for flow discrepancies
    conservation_slack = {}  # Slack variables for flow conservation

    sourceID = 0
    sinkID = 452

    # Populate variables based on incoming and outgoing edges for each node u
    for u in graph.vertices():
        u_id = vertex_to_id[u]
        if u_id == sourceID or u_id == sinkID:
            continue

        x[u_id] = {}
        f[u_id] = {}
        c[u_id] = {}
        abs_diff[u_id] = {}
        diff_pos[u_id] = {}
        diff_neg[u_id] = {}
        flow_slack[u_id] = {}
        conservation_slack[u_id] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=conservation_slack_tolerance,
                                          name=f"conservation_slack_{u_id}")
        for e_in in u.in_edges():
            i_id = vertex_to_id[e_in.source()]
            for e_out in u.out_edges():
                j_id = vertex_to_id[e_out.target()]
                path = (i_id, u_id, j_id)
                x[u_id][i_id, j_id] = m.addVar(vtype=GRB.BINARY, name=f"x_{i_id}_{u_id}_{j_id}")
                f[u_id][i_id, j_id] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_{i_id}_{u_id}_{j_id}")
                c[u_id][i_id, j_id] = min(graph.ep['weight'][e_in], graph.ep['weight'][e_out])
                abs_diff[u_id][i_id, j_id] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"abs_diff_{i_id}_{u_id}_{j_id}")
                diff_pos[u_id][i_id, j_id] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"diff_pos_{i_id}_{u_id}_{j_id}")
                diff_neg[u_id][i_id, j_id] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"diff_neg_{i_id}_{u_id}_{j_id}")
                flow_slack[u_id][i_id, j_id] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=flow_slack_tolerance,
                                               name=f"flow_slack_{i_id}_{u_id}_{j_id}")

    # Objective Function
    m.setObjective(
        quicksum(x[u_id][i_id, j_id] for u_id in x for i_id, j_id in x[u_id]) +
        quicksum(abs_diff[u_id][i_id, j_id] for u_id in abs_diff for i_id, j_id in abs_diff[u_id]),
        GRB.MINIMIZE
    )

    # Constraints to define absolute differences and include slack
    for u_id in abs_diff:
        for i_id, j_id in abs_diff[u_id]:
            m.addConstr(abs_diff[u_id][i_id, j_id] == diff_pos[u_id][i_id, j_id] + diff_neg[u_id][i_id, j_id])
            m.addConstr(f[u_id][i_id, j_id] - c[u_id][i_id, j_id] == diff_pos[u_id][i_id, j_id] - diff_neg[u_id][i_id, j_id])

    # At least one path through each node u
    for u_id in x:
        m.addConstr(quicksum(x[u_id][i, j] for i, j in x[u_id]) >= max(1,len(x[u_id]) - 1) )

    # Flow conservation at each node u with slack
    for u in graph.vertices():
        u_id = vertex_to_id[u]
        if u_id == sourceID or u_id == sinkID:
            continue
        inflow = quicksum(f[u_id][i_id, u_id] for i_id in vertex_to_id if (i_id, u_id) in f[u_id])
        outflow = quicksum(f[u_id][u_id, j_id] for j_id in vertex_to_id if (u_id, j_id) in f[u_id])
        m.addConstr(inflow - outflow <= conservation_slack[u_id])
        m.addConstr(outflow - inflow <= conservation_slack[u_id])

    # Big-M constraints to link x and f
    for u_id in f:
        for i_id, j_id in f[u_id]:
            m.addConstr(f[u_id][i_id, j_id] <= M * x[u_id][i_id, j_id])
            m.addConstr(f[u_id][i_id, j_id] * M >= x[u_id][i_id, j_id])

    # Solve the model
    m.optimize()

    # Output results
    x_opt = {}
    f_opt = {}
    if m.status == GRB.INFEASIBLE:
        print("Model is infeasible; computing IIS")
        m.computeIIS()
        m.write("model.ilp")
        print("IIS written to model.ilp")
        print("The following constraints cannot be satisfied:")
        for c in m.getConstrs():
            if c.IISConstr:
                print(f"{c.constrName} is part of the IIS.")
    elif m.status == GRB.OPTIMAL:
        print("Optimal solution found.")
        for u_id in x:
            x_opt[u_id] = {}
            f_opt[u_id] = {}
            for (i_id, j_id), var in x[u_id].items():
                x_opt[u_id][i_id, j_id] = var.X  # Store the optimized value
            for (i_id, j_id), var in f[u_id].items():
                f_opt[u_id][i_id, j_id] = var.X
        return x_opt, f_opt
    else:
        print("No optimal solution found.")
        return None, None



# 这里利用约束条件，使得其成为绝对值差异最小化（函数仍未完成，存在一些问题）
def optimize_flow_as_abs(graph):
    m = Model("lp")
    
    # 创建流量变量
    flow_vars = m.addVars([(e.source(), e.target()) for e in graph.edges()], name="flow", lb=0, vtype=GRB.CONTINUOUS)

    # 创建差异的正部分和负部分变量
    diff_pos_vars = m.addVars(graph.vertices(), name="diff_pos", lb=0, vtype=GRB.CONTINUOUS)
    diff_neg_vars = m.addVars(graph.vertices(), name="diff_neg", lb=0, vtype=GRB.CONTINUOUS)

    # 目标函数：最小化所有正部分和负部分差异变量的总和
    m.setObjective(sum(diff_pos_vars[v] + diff_neg_vars[v] for v in graph.vertices()), GRB.MINIMIZE)

    # 流量守恒约束，除了源点和汇点
    source = graph.vertex(0)
    sink = graph.vertex(graph.num_vertices() - 1)
    for v in graph.vertices():
        if v == source or v == sink:
            continue
        inflow = sum(flow_vars[e.source(), e.target()] for e in v.in_edges())
        outflow = sum(flow_vars[e.source(), e.target()] for e in v.out_edges())
        m.addConstr(inflow == outflow, f"conservation_{int(v)}")

    # 差异约束
    coverage_property = graph.vertex_properties["coverage"]
    for v in graph.vertices():
        inflow = sum(flow_vars[e.source(), e.target()] for e in v.in_edges())
        coverage = coverage_property[v]
        # 约束：inflow - coverage = diff_pos - diff_neg
        m.addConstr(inflow - coverage == diff_pos_vars[v] - diff_neg_vars[v], f"diff_{int(v)}")

    # 求解模型
    m.optimize()

    # 打印结果
    if m.status == GRB.OPTIMAL:
        print("Optimal solution found.")

        # 创建一个边属性来存储流量值
        flow_property = graph.new_edge_property("float")
        coverage_property = graph.vertex_properties['coverage']
        
        # 遍历边，更新边的流量属性
        for e in graph.edges():
            source = int(e.source())
            target = int(e.target())
            flow_value = flow_vars[source, target].X  # 取整数部分
            if flow_value >= 0.0000001:
                flow_property[e] = flow_value
            else:
                flow_property[e] = coverage_property[source] # 如果流量为0的话就将源点的丰度赋予

        # 将流量属性绑定到图中
        graph.edge_properties["flow"] = flow_property
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
        flow_property = graph.new_edge_property("float")

        # 遍历边，更新边的流量属性
        for e in graph.edges():
            source = int(e.source())
            target = int(e.target())
            flow_value = flow_vars[source, target].X  # 取整数部分
            if flow_value >= 0.0000001:
                flow_property[e] = flow_value
                # print(f"Flow on edge {source}->{target}: {flow_property[e]}")
            else :
                flow_property[e] = 0.0000001

        # 将流量属性绑定到图中
        graph.edge_properties["flow"] = flow_property
    else:
        print("Optimal solution not found.")


# Find the "shortest" Path by dijkstra 使用dijkstra找权值的最小的路径
def dijkstra(graph,a):
    flow_graph = copy.deepcopy(graph)
    flow_property = flow_graph.edge_properties["flow"]
    # 最小堆
    heap = []
    path = {}
    # dist[i] 记录源点到i点的距离最小值
    # 初始化为无穷大
    N = flow_graph.num_vertices() # reprense the number of vertices
    dist = [np.inf] * N
    visited = [False] * N
    prev = [-1] * N
    sourceId = 0
    sinkId = N - 1

    dist[sourceId] = 0
    heap = [(0,sourceId)] # first key is distance and the second is ID of vertex

    while heap:
        curDist, curId = heapq.heappop(heap)
        if visited[curId]:
            continue
        visited[curId] = True

        curVertex = flow_graph.vertex(curId)
        for e in curVertex.out_edges():
            v = int(e.target())

            weight = flow_property[e]
            if curDist + weight < dist[v]:
                dist[v] = curDist + weight
                heapq.heappush(heap,(dist[v], v))
                prev[v] = curId # record ID of previous node

    # build the shortest Path
    def build_path(prev, tragetId, flow_graph):
        # Backtrace from the end to find the Path
        path = []
        flow_property = flow_graph.edge_properties["flow"]
        coverage_property = flow_graph.vertex_properties["coverage"]

        while tragetId != -1:
            path.append(int(tragetId))
            prevId = prev[tragetId]

            if prevId == -1:
                break

            prevNode = flow_graph.vertex(prevId)
            tragetNode = flow_graph.vertex(tragetId)
            curEdge = flow_graph.edge(prevNode,tragetNode,all_edges=True)
            
            # update current edge's flow
            if curEdge is not None:
                # multiple edges
                for edge in curEdge:
                    coverage = coverage_property[prevNode]
                    new_flow = flow_property[edge] + a * coverage
                    flow_property[edge] = new_flow
                        
            tragetId = prevId
        path.reverse()

        return path
    
    if dist[sinkId] == np.inf:
        print(f'No path leads to the endpoint')
    else:
        path = build_path(prev, sinkId, flow_graph)
        print(f'curPath value is {dist[sinkId]}')

    return path,flow_graph


# 修正版dijkstra寻找最大路
def dijkstra_max_path(graph, a):
    flow_graph = copy.deepcopy(graph)
    flow_property = flow_graph.edge_properties["flow"]
    heap = []
    path = {}
    N = flow_graph.num_vertices()
    dist = [-np.inf] * N  # 初始化所有距离为负无穷大
    visited = [False] * N
    prev = [-1] * N
    sourceId = 0
    sinkId = N - 1

    dist[sourceId] = 0
    heap = [(-0, sourceId)]  # 使用负号来让 heapq 作为最大堆使用

    while heap:
        curDist, curId = heapq.heappop(heap)
        curDist = -curDist  # 取负还原实际距离
        if visited[curId]:
            continue
        visited[curId] = True

        curVertex = flow_graph.vertex(curId)
        for e in curVertex.out_edges():
            v = int(e.target())
            weight = flow_property[e]
            if curDist + weight > dist[v]:  # 寻找更大的路径
                dist[v] = curDist + weight
                heapq.heappush(heap, (-(dist[v]), v))  # 使用负号
                prev[v] = curId

    def build_path(prev, targetId, flow_graph):
        path = []
        flow_property = flow_graph.edge_properties["flow"]
        coverage_property = flow_graph.vertex_properties["coverage"]
        
        while targetId != -1:
            path.append(int(targetId))
            prevId = prev[targetId]
            if prevId == -1:
                break
            
            prevNode = flow_graph.vertex(prevId)
            targetNode = flow_graph.vertex(targetId)
            curEdge = flow_graph.edge(prevNode, targetNode, all_edges=True)
            
            if curEdge is not None:
                for edge in curEdge:
                    # coverage = coverage_property[prevNode] # 用丰度作为调参的依据
                    new_flow = max(0.0000001,flow_property[edge] - a * math.exp(flow_property[edge]))
                    flow_property[edge] = new_flow

            targetId = prevId
        path.reverse()

        return path
    
    if dist[sinkId] == -np.inf:
        print('No path leads to the endpoint')
    else:
        path = build_path(prev, sinkId, flow_graph)
        print(f'curPath value is {dist[sinkId]}')

    return path, flow_graph



def find_path(prev, targetID):
    path = []

    while targetID != -1:
        path.append(int(targetID))
        targetID = prev.get(targetID, -1)

    path.reverse()

    return path


# 随机找10条路径
def dfs(flow_graph, vertexID, prev, paths_found, max_paths=10):
    # 到达目标节点并且找到的路径少于10条
    if vertexID == flow_graph.num_vertices() - 1 and len(paths_found) < max_paths:
        path = find_path(prev, vertexID)
        paths_found.append(path)  # 存储找到的路径
        graph_init.export_pathExtraction_polio_fasta(flow_graph, path, len(paths_found))
        return
    
    # 如果已经找到10条路径，则返回
    if len(paths_found) >= max_paths:
        return

    v = flow_graph.vertex(vertexID)
    # 获取所有子节点，并随机打乱顺序以随机化搜索过程
    children = list(v.out_neighbours())
    random.shuffle(children)
    for e in children:
        childID = int(e)
        if childID not in prev:  # 避免循环
            prev[childID] = vertexID
            dfs(flow_graph, childID, prev, paths_found, max_paths)
            del prev[childID]  # 回溯


def validate_random_dfs(graph):      
    # 随机找10条路径然后进行比较相似度
    prev = {}
    paths_found = []
    flow_graph = copy.deepcopy(graph)
    # 开始DFS
    dfs(flow_graph, 0, prev, paths_found)

    validate_data.main()


def linear_search(graph):
    # 线性搜索 a * cov * b * log10(cov)
    for a_values in range(1,11,1):
        for b_values in range(1,11,1):
            flow_graph = copy.deepcopy(graph)

            with open('validate-data.txt','a') as f:
                line = "parameter setting\n"
                line += " ".join(f"a:{a_values}, b:{b_values}\n")
                f.write(line)

            # 重新开一组测试机需要对上一组路径信息清空
            with open("Path_info.txt", "w") as file:
                    file.truncate(0)

            with open("validation-set.fasta", "w") as file:
                    file.truncate(0)

            for i in range(20):
                path,flow_graph = dijkstra(flow_graph,a_values,b_values)
                graph_init.export_flow_info(flow_graph) # 用于观察路径的流量变化（可选注释）
                graph_init.export_path_info(path,i) # 观察路径寻找了什么节点（可选注释）
                graph_init.export_pathExtraction_polio_fasta(flow_graph,path,i) # 需要将路径信息导出才可以验证
                graph_init.export_pathExtraction_polio_txt(flow_graph,path,i) # fasta不易打开所以使用txt（可选注释）
        
            validate_data.main()


def get_random_successor(vertex):
    # 收集所有从给定顶点出发的边
    out_edges = list(vertex.out_edges())
    
    if not out_edges:
        return None  # 没有出边的情况
    
    # 随机选择一个出边
    chosen_edge = random.choice(out_edges)
    
    # 返回选择的出边的目标顶点
    return chosen_edge.target()


def get_successor_throughX(x, f,cur_vertexID):
    print()
    successor = -1
    curi = -1

    for i,j in x[cur_vertexID]:
        max_flag = 0

        if x[cur_vertexID][i,j] == 1 and f[cur_vertexID][i,j] > max_flag:
            successor = j
            curi = i
            max_flag = f[cur_vertexID][i,j]

    if successor == -1:
        print(f"can't find successor, curVertexID:{cur_vertexID}")

    return successor,curi



def path_extraction_X_sc(x, f):
    # 这里仅对6-graph特殊处理
    if f[1][0,3]> f[2][0,3]:
        last_Vertex = 1
    else:
        last_Vertex = 2

    start_Vertex = 3
     
    u = start_Vertex
    path = [0,last_Vertex,u]
    while u != 452:
        child = []
        curi = last_Vertex

        # 寻找下一个结点
        for xi,xj in x[u]:
            if xi == last_Vertex and x[u][xi,xj] == 1:
                child.append(xj)
            
        max_flag= 0
        next_j = -1
        
        # 寻找通路中的流量最大的下一个节点（有多个节点）
        for j in child:
            if f[u][xi, j] > max_flag:
                max_flag = f[u][xi,j]
                next_j = j
        
        # 因为这里的条件太强，导致不存在通路
        if next_j == -1:
            # cur_vertex = id_to_vertex[u]
            # child_Vertex = get_random_successor(cur_vertex) #随机找一条路走
            # next_j = vertex_to_id[child_Vertex]

            # 找u点的通路（一定能找到因为线性约束条件）
            child_VertexID, curi = get_successor_throughX(x,f,u)
            next_j = child_VertexID

        f[u][curi,next_j] = max(0.0001, f[u][curi,next_j] - math.exp(f[u][curi,next_j]) * 0.1)
        if f[u][xi,next_j] == 0.0001:
            print(f"{u} {xi}, {next_j}")

        last_Vertex = u
        u = next_j
        path.append(u)
        # print(f"{u}", end=' ')

    return path

def path_extraction_X(x,f):
     # 这里仅对6-graph特殊处理
    if f[1][0,3]> f[2][0,3]:
        last_Vertex = 1
    else:
        last_Vertex = 2

    start_Vertex = 3
     
    u = start_Vertex
    path = [0,last_Vertex,u]
    while u != 452:
        child = []
        # 寻找下一个结点
        for xi,xj in x[u]:
            if x[u][xi,xj] == 1:
                child.append((xi,xj))
            
        max_flag= 0
        next_j = -1
        curi = 0
        curj = 0


        # 寻找通路中的流量最大的下一个节点（有多个节点）
        for i,j in child:
            if f[u][i, j] > max_flag:
                max_flag = f[u][i,j]
                next_j = j
                curi = i
                curj = j
        
        f[u][curi,curj] = max(0.0001, f[u][curi,next_j] - math.exp(f[u][curi,next_j]) * 0.2)
        if f[u][xi,next_j] == 0.0001:
            print(f"{u} {xi}, {next_j}")

        last_Vertex = u
        u = next_j
        path.append(u)
        # print(f"{u}", end=' ')

    return path


def op_lp():
    file_path = '6-graph.fasta'
    coverage_file_path = '6-coverage.txt'
    graph,id_to_vertex,vertex_to_id = graph_init.build_graph_from_fasta_new(file_path)
    graph_init.update_coverage_new(graph, id_to_vertex, coverage_file_path)
    graph_init.read_edgeWeight_new(graph, id_to_vertex, "edgeWeight-graph.txt")
    # 构造边权
    # filepaths = ["6Polio-reads/6Polio1.fasta", "6Polio-reads/6Polio2.fasta"]
    # fasta_data = graph_init.load_fasta_data(filepaths)  # 在程序开始时预加载数据
    # graph_init.edge_weight_multiprocess(graph,fasta_data)
    x, f = optimize_flow_as_min_paths(graph, vertex_to_id, flow_slack_tolerance=0.1, conservation_slack_tolerance=0.1)
    #clean file content
    # 打开文件并截断其内容
    with open("validation-set.txt", "w") as file:
        file.truncate(0)

    with open("validation-set.fasta", "w") as file:
        file.truncate(0)

    with open("Path_info.txt", "w") as file:
        file.truncate(0)

    with open("validate-data.txt", "w") as file:
        file.truncate(0)

    with open("flow_info.txt", "w") as file:
        file.truncate(0)

    # 注意这里不能轻易打开否者会将保存下来的weight权值清空
    # with open("edgeWeight-graph.txt", "w") as file:
    #     file.truncate(0)

    # 优化过的线性规划
    for i in range(0,15):
        flow_graph = copy.deepcopy(graph)
        # 强条件
        # path = path_extraction_X_sc(x,f)  
        # 弱条件
        path = path_extraction_X(x,f)
        graph_init.export_pathExtraction_polio_fasta(flow_graph,path,num=i)
    
    


def lp():
    file_path = '6-graph.fasta'
    coverage_file_path = '6-coverage.txt'
    graph,id_to_vertex,vertex_to_id = graph_init.build_graph_from_fasta_new(file_path)
    graph_init.update_coverage_new(graph, id_to_vertex, coverage_file_path)

    # 线性规划
    # optimize_flow_as_abs(graph)
    optimize_flow_as_qp(graph)


    #clean file content
    # 打开文件并截断其内容
    with open("validation-set.txt", "w") as file:
        file.truncate(0)

    with open("validation-set.fasta", "w") as file:
        file.truncate(0)

    with open("Path_info.txt", "w") as file:
        file.truncate(0)

    with open("validate-data.txt", "w") as file:
        file.truncate(0)

    with open("flow_info.txt", "w") as file:
        file.truncate(0)

    graph_init.edge_flow_normalizetion(graph)
    
    graph_init.export_flow_info(graph)

    for i in range(15):
        path,graph = dijkstra_max_path(graph,0.27)
        graph_init.export_pathExtraction_polio_fasta(graph,path,i) # 需要将路径信息导出才可以验证
    


if __name__ == "__main__":
    lp()


