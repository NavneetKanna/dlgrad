import networkx as nx
import pydot
import os
nodes = {}
edges = []
nodess = []
labels = {}
idx = 0
# A better way to write this ??
# Should we check object or shape ??

# One of the input can be none such as in activation functions
def add_nodes(op, output, input1, input2=None):
    if op == 'Relu':
        op = f"a{idx}"
    nodess.append(op)
    
    # Check if the output of the previous and input1/input2 shape is the same 
    print("-------------------------------")
    print("nodes")
    print(nodes)
    print(f"output {output}")
    print(f"input1 {input1}")
    print(f"input2 {input2}")
    if nodes:
        print(f"inside nodes")
        print("last item")
        print(nodes[list(nodes.keys())[-1]][-1])
        if nodes[list(nodes.keys())[-1]][-1] in [input1, input2]:
            print("last item matches")
            if input2:
                print("if input2")
                nodes[op] = [input1, input2, output]
                print("nodes")
                print(nodes)
            else:
                print("if not input2")
                nodes[op] = [input1, output]
                print("nodes")
                print(nodes)
    else:
        print("if not nodes")
        if input2:
            print("if input2")
            nodes[op] = [input1, input2, output]
            print("nodes")
            print(nodes)
        else:
            nodes[op] = [input1, output]
    

def draw_cg():
    G = nx.DiGraph()
    for i in nodes:
        print(f"i {i}")
        print(f"nodes[i] {nodes[i]}")
        G.add_node(i)
        for j in nodes[i][:-1]:
            edges.append((j, i))
       
        edges.append((i, nodes[i][-1]))

        # print(nodes[i][-1])
        # G.add_nodes_from(nodess)
       

    G.add_edges_from(edges) 
    nx.drawing.nx_pydot.write_dot(G, r'/mnt/c/Users/navne/Documents/vs_code/dlgrad_main/dlgrad/net.dot')
    # (graph,) = pydot.graph_from_dot_file(r'dlgrad/net.dot')
    (graph,) = pydot.graph_from_dot_file(r'/mnt/c/Users/navne/Documents/vs_code/dlgrad_main/dlgrad/net.dot')
    graph.write_png(r'/mnt/c/Users/navne/Documents/vs_code/dlgrad_main/dlgrad/net.png')

