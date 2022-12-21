import networkx as nx
import graphviz 
import pydot
import os
nodes = [] 
edges = []
output_list = []
idx = 0
s = graphviz.Digraph(node_attr={'shape': 'record'}, format='png')

# A better way to write this ??
# Should we check object or shape ??



def draw_edge_1_op(input: str, op: str):
    edges.append((input, op))

def draw_edge_2_op(input: str, op: str):
    edges.append((input, op))

def draw_edge_op_output(op: str, output: str):
    edges.append((op, output))

# input2 can be none for UnaryOps 
def draw_graph(op: str, output: tuple, input1: tuple, input2: tuple=None):
    global idx

    # We need to connect the previous output to the current op
    if output_list:
        input1, input1_label = (*input1,)
        if input2: input2, input2_label = (*input2,)
        output, output_label = (*output,)

        s.node(str(idx), op)
        idx += 1

        # Since either input1 or input2 is the same as the last output
        # The code will change slightly because of the labelling 
        if output_list[-1][0] == input1:
            draw_edge_1_op(output_list[-1][-1], str(idx-1))

            if input2:
                s.node(str(idx), f'<f0> {input2_label}|<f1> {input2}')
                draw_edge_2_op(str(idx), str(idx-1))
                idx += 1
            else:
                # Otherwise it will draw edge from input of the UnaryOps
                idx += 1

            s.node(str(idx), f'<f0> {output_label}|<f1> {output}')
            output_list.append((output, str(idx)))
            draw_edge_op_output(str(idx-2), str(idx))
            idx += 1
        else:
            draw_edge_2_op(output_list[-1][-1], str(idx-1))

            s.node(str(idx), f'<f0> {input1_label}|<f1> {input1}')
            draw_edge_1_op(str(idx), str(idx-1))
            idx += 1

            s.node(str(idx), f'<f0> {output_label}|<f1> {output}')
            output_list.append((output, str(idx)))
            draw_edge_op_output(str(idx-2), str(idx))
            idx += 1
    else:
        s.node(str(idx), op)
        idx += 1

        input1, input1_label = (*input1,)
        s.node(str(idx), f'<f0> {input1_label}|<f1> {input1}')
        draw_edge_1_op(str(idx), str(idx-1))
        idx += 1

        input2, input2_label = (*input2,)
        s.node(str(idx), f'<f0> {input2_label}|<f1> {input2}')
        draw_edge_2_op(str(idx), str(idx-2))
        idx += 1

        output, output_label = (*output,)
        output_list.append((output, str(idx)))
        s.node(str(idx), f'<f0> {output_label}|<f1> {output}')
        draw_edge_op_output(str(idx-3), str(idx))
        idx += 1

def display_graph(): 
    s.edges(edges)
    s.render('graph', directory='/mnt/c/Users/navne/Documents/vs_code/dlgrad_main/dlgrad')
   
