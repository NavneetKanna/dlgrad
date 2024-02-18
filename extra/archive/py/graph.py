"""
import graphviz 

nodes = [] 
edges = []
output_list: list[tuple] = []
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
                if input1_label is None:
                    s.node(str(idx), f'<f0> {input2}')
                else:
                    s.node(str(idx), f'<f0> {input2_label}|<f1> {input2}')

                draw_edge_2_op(str(idx), str(idx-1))
                idx += 1
            else:
                # Otherwise it will draw edge from input of the UnaryOps
                idx += 1
            
            if input1_label is None:
                s.node(str(idx), f'<f0> {output}')
            else:
                s.node(str(idx), f'<f0> {output_label}|<f1> {output}')

            output_list.append((output, str(idx)))
            draw_edge_op_output(str(idx-2), str(idx))
            idx += 1
        else:
            draw_edge_2_op(output_list[-1][-1], str(idx-1))

            if input1_label is None:
                s.node(str(idx), f'<f0> {input1}')
            else:
                s.node(str(idx), f'<f0> {input1_label}|<f1> {input1}')

            draw_edge_1_op(str(idx), str(idx-1))
            idx += 1

            if input1_label is None:
                s.node(str(idx), f'<f0> {output}')
            else:
                s.node(str(idx), f'<f0> {output_label}|<f1> {output}')

            output_list.append((output, str(idx)))
            draw_edge_op_output(str(idx-2), str(idx))
            idx += 1
    else:
        s.node(str(idx), op)
        idx += 1

        input1, input1_label = (*input1,)
        if input1_label is None:
            s.node(str(idx), f'<f0> {input1}')
        else:
            s.node(str(idx), f'<f0> {input1_label}|<f1> {input1}')
        draw_edge_1_op(str(idx), str(idx-1))
        idx += 1

        input2, input2_label = (*input2,)
        if input2_label is None:
            s.node(str(idx), f'<f0> {input2}')
        else:
            s.node(str(idx), f'<f0> {input2_label}|<f1> {input2}')
        draw_edge_2_op(str(idx), str(idx-2))
        idx += 1

        output, output_label = (*output,)
        output_list.append((output, str(idx)))
        if output_label is None:
            s.node(str(idx), f'<f0> {output}')
        else:
            s.node(str(idx), f'<f0> {output_label}|<f1> {output}')
        draw_edge_op_output(str(idx-3), str(idx))
        idx += 1

def save_graph(): 
    s.edges(edges)
    s.render('graph', directory='/mnt/c/Users/navne/Documents/vs_code/dlgrad/notes')
    
import numpy as np
class CG:
    idx = 1
    stop_processing = False

    @classmethod
    def add_nodes(cls, op: str, output: np.ndarray, input1: np.ndarray, input2: np.ndarray=None):

        # (128, 64) = (128, 784) @ (64, 784)
        if op == 'matmul':
            draw_graph('matmul', (('BS', output.shape[1]), None), (input2.shape, f'fc{CG.idx}.weight'), (('BS', input1.shape[1]), None),)
        
        # (128, 64) = (128, 64) + (1, 64)
        elif op == 'add':
            draw_graph('add', (('BS', output.shape[1]), None), (input2.shape, f'fc{CG.idx}.bias')), (('BS', input1.shape[1]), None)

        # (128, 64) = (128, 64)
        elif op == 'ReLU':
            draw_graph('ReLU', (('BS', output.shape[1]), None), (('BS', input1.shape[1]), None))
            CG.idx += 1
            
        # () = (128, 10)
        elif op == 'loss':
            draw_graph('Cross-Entropy Loss', (output.shape, 'Loss'), (('BS', input1.shape[1]), 'Predictions'))
            CG.stop_processing = True
"""