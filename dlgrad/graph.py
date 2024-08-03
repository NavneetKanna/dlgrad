import networkx as nx
from networkx.drawing.nx_pydot import write_dot

import os
import atexit

from dlgrad.helpers import BinaryOps, BufferOps, UnaryOps, get_graph


class Graph:
    def __init__(self):
        self.G = nx.DiGraph() if get_graph() else None
        self.ops_colour = {BinaryOps: '#e74c3c', UnaryOps: '#1abc9c', BufferOps: '#f1c40f'}
        self.id = 0
        self.nodes = []

        atexit.register(self.save_graph)

    def create_id(self):
        self.id += 1
        return self.id 

    def add_node(self, node):
        if self.G is not None:
            if node.properties.metadata['node_id'] is None:
                node.properties.metadata['node_id'] = self.create_id()
            else: 
                return
            
            # for broadcast
            if node.properties.metadata['ops'] is None:
                label = "BROADCAST"
                colour = '#51f542'
            else:
                label = f"{node.properties.metadata['ops']}\n{node.properties.metadata['created_by']}"
                colour = self.ops_colour[node.properties.metadata['ops']]

            self.G.add_node(
                node.properties.metadata['node_id'], 
                label=label, 
                fillcolor=colour, 
                color="black",  
                style="filled, bold"
            )

    def add_edge(self, child, parents: tuple):
        if self.G is not None:
            self.add_node(child)
            for p in parents:
                self.add_node(p)
                self.G.add_edge(p.properties.metadata['node_id'], child.properties.metadata['node_id'])
                # for broadcast
                if child.properties.metadata['ops'] is None:
                    p.properties.metadata['node_id'] = child.properties.metadata['node_id']


    def save_graph(self):
        print("Saving graph to /tmp")
        write_dot(self.G, '/tmp/file.dot')
        os.system('dot -Tsvg /tmp/file.dot -o /tmp/graph.svg')
        print("Done")

graph_instance = Graph()

add_node = graph_instance.add_node
add_edge = graph_instance.add_edge
