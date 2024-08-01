import networkx as nx

from dlgrad.helpers import BinaryOps, BufferOps, UnaryOps, get_graph


class Graph:
    def __init__(self):
        self.G = nx.DiGraph() if get_graph() else None
        self.ops_colour = {BinaryOps: '#e74c3c', UnaryOps: '#1abc9c', BufferOps: '#f1c40f'}
        self.id = 0

    def get_create_id(self):
        self.id += 1
        return self.id 

    def add_node(self, label: str, ops):
        if self.G is not None:
            self.G.add_node(
                next(self.create_id_iter), 
                label, 
                fillcolor=self.ops_colour[ops], 
                color="black",  
                style="filled, bold"
            )

    def add_edge(self, source: str, target: str):
        if self.G is not None:
            self.G.add_edge(source, target)

graph_instance = Graph()

add_node = graph_instance.add_node
add_edge = graph_instance.add_edge
