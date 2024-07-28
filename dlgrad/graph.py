import networkx as nx
from dlgrad.helpers import get_graph, BinaryOps, UnaryOps, BufferOps


class Graph:
    def __init__(self):
        self.G = nx.DiGraph() if get_graph() else None
        self.ops_colour = {}

    def add_node(self, label: str):
        if self.G is not None:
            self.G.add_node(label)

    def add_edge(self, source: str, target: str):
        if self.G is not None:
            self.G.add_edge(source, target)

graph_instance = Graph()

add_node = graph_instance.add_node
add_edge = graph_instance.add_edge
