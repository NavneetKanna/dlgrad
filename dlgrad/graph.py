import atexit
import os

import networkx as nx
from networkx.drawing.nx_pydot import write_dot

from dlgrad.helpers import get_graph


class Graph:
    def __init__(self):
        self.G = nx.DiGraph() if get_graph() else None
        self.ops_colour = {
            "BinaryOps": "#fd7f6f",
            "UnaryOps": "#bd7ebe",
            "BufferOps": "#7eb0d5",
        }
        self.id = 0

        if get_graph():
            atexit.register(self.save_graph)

    def create_id(self):
        self.id += 1
        return self.id

    def add_node(self, node):
        if self.G is not None:
            if node.properties.metadata["node_id"] is None:
                node.properties.metadata["node_id"] = self.create_id()
            else:
                return

            # for broadcast
            if node.properties.metadata["ops"] is None:
                label = f"{node.shape}\nBROADCAST"
                colour = "#b2e061"
                style = "filled, dashed"
            else:
                label = f"{node.shape}\n{node.properties.metadata['ops']}\n{node.properties.metadata['created_by']}"
                colour = self.ops_colour[node.properties.metadata["ops"]]
                style = "filled, bold"

            self.G.add_node(
                node.properties.metadata["node_id"],
                label=label,
                fillcolor=colour,
                color="black",
                style=style,
            )

    def add_edge(self, child, parents: tuple):
        if self.G is not None:
            self.add_node(child)
            for p in parents:
                self.add_node(p)
                self.G.add_edge(
                    p.properties.metadata["node_id"],
                    child.properties.metadata["node_id"],
                )
                # for broadcast
                if child.properties.metadata["ops"] is None:
                    p.properties.metadata["node_id"] = child.properties.metadata["node_id"]

    def save_graph(self):
        print("Saving graph /tmp/graph.svg")
        write_dot(self.G, "/tmp/file.dot")
        os.system("dot -Tsvg /tmp/file.dot -o /tmp/graph.svg")
        print("Done")


graph_instance = Graph()

add_node = graph_instance.add_node
add_edge = graph_instance.add_edge
