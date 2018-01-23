from itertools import chain
from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class Neuron:
    activation_functions = {
        'linear': lambda x: x,
        'tanh': lambda x: np.tanh(np.radians(x)),
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        #'relu': lambda x: np.max(0., x),
        'softplus': lambda x: np.log(1 + np.exp(x)),
        'sin': lambda x: np.sin(np.radians(x)),
        'cos': lambda x: np.cos(np.radians(x))
    }
    def __init__(self, name, activation_function='linear'):
        if activation_function not in Neuron.activation_functions:
            raise ValueError("The provided activation function is not valid.")
            
        self.name = name
        self.activation_function = Neuron.activation_functions[activation_function]

class Connection:
    def __init__(self, neuron_from, neuron_to, weight, 
                 enabled=True, innovation_number=0):
        self.neuron_from = neuron_from
        self.neuron_to = neuron_to
        self.weight = weight
        self.enabled = enabled
        self.innovation_number = innovation_number

class Network:
    innovation_counter = 0

    def __init__(self, inputs, outputs, connections, hidden_nodes=None):
        self.inputs = inputs
        self.outputs = outputs
        self.connections = []
        self.hidden_nodes = hidden_nodes
        
        # Get all unique nodes in the network
        if hidden_nodes is None:
            to_nodes = [x.neuron_from for x in connections]
            from_nodes = [x.neuron_to for x in connections]
            other_nodes = to_nodes + from_nodes
        else:
            other_nodes = hidden_nodes
        all_nodes = set(inputs + outputs + other_nodes)
        self.neurons = list(all_nodes)
        
        # Create adjacency dict of dicts & neighbor dict of lists
        self.adjacency = {}
        self.neighbors = {}
        self.neighbors_rev = {}
        for neuron in self.neurons:
            self._update_neuron(neuron)
        
        for connection in connections:
            self.add_connection(connection)

    def __deepcopy__(self, memo):
        return Network(
            deepcopy(self.inputs, memo),
            deepcopy(self.outputs, memo),
            deepcopy(self.connections, memo),
            deepcopy(self.hidden_nodes, memo)
        )
          
    def has_connection(self, neuron_from, neuron_to):
        return (
            self.adjacency[neuron_from][neuron_to] is not None
            and self.adjacency[neuron_from][neuron_to].enabled
        )
    
    def get_neighbors(self, neuron_from):
        return filter(
            lambda x: self.has_connection(neuron_from, x), 
            self.neighbors[neuron_from]
        )
    
    def get_neurons_to(self, neuron_to):
        return filter(
            lambda x: self.has_connection(x, neuron_to), 
            self.neighbors_rev[neuron_to]
        )

    def add_neuron(self, neuron):
        self.neurons.append(neuron)
        self._update_neuron(neuron)

    def add_connection(self, connection, innovation_number=None):
        if not self.check_cycle(connection):
            if innovation_number is None and connection.innovation_number == 0:
                connection.innovation_number = Network.innovation_counter
                Network.innovation_counter += 1
            else:
                connection.innovation_number = [
                    innovation_number, connection.innovation_number
                ][innovation_number is None]
            self.connections.append(connection)
            self._update_connection(connection)

    def _update_connection(self, connection):
        self.adjacency[connection.neuron_from][connection.neuron_to] = connection
        self.neighbors[connection.neuron_from].append(connection.neuron_to)
        self.neighbors_rev[connection.neuron_to].append(connection.neuron_from)

    def _update_neuron(self, neuron):
        self.adjacency[neuron] = defaultdict(lambda: None)
        self.neighbors[neuron] = []
        self.neighbors_rev[neuron] = []

    def check_cycle(self, connection):
        """Check if appending connection results in a cycle, this
        occurs when we meet `connection.neuron_from` twice"""
        _from, _to = connection.neuron_from, connection.neuron_to

        # Do not add a connection from a neuron to itself
        if _from == _to:
            return True

        visited, t = {_to}, {_to}
        while len(t) > 0:
            # Get all neighbors from our the newly explored ones
            # Filter out the ones that are already in `visited`
            # and enabled
            t = list(filter(
                lambda x: x not in visited, 
                set(y for x in t for y in self.get_neighbors(x)
                    if self.adjacency[x][y].enabled)
            ))
            visited = visited.union(t)

            if _from in visited:
                # We meet `connection.neuron_from` twice
                return True
        return False

    def visualize(self):
        g = nx.DiGraph()
        # Draw our graph
        n_colors = []
        for n in sorted(self.neurons, key=lambda x: x.name):
            g.add_node(n.name)
            if n in self.inputs:
                n_colors.append('purple')
            elif n in self.outputs:
                n_colors.append('yellow')
            else:
                n_colors.append('blue')
        
        e_colors = []
        e_widths = []
        for conn in self.connections:
            if conn.enabled:
                w = g.add_edge(conn.neuron_from.name, conn.neuron_to.name, weight=conn.weight)
                e_widths.append(abs(conn.weight) * 5)
                e_colors.append(['red', 'green'][conn.weight > 0])

        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(g)
        nx.draw_networkx_nodes(g, pos, node_color=n_colors)
        nx.draw_networkx_labels(g, pos)
        nx.draw_networkx_edges(g, pos, width=e_widths, edge_color=e_colors)
        plt.show()
        
    def required_for_output(self):
        # Keep track of all nodes going to the outputs
        req = set(self.outputs)
        t = req

        while t:  # While we can find new nodes
            # Find new nodes not in req with an edge to a node in req
            # TODO: replace with filter()
            t = set(y for x in req for y in self.get_neurons_to(x) if y not in req)

            # Remove the input nodes
            t -= set(self.inputs)

            # Update nodes that go to the outputs
            req = [req, req.union(t)][t is not None]

        return req


    def feed_forward_layers(self):
        # Get the nodes that eventually lead to an output node
        required = self.required_for_output()
        t = set(self.inputs)

        layers = [set(self.inputs)]
        while t:
            # Get all nodes from layers (flatten the list)
            s = set(chain(*layers))

            # Find new candidates in neighbors of already found nodes
            # TODO: replace with filter()
            c = set(y for x in s for y in self.get_neighbors(x) if y not in s)
            
            # Discard unrequired nodes
            c = c.intersection(set(required))

            # Keep only the used nodes whose entire input set is contained in s.
            # TODO: replace with filter()
            t = set(n for n in c if all(a in s for a in self.get_neurons_to(n)))
            
            # Append the new layer, if not empty
            if len(t): layers.append(t)

        # Remove the inputs
        layers.remove(set(self.inputs))
        return layers
        
    def forward_pass(self, feature_vector):
        layers = self.feed_forward_layers()
        if len(layers) == 0:
            return [0.]*len(self.outputs)

        values = {}
        for input_neuron, value in zip(self.inputs, feature_vector):
            values[input_neuron] = value
            
        for layer in layers:
            for node in layer:
                values[node] = 0
                for rev_neighbor in self.get_neurons_to(node):
                    values[node] += values[rev_neighbor] * self.adjacency[rev_neighbor][node].weight
                values[node] = node.activation_function(values[node])
                
        return [values[x] for x in self.outputs]

    def genome_str(self):
        for conn in self.connections:
            print('{} --> {}'.format(conn.neuron_from.name, conn.neuron_to.name))
            print('enabled={}'.format(conn.enabled))
            print('weight={}'.format(conn.weight))
            print('innovation={}'.format(conn.innovation_number))
            print('-'*50)