import numpy as np
from nn import Network, Neuron, Connection
from copy import deepcopy


def fitness(network):
    """Propagate our feature vectors through the network, get the prediction and
    measure the MSE"""
    x = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
    y = [   0.,       1.,       1.,       0.]
    return -sum([(network.forward_pass(feature_vector)[0] - label) ** 2 for feature_vector, label in zip(x, y)])

def connection_weight_mutate(network, sigma=0.1):
    """Take a random connection from the network and add noise to the weight"""
    rand_conn = np.random.choice(network.connections)
    rand_conn.weight += np.random.normal(0, sigma)
    
def introduce_connection(network, sigma=0.5, innovations={}):
    """Take two random neurons and introduce a connection with random weight"""
    rand_node1 = np.random.choice(list(set(network.neurons) - set(network.outputs)))
    rand_node2 = np.random.choice(list(set(network.neurons) - set(network.inputs)))
    if network.adjacency[rand_node1][rand_node2] is None:
        conn = Connection(rand_node1, rand_node2, np.random.normal(0, sigma))
        if (rand_node1.name, rand_node2.name) not in innovations:
            network.add_connection(conn)
            innovations[(rand_node1.name, rand_node2.name)] = conn.innovation_number
        else:
            network.add_connection(conn, innovation_number=innovations[(rand_node1.name, rand_node2.name)])
    else:
        network.adjacency[rand_node1][rand_node2].enabled = True
    
    # TODO: check if we do not create a cycle!!!

def introduce_node(network, sigma=1, innovations={}):
    """Take a random connection from the network (A -> B) and introduce a new node
    C with connections: A -> C -> B"""
    # Take a random connection and disable it, we are introducing two new ones
    if len(network.connections) == 0: 
        return

    rand_conn = np.random.choice(network.connections)
    rand_conn.enabled = False
    
    # Create a new neuron
    new_neuron = Neuron(len(network.neurons) + 1)
    network.add_neuron(new_neuron)
    
    # Create two new connections
    new_conn1 = Connection(rand_conn.neuron_from, new_neuron, 1)
    # Take the weight of the disabled connection
    new_conn2 = Connection(new_neuron, rand_conn.neuron_to, rand_conn.weight)

    if (rand_conn.neuron_from.name, rand_conn.neuron_to.name) not in innovations:
        network.add_connection(new_conn1)
        network.add_connection(new_conn2)
        innovations[(rand_conn.neuron_from.name, rand_conn.neuron_to.name)] = (new_conn1.innovation_number, new_conn2.innovation_number)
    else:
        in_nr1, in_nr2 = innovations[(rand_conn.neuron_from.name, rand_conn.neuron_to.name)]
        network.add_connection(new_conn1, innovation_number=in_nr1)
        network.add_connection(new_conn2, innovation_number=in_nr2)

    
def flip_connection(network):
    """Take a random connection and flip the enabled state"""
    rand_conn = np.random.choice(network.connections)
    if rand_conn.enabled:
        rand_conn.enabled = False
    else:
        if not network.check_cycle(rand_conn):
            rand_conn.enabled = True
            
    rand_conn.enabled = not rand_conn.enabled
    
def change_activation(network):
    """Take a random neuron and change its activation function"""
    rand_node = np.random.choice(network.neurons)
    rand_act_function = np.random.choice(list(Neuron.activation_functions.keys()))
    rand_node.activation_function = Neuron.activation_functions[rand_act_function]
    
def cross_over(network1, network2):
    matching_connection_innov_nrs = list(
        set([x.innovation_number for x in network1.connections])
            .intersection(set([x.innovation_number for x in network2.connections]))
    )

    if len(matching_connection_innov_nrs) > 0:
        min_innov_nr = min(matching_connection_innov_nrs)
        max_innov_nr = max(matching_connection_innov_nrs)
    else:
        min_innov_nr = max_innov_nr = 0
    
    match_conn_p1 = []
    disjoint_conn_p1 = []
    excess_conn_p1 = []
    
    for conn in network1.connections:
        if conn.innovation_number in matching_connection_innov_nrs:
            match_conn_p1.append(conn)
        else:
            if min_innov_nr < conn.innovation_number < max_innov_nr:
                disjoint_conn_p1.append(conn)
            else:
                excess_conn_p1.append(conn)

    match_conn_p1 = sorted(match_conn_p1, key=lambda x: x.innovation_number)
    disjoint_conn_p1 = sorted(disjoint_conn_p1, key=lambda x: x.innovation_number)
    excess_conn_p1 = sorted(excess_conn_p1, key=lambda x: x.innovation_number)

    match_conn_p2 = []
    disjoint_conn_p2 = []
    excess_conn_p2 = []
    
    for conn in network2.connections:
        if conn.innovation_number in matching_connection_innov_nrs:
            match_conn_p2.append(conn)
        else:
            if min_innov_nr < conn.innovation_number < max_innov_nr:
                disjoint_conn_p2.append(conn)
            else:
                excess_conn_p2.append(conn)

    match_conn_p2 = sorted(match_conn_p2, key=lambda x: x.innovation_number)
    disjoint_conn_p2 = sorted(disjoint_conn_p2, key=lambda x: x.innovation_number)
    excess_conn_p2 = sorted(excess_conn_p2, key=lambda x: x.innovation_number)

    new_conn = []
    for conn1, conn2 in zip(match_conn_p1, match_conn_p2):
        new_conn.append(np.random.choice([conn1, conn2]))

    if fitness(network1) > fitness(network2):
        new_conn += disjoint_conn_p1 + excess_conn_p1
    elif fitness(network2) > fitness(network1):
        new_conn += disjoint_conn_p2 + excess_conn_p2
    else:
        new_conn += disjoint_conn_p1 + excess_conn_p1 + disjoint_conn_p2 + excess_conn_p2

    return Network(network1.inputs, network1.outputs, new_conn)


def genetic(inputs, outputs, population_size=3, n_generations=5, mutate_add_node=0.5,
            mutate_add_connection=0.5, mutate_weights=0.1, mutate_flip_connection=0.1,
            mutate_change_activation=0.1):
    population = []
    for _ in range(population_size):
        population.append(Network(inputs, outputs, []))

    connection_innovations = {}
    node_innovations = {}
    for gen in range(n_generations):
        print('Generation {}'.format(gen))
        print('='*50)
        for i in range(len(population)):
            print('Individual {}:'.format(i))
            population[i].genome_str()

        for i in range(population_size):
            if np.random.rand() < mutate_add_connection:
                introduce_connection(population[i], innovations=connection_innovations)
            if np.random.rand() < mutate_add_node:
                introduce_node(population[i], innovations=node_innovations)

        population.append(cross_over(population[0], population[1]))



inputs = [Neuron(1), Neuron(2)]
outputs = [Neuron(3)]
connections = []

genetic(inputs, outputs)