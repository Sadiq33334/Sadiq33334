import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Distance matrix (using infinity for non-existent paths)
INF = float('inf')
distance_matrix = np.array([
    [INF, INF, INF, 1, 1],
    [INF, INF, 3, INF, 8],
    [INF, 3, INF, 6, 2],
    [1, INF, 6, INF, 7],
    [1, 8, 2, 7, INF]
])

# Parameters
HMS = 3  # Harmony memory size
HMCR = 0.5  # Harmony memory consideration rate
PAR = 0.3  # Pitch adjustment rate
NI = 400  # Total number of iterations

# Helper function to calculate the total distance of a route
def calculate_distance(route, distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        if distance_matrix[route[i], route[i + 1]] == INF:
            return INF 
        total_distance += distance_matrix[route[i], route[i + 1]]
    return total_distance

# Initialize harmony memory with random routes that visit each city exactly once
def initialize_harmony_memory(HMS, num_cities):
    harmony_memory = []
    for _ in range(HMS):
        while True:
            route = list(range(num_cities))
            random.shuffle(route)
            distance = calculate_distance(route, distance_matrix)
            if distance!= INF and len(set(route)) == num_cities:
                harmony_memory.append((route, distance))
                break
    return harmony_memory

# Improvisation of a new harmony (route) ensuring all cities are visited exactly once
def improvise_new_harmony(harmony_memory, num_cities):
    new_route = []
    visited = set()
    
    for i in range(num_cities):
        if random.random() < HMCR and harmony_memory:  # Harmony memory consideration
            selected_route = random.choice(harmony_memory)[0]
            if selected_route[i] not in visited:
                new_route.append(selected_route[i])
                visited.add(selected_route[i])
        else:
            # Choose a random unvisited city
            remaining_cities = list(set(range(num_cities)) - visited)
            if remaining_cities:
                new_route.append(random.choice(remaining_cities))
                visited.add(new_route[-1])
    
    # Ensure the route is complete
    remaining_cities = list(set(range(num_cities)) - visited)
    new_route.extend(remaining_cities)
    
    # Pitch adjustment
    if random.random() < PAR and len(new_route) == num_cities:
        i, j = random.sample(range(num_cities), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
    
    return new_route

# Harmony Search Algorithm
def harmony_search(HMS, HMCR, PAR, NI, distance_matrix):
    num_cities = distance_matrix.shape[0]
    initial_routes = initialize_harmony_memory(HMS, num_cities)
    all_solutions = []

    print("Initial Harmony Memory:")
    for i, (route, distance) in enumerate(initial_routes):
        city_names = [chr(ord('A') + city) for city in route]  
        print(f"Route {i+1}: {city_names}, Distance: {distance}")

    iterations_per_route = NI // HMS

    for initial_index in range(HMS):
        harmony_memory = [initial_routes[initial_index]]
        best_solutions = []

        for iteration in range(iterations_per_route):
            new_route = improvise_new_harmony(harmony_memory, num_cities)
            new_distance = calculate_distance(new_route, distance_matrix)
            if new_distance < max(harmony_memory, key=lambda x: x[1])[1]:
                harmony_memory[harmony_memory.index(max(harmony_memory, key=lambda x: x[1]))] = (new_route, new_distance)
            best_solution = min(harmony_memory, key=lambda x: x[1])
            best_solutions.append(best_solution)
            
            # Print each iteration's result
            city_names = [chr(ord('A') + city) for city in best_solution[0]]
            print(f"Initial Route {initial_index + 1}, Iteration {iteration + 1}: {city_names}, Distance: {best_solution[1]}")
        
        all_solutions.append(best_solutions)

        best_route = best_solution[0]
        city_names = [chr(ord('A') + city) for city in best_route]  
        print(f"Route {initial_index + 1} Final Best Route: {city_names}, Distance: {best_solution[1]}")

        # Plot the best routes
    for i, best_solutions in enumerate(all_solutions):
        distances = [solution[1] for solution in best_solutions]
        plt.plot(range(len(distances)), distances, label=f"Route {i+1}")
    plt.xlabel("Iteration")
    plt.ylabel("Distance")
    plt.title("Harmony Search Algorithm")
    plt.legend()
    plt.show()

    # Create a graph with NetworkX
    G = nx.DiGraph()
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[0]):
            if distance_matrix[i, j]!= INF:
                G.add_edge(i, j, weight=distance_matrix[i, j])

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

harmony_search(HMS, HMCR, PAR, NI, distance_matrix)

