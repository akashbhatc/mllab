def heuristic(node):
    heuristic_values = {'A': 3, 'B': 4, 'C': 2, 'D': 6, 'G': 0, 'S': 5}
    return heuristic_values[node]

def a_star_algorithm(graph, start_node, goal_node):

    open_list = [start_node]
    closed_list = set()

    g_costs = {start_node: 0}

    parents = {start_node: start_node}

    while open_list:

        open_list.sort(key=lambda node: g_costs[node] + heuristic(node), reverse=True)
        current_node = open_list.pop()
        
        if current_node == goal_node:
            path = []

            while parents[current_node] != current_node:
                path.append(current_node)
                current_node = parents[current_node]

            path.append(start_node)
            path.reverse()

            print(f'Path found: {path}')
            return path

        for (neighbor, weight) in graph[current_node]:
            if neighbor not in open_list and neighbor not in closed_list:
                open_list.append(neighbor)
                parents[neighbor] = current_node
                g_costs[neighbor] = g_costs[current_node] + weight
            else:
                if g_costs[neighbor] > g_costs[current_node] + weight:
                    g_costs[neighbor] = g_costs[current_node] + weight
                    parents[neighbor] = current_node

                    if neighbor in closed_list:
                        closed_list.remove(neighbor)
                        open_list.append(neighbor)
                        
        closed_list.add(current_node)

    print('Path does not exist!')
    return None

graph = {
    'S': [('A', 1), ('G', 10)],
    'A': [('B', 2), ('C', 1)],
    'B': [('D', 5)],
    'C': [('D', 3), ('G', 4)],
    'D': [('G', 2)]
}

a_star_algorithm(graph, 'S', 'G')
