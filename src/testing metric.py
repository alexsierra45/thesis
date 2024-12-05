import random
from scipy.optimize import linear_sum_assignment
import numpy as np

# Generar el grafo bipartito completo con pesos aleatorios
def generate_graph(n, m):
    graph = [[random.random() for _ in range(m)] for _ in range(n)]
    return graph

# Resolver el problema con el algoritmo de Kuhn-Munkres (Hungarian Algorithm)
def solve_with_kuhn_munkres(graph):
    cost_matrix = -np.array(graph)  # Convertimos a negativo porque scipy resuelve el problema de minimización
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    max_weight = -cost_matrix[row_ind, col_ind].sum()
    return max_weight

# Resolver el problema con el algoritmo codicioso proporcionado
def solve_with_greedy(graph):
    n, m = len(graph), len(graph[0])
    edges = []
    # Crear una lista de todas las aristas con sus pesos
    for i in range(n):
        for j in range(m):
            edges.append((i, j, graph[i][j]))
    # Ordenar las aristas por peso de mayor a menor
    edges.sort(key=lambda x: x[2], reverse=True)
    
    # Emparejar de forma codiciosa
    matched_rows = set()
    matched_cols = set()
    max_weight = 0
    for u, v, weight in edges:
        if u not in matched_rows and v not in matched_cols:
            matched_rows.add(u)
            matched_cols.add(v)
            max_weight += weight
    return max_weight

# Función principal
def main():    
    valid = 0
    values = []
    for i in range(1000):
        n = random.randint(10, 100)
        m = random.randint(10, 100) 

        # Generar el grafo bipartito completo
        graph = generate_graph(n, m)
        
        # Resolver con Kuhn-Munkres
        kuhn_munkres_result = solve_with_kuhn_munkres(graph)
        
        # Resolver con el algoritmo codicioso
        greedy_result = solve_with_greedy(graph)

        if kuhn_munkres_result - greedy_result > 1e-6:
            valid += 1
            values.append(kuhn_munkres_result - greedy_result)

    values = np.array(values)
    print(values.mean())

# Ejecutar el script
if __name__ == "__main__":
    main()
