import numpy as np

def make_coocurrence_matrix(token_list, token_size, alpha=98, theta=1000):
    token_list = [list(filter(lambda x: x > 3, token)) for token in token_list]
    exchange_matrix = np.zeros(shape=(token_size, token_size))
    for token in token_list:
        for i in range(1, len(token)):
            if token[i] == token[i - 1]:
                continue
            exchange_matrix[token[i - 1]][token[i]] += 1
    
    # smoothing
    exchange_matrix = np.where(exchange_matrix >= alpha, exchange_matrix, 0) 
    exchange_matrix = exchange_matrix / theta
    exchange_matrix = np.where(exchange_matrix > 0, np.exp(exchange_matrix), 0)
    
    # row normalization
    for i in range(token_size):
        row_sum = sum(exchange_matrix[i]) + np.exp(1)
        for j in range(token_size):
            if exchange_matrix[i][j] != 0:
                exchange_matrix[i][j] = exchange_matrix[i][j] / row_sum
    
    # diagonal equal 1
    for i in range(token_size):
        exchange_matrix[i][i] = 1

    # simetrization
    for i in range(token_size):
        for j in range(token_size):
            exchange_matrix[i][j] = max(exchange_matrix[i][j], exchange_matrix[j][i])
    return exchange_matrix