def w_sum(a, b):
    assert(len(a) == len(b))
    ret = 0 
    for i in range(len(a)):
        ret += (a[i] * b[i])
    
    return ret

def ele_mul(number, vector):
    output = [0 for i in range(len(vector))]
    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output

def vector_mat_mul(vector, matrix):
    assert(len(vector) == len(matrix))
    output = [0 for i in range(len(vector))]
    for i in range(len(vector)):
        output[i] = w_sum(vector, matrix[i])
    return output

def zeros_matrix(a, b):
    return [[0 for i in range(b)] for j in range(a)]

def outer_prod(vec_a, vec_b):
    out = zeros_matrix(len(vec_a), len(vec_b))
    for i in range(len(vec_a)):
        for j in range(len(vec_b)):
            out[i][j] = vec_a[i]*vec_b[j]

    return out