from helper_functions import w_sum, ele_mul, vector_mat_mul

def one_to_one_neural_network(input, weight):
    pred = input * weight
    return pred

def many_to_one_neural_network(input, weights):
    pred = w_sum(input, weights)
    return pred

def one_to_many_neural_network(input, weights):
    pred = ele_mul(input, weights)
    return pred

def many_to_many_neural_network(input, weights):
    pred = vector_mat_mul(input, weights)
    return pred