import numpy as np
from keras.datasets import mnist

#Lock the seed to make it replicable
np.random.seed(seed=40)

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def nn(input, weights):
    return np.matmul(input, weights)


#Turn the 2d list into a 1d one and nomralize the numbers to prevent the gradient decent from getting wild (big delta * big input with a reasonable alpha = big mess)
def flatten_list(lst):
    return [i/255 for sub in lst for i in sub]

#convert raw number to a list representng what we want the output to be
def convert_y_train_to_binary(y_train):
    ret = np.array([0.0 for i in range(10)])
    ret[y_train] = 1.0
    return ret

#init weights
weights = np.random.random_sample((len(flatten_list(x_train[0])),len(convert_y_train_to_binary(y_train[0]))))

error = np.array([0.0 for i in range(10)])
delta = np.array([0.0 for i in range(10)])

alpha = 0.01

#Train our nn
for i in range(len(x_train)):
    input = flatten_list(x_train[i])
    true = convert_y_train_to_binary(y_train[i])
        
    pred = nn(input, weights)

    for i in range(len(true)):
        error[i] = (pred[i] - true[i]) **2
        delta[i] = pred[i] - true[i]

    weight_deltas = np.outer(input, delta)*alpha
    weights -= weight_deltas

input = flatten_list(x_test[0])
true = convert_y_train_to_binary(y_test[0])

test_pred = nn(input, weights)

print("TEST PRED: " + str(test_pred))
print("TEST EXPECTED: "+ str(true))
print("ERROR: " + str(error))
