from helper_functions import ele_mul, outer_prod
from neural_networks import many_to_one_neural_network, one_to_many_neural_network, many_to_many_neural_network

#Gradient descent on a many to one neural network
print("Many to One:")

weights = [0.1, 0.2, -.1]
toes = [8.5 , 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2 , 1.3, 0.5, 1.0]
win_or_lose_binary = [1, 1, 0, 1]
true = win_or_lose_binary[0]
input = [toes[0],wlrec[0],nfans[0]]
for iter in range(3):
    pred = many_to_one_neural_network(input, weights)
    error = (pred-true) **2
    delta = pred - true
    weight_deltas = ele_mul(delta, input)

    alpha = 0.01
    for i in range(len(weights)):
        weights[i] -= weight_deltas[i] * alpha
    print("ITTERATION: " + str(iter))
    print("PRED: " + str(pred))
    print("ERROR: " + str(error))
    print("DELTA: " + str(delta))
    print("Weights: " + str(weights))
    print("Weights DELTAS: ")
    print(weight_deltas)
    print()

#Gradient descent on a one to many neural network
print("ONE TO MANY:")
alpha = 0.1
weights = [0.3, 0.2, 0.9]
wlrec = [.65, 1.0, 1.0, .9]
hurt = [0.1, 0.0, 0.0, 0.1]
win = [ 1, 1, 0, 1]
sad = [0.1, 0.0, 0.1, 0.2]
input = wlrec[0]
true = [hurt[0], win[0], sad[0]]
error = [0, 0, 0]
deltas = [0, 0, 0]
for iter in range(30):
    pred = one_to_many_neural_network(input, weights)
    for i in range(len(true)):
        error[i] = (pred[i] - true[i]) ** 2
        deltas[i] = pred[i] - true[i]
    weight_deltas = ele_mul(input, deltas)
    for i in range(len(weight_deltas)):
        weights[i] -= (weight_deltas[i] * alpha)

    print("ITTERATION: " + str(iter))
    print("PRED: " + str(pred))
    print("ERROR: " + str(error))
    print("DELTA: " + str(delta))
    print("Weights: " + str(weights))
    print("Weights DELTAS: ")
    print(weight_deltas)
    print()

#Gradient Descent on a many to many network, I think I have something wrong here
print("")
print("--MANY TO MANY--")
print("")

            # toes %win # fans
weights = [ [0.1, 0.1, -0.3],# hurt?
            [0.1, 0.2, 0.0], # win?
            [0.0, 1.3, 0.1] ]# sad?

wlrec = [0.65,0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]
hurt = [0.1, 0.0, 0.0, 0.1]
win = [ 1, 1, 0, 1]
sad = [0.1, 0.0, 0.1, 0.2]
alpha = 0.01
input = [toes[0],wlrec[0],nfans[0]]
true = [hurt[0], win[0], sad[0]]
error = [0, 0, 0]
delta = [0, 0, 0]

for iter in range(3):
    print("ITTERATION: " + str(iter))
    pred = many_to_many_neural_network(input, weights)
    print("PREDICTION: " + str(pred))
    print("WEIGHTS: "+ str(weights))
    for i in range(len(true)):
        error[i] = (pred[i] - true[i]) ** 2
        delta[i] = pred[i] - true[i]
    weight_deltas = outer_prod(input, delta)

    for i in range(len(weights)):
        for j in range(len(weights[0])):
            weights[i][j] -= weight_deltas[i][j] * alpha
    
    print("ERROR: " + str(error))
    print("DELTA: " + str(delta))
    print("weight_deltas: " + str(weight_deltas))
    print()