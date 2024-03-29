import numpy as np
streetlights = np.array( [ [1,0,1],
                          [0,1,1],
                          [0,0,1],
                          [1,1,1],
                          [0,1,1],
                          [1,0,1]])
walk_vs_stop = np.array([ [0],
                         [1],
                         [0],
                         [1],
                         [1],
                         [0]])

weights = np.array([0.5,0.48,-0.7])
alpha = 0.1

input = streetlights[0]
goal_prediction = walk_vs_stop[0]

for itteration in range(40):
    error_for_all_lights = 0
    for row in range(len(walk_vs_stop)):
        input = streetlights[row]
        goal_prediction = walk_vs_stop[row]
        prediction = input.dot(weights)
        error = (goal_prediction-prediction)**2
        error_for_all_lights += error
        deltas = prediction - goal_prediction
        weights = weights - (alpha * (input * deltas))
        print("PREDICTION: "+ str(prediction))
    print("ERROR: " + str(error_for_all_lights))

