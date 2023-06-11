'''
This program implements a neural network with a hidden layer, using matrices with numpy, that learns to play determine if a 3x3 array contains a 'X' or a 'O'.
The neural network consists of 9 input nodes, 9 hidden nodes and 2 output nodes.
The neural network is trained using test data from testData.py.
The neural network is tested using test data from testData.py.

Author: Leandro de Nijs

Goals of this program:
- Numpy python implementation of a neural network to distinguish between 'X' and 'O'.
- Consists of only matrices and vectors made with numpy.
- includes a hidden layer.
- Softmax function is used as activation function.
- MSE is used as cost function.
'''

import testData
import math
import numpy as np

nrOfColumns = 3
nrOfRows = 3
nrOfSymbols = 2

symbolVecs = {'O': (1, 0), 'X': (0, 1)}

costThreshold = 0.01
learningRate = 0.1

def computeAverageCost():
    ''' Computes the average cost of the neural network. '''
    totalCost = 0
    for data in testData.trainingSet:
        inputVector = np.array([[]])
        for row in range(nrOfRows):
            for column in range(nrOfColumns):
                inputVector = np.append(inputVector, np.array([data[0][row][column]]))
        totalCost += costFunc(softmax(np.dot(weightsHiddenOutput, softmax(np.dot(weightsInputHidden, inputVector)))), symbolVecs[data[1]])
    return totalCost / len(testData.trainingSet)

def costFunc(output, target):
    ''' Computes the cost of the neural networ using the MSE cost function.
    :param output: The output of the neural network.
    :param target: The expected output of the neural network.
    :return: The cost of the neural network.'''
    return sum([(output[i] - target[i])**2 for i in range(len(output))])

def softmax(output):
    ''' Computes the softmax of the output of the neural network.
    :param output: The output of the neural network.
    :return: The softmax of the output of the neural network.
    '''
    sumOfExp = sum([math.exp(x) for x in output])
    return [math.exp(x) / sumOfExp for x in output]

def sigmoid(output):
    ''' Computes the sigmoid of the output of the neural network.
    :param output: The output of the neural network.
    :return: The sigmoid of the output of the neural network.
    '''
    return [1 / (1 + math.exp(-x)) for x in output]

def findBestLink(cost):
    ''' Finds the best link to change to improve the neural network.
    :param cost: The current cost of the neural network.
    :return: The best link to change to improve the neural network and the cost
    '''
    bestLink = []
    bestCost = cost
    for i in range(len(weightsInputHidden)):
        for j in range(len(weightsInputHidden[i])):
            weightsInputHidden[i][j] += learningRate
            cost = computeAverageCost()
            if cost <= bestCost:
                bestLink = ['input_hidden', i, j]
                bestCost = cost
            weightsInputHidden[i][j] -= learningRate

    for i in range(len(weightsHiddenOutput)):
        for j in range(len(weightsHiddenOutput[i])):
            weightsHiddenOutput[i][j] += learningRate
            cost = computeAverageCost()
            if cost <= bestCost:
                bestLink = ['hidden_output', i, j]
                bestCost = cost
            weightsHiddenOutput[i][j] -= learningRate
    
    return bestLink, bestCost

def train():
    ''' Trains the neural network using the backpropagation algorithm.'''
    print('training...')
    averageCost = computeAverageCost()
    currentIteration = 0
    print('average cost before training: ' + str(averageCost))
    while averageCost > costThreshold:
        bestLink, bestCost = findBestLink(averageCost)

        if bestLink:
            if bestLink[0] == 'input_hidden':
                weightsInputHidden[bestLink[1]][bestLink[2]] += learningRate
                averageCost = bestCost
            elif bestLink[0] == 'hidden_output':
                weightsHiddenOutput[bestLink[1]][bestLink[2]] += learningRate
                averageCost = bestCost
        else:
            break

        currentIteration += 1
        print('iteration: ' + str(currentIteration) + ' average cost: ' + str(averageCost))
    print('average cost after training: ' + str(averageCost))

def predict():
    ''' Tests the neural network using the test data.'''
    print('testing...')
    for data in testData.testSet:
        inputVector = np.array([[]])
        for row in range(nrOfRows):
            for column in range(nrOfColumns):
                inputVector = np.append(inputVector, np.array([data[0][row][column]]))
        output = softmax(np.dot(weightsHiddenOutput, softmax(np.dot(weightsInputHidden, inputVector))))
        print('input: ' + str(data[0]) + ' output: ' + str(output) + ' expected: ' + str(symbolVecs[data[1]]))
    
            
if __name__ == '__main__':
    inputVector = np.array([])
    weightsInputHidden = np.random.randint(-3, 5, size=(nrOfRows * nrOfColumns, nrOfRows * nrOfColumns)).astype(float)
    weightsHiddenOutput = np.random.randint(-3, 5, size=(nrOfSymbols, nrOfRows * nrOfColumns)).astype(float)
    print(weightsHiddenOutput)
    train()
    predict()
