'''
This program implements a neural network, using matrices (not numpy), that learns to play determine if a 3x3 array contains a 'X' or a 'O'.
The neural network consists of 9 input nodes and 2 output nodes.
The neural network is trained using test data from testData.py.
The neural network is tested using test data from testData.py.
The neural network is trained using the backpropagation algorithm.

Author: Leandro de Nijs

Goals of this program:
- Pure python implementation of a neural network to distinguish between 'X' and 'O'.
- Consists of only matrices and vectors.
- Softmax function is used as activation function.
- Training is done using the backpropagation algorithm.
- MSE is used as cost function.
'''

import testData
import math
import random

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
        inputVector = []
        for row in range(nrOfRows):
            for column in range(nrOfColumns):
                inputVector.append(data[0][row][column])
        totalCost += costFunc(softmax(matrixMult(weightsInputOutput, inputVector)), symbolVecs[data[1]])
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

def matrixMult(matrix, vector):
    ''' Multiplies a matrix with a vector. 
    :param matrix: The matrix.
    :param vector: The vector.
    :return: The product of the matrix and the vector.
    '''
    return [sum([matrix[i][j] * vector[j] for j in range(len(vector))]) for i in range(len(matrix))]

def train():
    ''' Trains the neural network using the backpropagation algorithm.'''
    print('training...')
    averageCost = computeAverageCost()
    currentIteration = 0
    print('average cost before training: ' + str(averageCost))
    while averageCost > costThreshold:
        bestLink = []
        bestCost = averageCost
        for i in range (len(weightsInputOutput)):
            for j in range (len(weightsInputOutput[i])):
                weightsInputOutput[i][j] += learningRate
                averageCost = computeAverageCost()
                if averageCost <= bestCost:
                    bestLink = [i, j]
                    bestCost = averageCost
                weightsInputOutput[i][j] -= learningRate
        if bestLink:
            weightsInputOutput[bestLink[0]][bestLink[1]] += learningRate
        else:
            break
        currentIteration += 1
        print('iteration: ' + str(currentIteration) + ' average cost: ' + str(averageCost))
    print('average cost after training: ' + str(averageCost))

def test():
    ''' Tests the neural network using the test data.'''
    print('testing...')

    for data in testData.testSet:
        inputVec = []
        for row in range(nrOfRows):
            for column in range(nrOfColumns):
                inputVec.append(data[0][row][column])
        output = softmax(matrixMult(weightsInputOutput, inputVec))
        print('input: ' + str(data[0]) + ' output: ' + str(output) + ' expected: ' + str(symbolVecs[data[1]]))
    
            
if __name__ == '__main__':
    # The input vector
    inputVector = []

    # The weights of the neural network
    weightsInputOutput = [[random.randrange(-3, 5) for i in range(nrOfRows * nrOfColumns)] for j in range(nrOfSymbols)]

    train()
    test()
