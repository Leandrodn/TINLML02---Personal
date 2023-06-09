import testData
import math
import random
import numpy as np

nrOfColumns = 3
nrOfRows = 3
nrOfSymbols = 2

symbolVecs = {'O': (1, 0), 'X': (0, 1)}

costThreshold = 0.01
learningRate = 0.1

def computeAverageCost():
    totalCost = 0
    for data in testData.trainingSet:
        inputVector = np.array([])
        for row in range(nrOfRows):
            for column in range(nrOfColumns):
                inputVector = np.append(inputVector, [data[0][row][column]])
        print('inputVector: ' + str(inputVector.reshape(-1, 1)))
        totalCost += costFunc(sigmoid(np.matmul(weightsHiddenOutput.transpose(), sigmoid(np.matmul(weightsInputHidden, inputVector)))), symbolVecs[data[1]])
    return totalCost / len(testData.trainingSet)

def costFunc(output, target):
    print('output: ' + str(output) + ' target: ' + str(target))
    return sum([(output[i] - target[i])**2 for i in range(len(output))])

def softmax(output):
    sumOfExp = sum([math.exp(x) for x in output])
    return [math.exp(x) / sumOfExp for x in output]

def sigmoid(output):
    print('output: ' + str(output))
    return [1 / (1 + math.exp(-x)) for x in output]

def matrixVecMult(matrix, vector):
    return [sum([matrix[i][j] * vector[j] for j in range(len(vector))]) for i in range(len(matrix))]

def matrixMult(matrix1, matrix2):
    return [[sum([matrix1[i][j] * matrix2[j][k] for j in range(len(matrix2))]) for k in range(len(matrix2[0]))] for i in range(len(matrix1))]

def train():
    print('training...')
    averageCost = computeAverageCost()
    currentIteration = 0
    print('average cost before training: ' + str(averageCost))
    while averageCost > costThreshold:
        bestLink = []
        bestCost = averageCost
        for i in range(len(weightsInputHidden)):
            for j in range(len(weightsInputHidden[i])):
                weightsInputHidden[i][j] += learningRate
                averageCost = computeAverageCost()
                if averageCost <= bestCost:
                    bestLink = ['input_hidden', i, j]
                    bestCost = averageCost
                weightsInputHidden[i][j] -= learningRate

        for i in range(len(weightsHiddenOutput)):
            for j in range(len(weightsHiddenOutput[i])):
                weightsHiddenOutput[i][j] += learningRate
                averageCost = computeAverageCost()
                if averageCost <= bestCost:
                    bestLink = ['hidden_output', i, j]
                    bestCost = averageCost
                weightsHiddenOutput[i][j] -= learningRate

        if bestLink:
            if bestLink[0] == 'input_hidden':
                weightsInputHidden[bestLink[1]][bestLink[2]] += learningRate
            elif bestLink[0] == 'hidden_output':
                weightsHiddenOutput[bestLink[1]][bestLink[2]] += learningRate
        else:
            break

        currentIteration += 1
        print('iteration: ' + str(currentIteration) + ' average cost: ' + str(averageCost))
    print('average cost after training: ' + str(averageCost))

def test():
    print('testing...')
    for data in testData.testSet:
        inputVec = []
        for row in range(nrOfRows):
            for column in range(nrOfColumns):
                inputVec.append(data[0][row][column])
        output = sigmoid(np.dot(np.dot(weightsHiddenOutput, weightsInputHidden), inputVector))
        print('input: ' + str(data[0]) + ' output: ' + str(output) + ' expected: ' + str(symbolVecs[data[1]]))
    
            
if __name__ == '__main__':
    inputVector = np.array([])
    weightsInputHidden = np.random.randint(-3, 5, size=(nrOfRows * nrOfColumns, nrOfRows * nrOfColumns))
    print(weightsInputHidden)
    weightsHiddenOutput = np.random.randint(-3, 5, size=(nrOfRows * nrOfColumns, nrOfSymbols))
    print(weightsHiddenOutput)
    train()
    # test()
