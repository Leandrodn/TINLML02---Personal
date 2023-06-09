# TODO implement biggest different + and -
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
    totalCost = 0
    for data in testData.trainingSet:
        inputVector = []
        for row in range(nrOfRows):
            for column in range(nrOfColumns):
                inputVector.append(data[0][row][column])
        totalCost += costFunc(softmax(matrixMult(weightsInputOutput, inputVector)), symbolVecs[data[1]])
    return totalCost / len(testData.trainingSet)

def costFunc(output, target):
    return sum([(output[i] - target[i])**2 for i in range(len(output))])

def softmax(output):
    sumOfExp = sum([math.exp(x) for x in output])
    return [math.exp(x) / sumOfExp for x in output]

def matrixMult(matrix, vector):
    return [sum([matrix[i][j] * vector[j] for j in range(len(vector))]) for i in range(len(matrix))]

def train():
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
    print('testing...')

    for data in testData.testSet:
        inputVec = []
        for row in range(nrOfRows):
            for column in range(nrOfColumns):
                inputVec.append(data[0][row][column])
        output = softmax(matrixMult(weightsInputOutput, inputVec))
        print('input: ' + str(data[0]) + ' output: ' + str(output) + ' expected: ' + str(symbolVecs[data[1]]))
    
            
if __name__ == '__main__':
    inputVector = []
    weightsInputOutput = [[random.randrange(-3, 5) for i in range(nrOfRows * nrOfColumns)] for j in range(nrOfSymbols)]

    train()
    test()
