'''
This program implements a neural network that learns to play determine if a 3x3 array contains a 'X' or a 'O'.
The neural network consists of 9 input nodes and 2 output nodes.
The neural network is trained using test data from testData.py.
The neural network is tested using test data from testData.py.

Author: Leandro de Nijs

Goals of this program:
- Pure python implementation of a neural network to distinguish between 'X' and 'O'.
- Consists of a Node class and a Link class.
- Softmax function is used as activation function.
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

class Node:
    ''' A node in the neural network. '''

    def __init__(self):
        self.value = 0
        self.links = []

    def getValue(self):
        ''' Returns the value of the node/sum of the values of the links.'''

        value = sum([link.getValue() for link in self.links])
        return value
    
    def addLink(self, link):
        ''' Adds a link to the node. '''

        self.links.append(link)

class Link:
    ''' A link between two nodes in the neural network. '''

    def __init__(self, inNode, outNode, weight=0):
        ''' Initializes a link between two nodes in the neural network.
        :param inNode: The node the link starts from.
        :param outNode: The node the link ends at.
        :param weight: The weight of the link.
        '''

        self.inNode = inNode
        self.weight = random.randrange(-3, 5)
        outNode.addLink(self)
    
    def getValue(self):
        ''' Returns the value of the link.'''

        return self.weight * self.inNode.value
    
def computeAverageCost():
    ''' Computes the average cost of the neural network. '''
    totalCost = 0
    for data in testData.trainingSet:
        for row in range(nrOfRows):
            for column in range(nrOfColumns):
                inNodes[row][column].value = data[0][row][column]
        totalCost += costFunc(softmax([outNode.getValue() for outNode in outNodes]), symbolVecs[data[1]])
    return totalCost / len(testData.trainingSet)

def costFunc(output, target):
    ''' Computes the cost of the neural network using the MSE cost function.
    :param output: The output of the neural network.
    :param target: The expected output of the neural network.
    :return: The cost of the neural network.
    '''
    return sum([(output[i] - target[i])**2 for i in range(len(output))])

def softmax(output):
    ''' Computes the softmax of the output of the neural network.
    :param output: The output of the neural network. 
    :return: The softmax of the output of the neural network.
    '''
    sumOfExp = sum([math.exp(x) for x in output])
    return [math.exp(x) / sumOfExp for x in output]

def train():
    ''' Trains the neural network using the backpropagation algorithm. '''
    print('training...')
    averageCost = computeAverageCost()
    currentIteration = 0
    print('average cost before training: ' + str(averageCost))
    while averageCost > costThreshold:
        bestLink = None
        bestCost = averageCost
        for link in links:
            link.weight += learningRate
            averageCost = computeAverageCost()
            if averageCost <= bestCost:
                bestLink = link
                bestCost = averageCost
            link.weight -= learningRate
        if bestLink:
            bestLink.weight += learningRate
        else:
            break
        currentIteration += 1
        print('iteration: ' + str(currentIteration) + ' average cost: ' + str(averageCost))
    print('average cost after training: ' + str(averageCost))

def predict():
    ''' Tests the neural network using the test data. '''
    print('testing...')
    for data in testData.testSet:
        for row in range(nrOfRows):
            for column in range(nrOfColumns):
                inNodes[row][column].value = data[0][row][column]
        output = softmax([outNode.getValue() for outNode in outNodes])
        print('input: ' + str(data[0]) + ' output: ' + str(output) + ' expected: ' + str(symbolVecs[data[1]]))
            
if __name__ == '__main__':
    #create nodes and links
    inNodes = [[Node() for i in range(nrOfColumns)] for j in range(nrOfRows)]
    outNodes = [Node() for i in range(nrOfSymbols)]
    links = []

    #create links between inNodes and outNodes
    for row in inNodes:
        for inNode in row:
            for outNode in outNodes:
                links.append(Link(inNode, outNode))

    train()
    predict()
