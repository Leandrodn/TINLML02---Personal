import testData
import math
import random

nrOfColumns = 3
nrOfRows = 3
nrOfSymbols = 2

symbolVecs = {'O': (1, 0), 'X': (0, 1)}

costThreshold = 0.01

class Node:
    def __init__(self):
        self.value = 0
        self.links = []

    def getValue(self):
        value = sum([link.getValue() for link in self.links])
        return value
    
    def addLink(self, link):
        self.links.append(link)

class Link:
    def __init__(self, inNode, outNode, weight=0):
        self.inNode = inNode
        self.weight = random.randrange(-3, 5)
        outNode.addLink(self)
    
    def getValue(self):
        return self.weight * self.inNode.value
    
def computeAverageCost():
    totalCost = 0
    for data in testData.trainingSet:
        for row in range(nrOfRows):
            for column in range(nrOfColumns):
                inNodes[row][column].value = data[0][row][column]
        totalCost += costFunc(softmax([outNode.getValue() for outNode in outNodes]), symbolVecs[data[1]])
    return totalCost / len(testData.trainingSet)

def costFunc(output, target):
    return sum([(output[i] - target[i])**2 for i in range(len(output))])

def softmax(output):
    sumOfExp = sum([math.exp(x) for x in output])
    return [math.exp(x) / sumOfExp for x in output]

def train():
    print('training...')
    averageCost = computeAverageCost()
    currentIteration = 0
    print('average cost before training: ' + str(averageCost))
    while averageCost > costThreshold:
        bestLink = None
        bestCost = averageCost
        for link in links:
            link.weight += 0.1
            averageCost = computeAverageCost()
            if averageCost <= bestCost:
                bestLink = link
                bestCost = averageCost
            link.weight -= 0.1
        if bestLink:
            bestLink.weight += 0.1
        else:
            break
        currentIteration += 1
        print('iteration: ' + str(currentIteration) + ' average cost: ' + str(averageCost))
    print('average cost after training: ' + str(averageCost))

def test():
    print('testing...')
    for data in testData.testSet:
        for row in range(nrOfRows):
            for column in range(nrOfColumns):
                inNodes[row][column].value = data[0][row][column]
        output = softmax([outNode.getValue() for outNode in outNodes])
        print('input: ' + str(data[0]) + ' output: ' + str(output) + ' expected: ' + str(symbolVecs[data[1]]))
            
if __name__ == '__main__':
    inNodes = [[Node() for i in range(nrOfColumns)] for j in range(nrOfRows)]
    outNodes = [Node() for i in range(nrOfSymbols)]
    links = []

    #create links between inNodes and outNodes
    for row in inNodes:
        for inNode in row:
            for outNode in outNodes:
                links.append(Link(inNode, outNode))

    train()
    test()
