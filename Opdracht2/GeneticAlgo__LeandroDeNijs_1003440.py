'''
Melody maker for assignment 2 found at https://wiztech.nl/hr/ti/tinlab_ml/opdrachten/opdracht_2.pdf
Author: Leandro de Nijs

This file contains the genetic algorithm for the melody maker.
This uses the (altered) Muser class from muser.py to generate the music.

Different goals and implementation of the genetic algorithm:
◦ Building blocks: genomes consists of chords (3 notes/layers) instead of a singular note/layers, also the song ends in a II V I progression. 7 chords per pitch are available
◦ Mutatie: the offspring is mutated randomly by replacing a random chord in the genome with a random chord from the chord list. The least favorite genome is replaced by a newly random genome.
◦ Recombinatie: the offspring is created by performing a single point crossover on the parents
◦ Selectiedruk: the selection pressure is determined by the fitness function which is based on the user input
◦ Diversiteit: their are in total 7 different chords per pitch. The base pitch is 4 but this can be changed for more diversity, however this is not done because of muscial reasons. 
◦ Heuristiek: Only uses Chords and used the II V I progression as a heuristic to end the song
'''

import muser as ms
import Chords as ch
import random
import time
import multiprocessing

from pychord import Chord
from playsound import playsound

nrOfBars = 2
nrOfChords = 4
muser = ms.Muser()

#II V I progression from chords
II_V_I = [ch.Chord(*[chordOne.lower() for chordOne in Chord('Dm').components_with_pitch(root_pitch=4)]).getTuple(),
           ch.Chord(*[chordTwo.lower() for chordTwo in Chord('G').components_with_pitch(root_pitch=4)]).getTuple(),
           ch.Chord(*[chordThree.lower() for chordThree in Chord('C').components_with_pitch(root_pitch=4)]).getTuple()]

II_V_I = [[chord[0] for chord in II_V_I],  [chord[1] for chord in II_V_I], [chord[2] for chord in II_V_I]]

class GeneticAlgorithm:
    def fitness(self, genome):
        '''
        Fitness function for the genetic algorithm
        :param genome: the genome to be evaluated
        :return: the fitness of the genome
        '''

        print('playing genome ...')
        playMusic(genome)
        print('genome played')
        return float(input('Enter fitness 1-5: '))
    
    def selectParents(self, population, populationFitness):
        '''
        Selects two parents from the population
        :param population: the population from which the parents are selected
        :param populationFitness: the population with the fitness of the population
        :return: the two parents
        '''

        parentOne, parentTwo = random.choices(population, weights = [fitness for fitness, genome in populationFitness], k = 2)
        print (parentOne, parentTwo)
        return parentOne, parentTwo

    def singlePointCrossover(self, parentOne, parentTwo):
        '''
        Performs a single point crossover on the two parents
        :param parentOne: the first parent
        :param parentTwo: the second parent
        :return: the two children
        '''

        crossoverPoint = random.randint(0, len(parentOne[0]) - 1)

        parentOne[0] = parentOne[0][:crossoverPoint] + parentTwo[0][crossoverPoint:]
        parentOne[1] = parentOne[1][:crossoverPoint] + parentTwo[1][crossoverPoint:]
        parentOne[2] = parentOne[2][:crossoverPoint] + parentTwo[2][crossoverPoint:]

        parentTwo[0] = parentTwo[0][:crossoverPoint] + parentOne[0][crossoverPoint:]
        parentTwo[1] = parentTwo[1][:crossoverPoint] + parentOne[1][crossoverPoint:]
        parentTwo[2] = parentTwo[2][:crossoverPoint] + parentOne[2][crossoverPoint:]

        return parentOne, parentTwo

    def mutate(self, genome, chordsList):
        '''
        Mutates the genome
        :param genome: the genome to be mutated
        :return: the mutated genome
        '''

        mutationPoint = random.randint(0, len(genome[0]) - 1)
        noteOne, noteTwo, noteThree = chordsList[random.randint(0, len(chordsList) - 1)].getTuple()
        genome[0][mutationPoint] = noteOne
        genome[1][mutationPoint] = noteTwo
        genome[2][mutationPoint] = noteThree

        return genome


def createChords(rootPitch = 4):
    '''
    Creates a list of chords using the pychord library and the Chord class from Chords.py
    :param rootPitch: the root pitch of the chords
    :return: the list of chords
    '''

    chordsList = []
    for i in range (65, 72):    # Chords from A(m) to G(m) in ASCII
        currentChord = Chord(chr(i)).components_with_pitch(root_pitch=rootPitch)
        currentMChord = Chord((f'{chr(i)}m')).components_with_pitch(root_pitch=rootPitch)
        chordsList.append(ch.Chord(currentChord[0].lower(), currentChord[1].lower(), currentChord[2].lower()))
        chordsList.append(ch.Chord(currentMChord[0].lower(), currentMChord[1].lower(), currentMChord[2].lower()))
    
    return chordsList

def createGenome(chordsList):
    '''
    Creates a genome
    :param chordsList: the list of chords
    :return: the genome
    '''

    genome = [[], [], []]
    for _ in range(nrOfBars * nrOfChords):
        noteOne, noteTwo, noteThree = chordsList[random.randrange(0, len(chordsList))].getTuple(2**random.randrange(1, 4))
        genome[0] += (noteOne, )
        genome[1] += (noteTwo, )
        genome[2] += (noteThree, )

    return genome

def playMusic(melody, finished = False):
    '''
    Plays the music generated by the genetic algorithm
    :param melody: the melody to be played
    :param finished: if the song is finished or not
    '''

    #generate the music of the chords and combine the layers
    muser.generate(melody[:2])
    muser.combineLayers(melody[2])
    
    #start a thread to play the song
    thread = multiprocessing.Process(target=playsound, args=('song.wav',))
    thread.start()
    time.sleep(nrOfBars * nrOfChords) if not finished else time.sleep(30)
    thread.terminate()
    time.sleep(1)

    muser.removeFiles() if not finished else None

def makeSong(population):
    '''
    Creates a song from the best genome
    :param population: the population
    '''
    song = [[],[],[]]

    #Create song and appends the II-V-I progression
    [song[0].append(note) for note in (population[0][0]+population[1][0])*3 + II_V_I[0]]
    [song[1].append(note) for note in (population[0][1]+population[1][1])*3 + II_V_I[1]]
    [song[2].append(note) for note in (population[0][2]+population[1][1])*3 + II_V_I[2]]

    playMusic(song, True)

if __name__ == '__main__':
    chords = createChords()
    genetic = GeneticAlgorithm()

    #Create population
    print('generation population ...')
    population = []
    for i in range(5):
        population.append(createGenome(chords))

    running = True
    while running:
        #Evaluate fitness
        populationFitness = [(genetic.fitness(genome), genome) for genome in population]
        populationFitness.sort(reverse=True)

        #sorted population
        population = [genome for fitness, genome in populationFitness]

        #Elitism
        nextGeneration = population[:2]

        for i in range(int(len(population) / 2 - 1)):
            #Select parents
            parentOne, parentTwo = genetic.selectParents(population, populationFitness)
            offspringOne, offspringTwo = genetic.singlePointCrossover(parentOne, parentTwo)

            #Mutate offspring
            offspringOne = genetic.mutate(offspringOne, chords)
            offspringTwo = genetic.mutate(offspringTwo, chords)

            nextGeneration += [offspringOne, offspringTwo]

        #replace worst with random
        nextGeneration.append(createGenome(chords))

        print('next generation ...')
        print(len(nextGeneration))

        population = nextGeneration

        continueRunning = input('Continue? (y/n): ')
        if continueRunning == 'n':
            running = False
            makeSong(population)
        else:
            random.shuffle(population)

    

