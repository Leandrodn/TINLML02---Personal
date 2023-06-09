class Chord:
    def __init__(self, noteOne, noteTwo, noteThree):
        '''
        Initializes the chord
        :param noteOne: the first note of the chord
        :param noteTwo: the second note of the chord
        :param noteThree: the third note of the chord
        '''

        self.noteOne = noteOne
        self.noteTwo = noteTwo
        self.noteThree = noteThree

    def getTuple(self, chordLength = 4):
        '''
        Returns a tuple of the chord with the given length
        :param chordLength: the length of the chord
        :return: the tuple of the chord
        '''
        return (self.noteOne, chordLength), (self.noteTwo, chordLength), (self.noteThree, chordLength)