import glob as gl
import os
import tomita.legacy.pysynth as ps

class Muser:
    def generate (self, song, filename = 'inbetween.wav'):
        for trackIndex, track in enumerate (song):
            ps.make_wav (
                track,
                bpm = 130,
                transpose = 0,
                pause = 0.1,
                boost = 1,
                repeat = 0,
                fn = self.getTrackFileName (trackIndex),
            )

        ps.mix_files (
            *[self.getTrackFileName (trackIndex) for trackIndex in range (len (song))],
            filename, 1
        )

        for fileName in gl.glob ('track_*.wav'):
            os.remove (fileName)

    def getTrackFileName (self, trackIndex):
        return f'track_{str (1000 + trackIndex) [1:]}.wav'
    
    def combineLayers(self, melody, fromFilename = 'inBetween.wav', toFilename = 'song.wav'):
        filename = 'inBetween2.wav'
        
        ps.make_wav (
        melody,
        bpm = 130,
        transpose = 0,
        pause = 0.1,
        boost = 1,
        repeat = 0,
        fn = filename,
        )

        ps.mix_files(fromFilename, filename, toFilename, 1)

    def removeFiles(self):
        os.remove('inBetween.wav')
        os.remove('inBetween2.wav')
        os.remove('song.wav')