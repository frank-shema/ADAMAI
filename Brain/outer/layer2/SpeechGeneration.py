# SpeechGeneration.py, By: Chance Brownfield
from Brain.utils import *

class SpeechGeneration():
    def __init__(self):
        pass

    def Speech(self, text, gmm=None):
        if gmm:
            tts_with_gmm(text, gmm)
        else:
            tts(text)