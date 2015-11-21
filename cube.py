from copy import deepcopy
import timeit
import sys, os
import random 
import argparse
import heapq

class Cube:
    def __init__(self):

        self.fUp = [['Y', 'Y', 'Y'],
                    ['Y', 'Y', 'Y'],
                    ['Y', 'Y', 'Y']]
        self.fDown = [['W', 'W', 'W'],
                        ['W', 'W', 'W'],
                        ['W', 'W', 'W']]
        self.fLeft = [['R', 'R', 'R'],
                        ['R', 'R', 'R'],
                        ['R', 'R', 'R']]
        self.fRight = [['O', 'O', 'O'],
                        ['O', 'O', 'O'],
                        ['O','O','O']]
        self.fFront = [['G','G','G'],
                        ['G', 'G', 'G'],
                        ['G', 'G', 'G']]
        self.fBack = [['B', 'B', 'B'],
                        ['B', 'B', 'B'],
                        ['B', 'B', 'B']]

    def currentState(self):
        # returns copy of state, changes made to this should not affect the stored faces
        return [deepcopy(face) for face in 
            [self.fUp, self.fDown, self.fLeft, self.fRight, self.fFront, self.fBack]]

