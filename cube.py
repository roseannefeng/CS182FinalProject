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
        # note: in functions that call face, functions will access the copy on state,
        # not the faces stored in the model itself

    def currentState(self):
        """ 
        Returns copy of state.
        Changes made to this do not affect the stored faces.
        """
        return [deepcopy(face) for face in 
            [self.fUp, self.fDown, self.fLeft, self.fRight, self.fFront, self.fBack]]

    def rotate90(self, state, face):
        """
        Rotates given face (int 0 through 5) on the state by
        90 degrees clockwise looking directly on face.
        Returns state', does not modify state or cube.
        """
        return

    def rotate(self, state, face, deg):
        """
        Rotates given face on the state by deg, which is an int 1, 2, or 3, by calling
        rotate90 deg times. Returns state', does not modify state or cube.
        """
        return

    def scramble(self, k):
        """
        Scrambles cube by calling k rotate functions which select a face and degree
        over a uniform random distribution. Modifies cube faces.
        """
        return

    def numConflicts(self, state, face):
        """
        Returns the number of squares on face that differ from their initial color.
        Note: in a 3x3 cube, this value will range from 1 to 8 because center square
        in a cube cannot move.
        """
        return

    def fGoal(self, state, face):
        """
        Returns True if face is a solid color.
        """
        return

    def goal(self, state):
        """ 
        Returns True if all faces are a solid color.
        """
        return

    def update(self, state):
        """
        Given state, update the faces of the model accordingly.
        """
        return

