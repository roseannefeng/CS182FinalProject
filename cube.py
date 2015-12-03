from copy import deepcopy
import timeit
import sys, os
import random 
import argparse
import heapq
from collections import defaultdict


#helper function that transposes matrix in order to make reassignment easier
def transpose(matrix):
    trmatrix = [[row[0] for row in matrix],[row[1] for row in matrix],  [row[2] for row in matrix]]
    return trmatrix

ultimate_list = []

def listToTuple(matrix):
    return tuple([tuple([tuple(i) for i in x]) for x in matrix])

# eyyyy hw1
class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    frontier = Queue()
    frontier.push((problem.currentState(), []))
    explored = set()
    while not(frontier.isEmpty()):
        s, p = frontier.pop()
        problem.update(s)
        if problem.isGoal(s):
            return p
#        explored.add(s)
        for a in problem.getSuccessors(s):
            s_ = a[0]
            p_ = p + [a[1]]
            if problem.isGoal(s_):
                return p_
            tuples_ = listToTuple(s_)
            if tuples_ not in explored:
                frontier.push((s_, p_))
                explored.add(tuples_)

    return []


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

        up, down, left, right, front, back = [deepcopy(f) for f in state]

        # up = self.fUp
        # down = self.fDown
        # left = self.fLeft
        # right = self.fRight
        # front = self.fFront
        # back = self.fBack

        # IF FACE IS UP (UP AND DOWN DONT CHANGE)
        if face == 0:
            """
            temp = self.fFront[0]
            self.fFront[0] = self.fRight[0]
            self.fRight[0] = self.fBack[0]
            self.fBack[0] = self.fLeft[0]
            self.fLeft[0] = temp
            """
            temp = front[0]
            front[0] = right[0]
            right[0] = back[0]
            back[0] = left[0]
            left[0] = temp

        # IF FACE IS DOWN (UP AND DOWN DONT CHANGE)
        elif face == 1:
            """
            temp = self.fFront[2]
            self.fFront[2] = self.fLeft[2]
            self.fLeft[2] = self.fBack[2]
            self.fBack[2] = self.fRight[2]
            self.fRight[2] = temp
            """
            temp = front[2]
            front[2] = left[2]
            left[2] = back[2]
            back[2] = right[2]
            right[2] = temp

        # IF FACE IS LEFT (LEFT AND RIGHT DONT CHANGE)
        elif face == 2:
            """
            relevant_matrices = [self.fFront, self.fDown, self.fBack, self.fUp]
            for i in relevant_matrices:
                i = transpose(i)

            temp = self.fFront[0]
            self.fFront[0] = self.fUp[0]
            self.fUp[0] = self.fBack[0]
            self.fBack[0] = self.fDown[0]
            self.fDown[0] = temp

            for ind, i in enumerate(relevant_matrices):
                i = transpose(i)

                if ind == 0:
                    self.fFront = i
                if ind == 1:
                    self.fDown = i
                if ind == 2:
                    self.fBack = i
                if ind == 3:
                    self.fUp = i
            """
            relevant_matrices = [front, down, back, up]
            for i in range(len(relevant_matrices)):
                matr = relevant_matrices[i]
                relevant_matrices[i] = transpose(matr)
            front, down, back, up = relevant_matrices

            temp = front[0]
            front[0] = up[0]
            up[0] = back[0]
            back[0] = down[0]
            down[0] = temp

            for ind, i in enumerate(relevant_matrices):
                i = transpose(i)

                if ind == 0:
                    front = i
                if ind == 1:
                    down = i
                if ind == 2:
                    back = i
                if ind == 3:
                    up = i


        # IF FACE IS RIGHT (LEFT AND RIGHT DONT CHANGE)
        elif face == 3:
            """
            relevant_matrices = [self.fFront, self.fDown, self.fBack, self.fUp]
            for i in relevant_matrices:
                i = transpose(i)
            temp = self.fFront[2]
            self.fFront[2] = self.fDown[2]
            self.fDown[2] = self.fBack[2]
            self.fBack[2] = self.fUp[2]
            self.fUp[2] = temp

            for ind, i in enumerate(relevant_matrices):
                i = transpose(i)

                if ind == 0:
                    self.fFront = i
                if ind == 1:
                    self.fDown = i
                if ind == 2:
                    self.fBack = i
                if ind == 3:
                    self.fUp = i
            """
            relevant_matrices = [front, down, back, up]
            for i in range(len(relevant_matrices)):
                matr = relevant_matrices[i]
                relevant_matrices[i] = transpose(matr)
            front, down, back, up = relevant_matrices
            temp = front[2]
            front[2] = down[2]
            down[2] = back[2]
            back[2] = up[2]
            up[2] = temp

            for ind, i in enumerate(relevant_matrices):
                i = transpose(i)

                if ind == 0:
                    front = i
                if ind == 1:
                    down = i
                if ind == 2:
                    back = i
                if ind == 3:
                    up = i

        # IF FACE IS FRONT (FRONT AND BACK DONT CHANGE)
        elif face == 4:
            """
            relevant_matrices = [self.fLeft, self.fDown, self.fRight, self.fUp]
            for i in relevant_matrices:
                i = transpose(i)
            temp = self.fLeft[2]
            self.fLeft[2] = self.fDown[2]
            self.fDown[2] = self.fRight[2]
            self.fRight[2] = self.fUp[2]
            self.fUp[2] = temp

            for ind, i in enumerate(relevant_matrices):
                i = transpose(i)

                if ind == 0:
                    self.fLeft = i
                if ind == 1:
                    self.fDown = i
                if ind == 2:
                    self.fRight = i
                if ind == 3:
                    self.fUp = i
            """
            relevant_matrices = [left, down, right, up]
            for i in range(len(relevant_matrices)):
                matr = relevant_matrices[i]
                relevant_matrices[i] = transpose(matr)
            left, down, right, up = relevant_matrices
            temp = left[2]
            left[2] = down[2]
            down[2] = right[2]
            right[2] = up[2]
            up[2] = temp

            for ind, i in enumerate(relevant_matrices):
                i = transpose(i)

                if ind == 0:
                    left = i
                if ind == 1:
                    down = i
                if ind == 2:
                    right = i
                if ind == 3:
                    up = i

        # IF FACE IS BACK (FRONT AND BACK DONT CHANGE)
        elif face == 5:
            """
            relevant_matrices = [self.fLeft, self.fDown, self.fRight, self.fUp]
            for i in relevant_matrices:
                i = transpose(i)
            temp = self.fLeft[0]
            self.fLeft[0] = self.fUp[0]
            self.fUp[0] = self.fRight[0]
            self.fRight[0] = self.fDown[0]
            self.fDown[0] = temp

            for ind, i in enumerate(relevant_matrices):
                i = transpose(i)

                if ind == 0:
                    self.fLeft = i
                if ind == 1:
                    self.fDown = i
                if ind == 2:
                    self.fRight = i
                if ind == 3:
                    self.fUp = i
            """
            relevant_matrices = [left, down, right, up]
            for i in range(len(relevant_matrices)):
                matr = relevant_matrices[i]
                relevant_matrices[i] = transpose(matr)
            left, down, right, up = relevant_matrices
            temp = left[0]
            left[0] = up[0]
            up[0] = right[0]
            right[0] = down[0]
            down[0] = temp

            for ind, i in enumerate(relevant_matrices):
                i = transpose(i)

                if ind == 0:
                    left = i
                if ind == 1:
                    down = i
                if ind == 2:
                    right = i
                if ind == 3:
                    up = i

        newState = [up, down, left, right, front, back]
        return newState



    def rotate(self, state, face, deg):
        """
        Rotates given face on the state by deg, which is an int 1, 2, or 3, by calling
        rotate90 deg times. Returns state', does not modify state or cube.
        """

        for i in range(deg):
            state = self.rotate90(state, face)
        ultimate_list.append((face,deg))
#        self.update(state)
#        cstate = self.currentState()
#        self.prettyPrint2(cstate)
#        total, faces = self.countColors(cstate)
#        print total.items() #sum([x for _, x in total.items()])
        return state

    def getSuccessors(self, state):
        successors = [] #tuple, returns (state after rotation, rotation)
        currentState = self.currentState()
        for f in range(0,6):
            for d in range(1,4):
                successors.append((self.rotate(currentState, f, d), (f,d)))
        return successors


    def scramble(self, k):
        """
        Scrambles cube by calling k rotate functions which select a face and degree
        over a uniform random distribution. Modifies cube faces.
        """

#        currentState = self.currentState()
        for i in range(k):

            face = random.randint(0, 5)
            degree = random.randint(1, 3)
            #print "rotating face {} by {} degrees".format(face, degree*90)
            self.update(self.rotate(self.currentState(), face, degree))

        #print "scrambled {} times:".format(k)
        #self.prettyPrint2(currentState)

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

    def isGoal(self, state):
        """ 
        Returns True if all faces are a solid color.
        """
        colors, faces = self.countColors(state)
        # check if all faces are a solid color
        solid = True
        for face in faces.values():
            if len(face) == 1:
                solid = solid and (face[0][1] == 9)
                if not(solid): #if a face is not a solid color
                    return solid
        return (solid
                and [x[0][1] for x in faces.values()] == [9,9,9,9,9,9]
                and len(set(colors.keys())) == 6)

    def countColors(self, state):
        faces = {}
        colors = defaultdict(int)
        for i,f in enumerate(['up','down','left','right','front','back']):
            currentFace = state[i]
            d = defaultdict(int)
            for row in currentFace:
                for item in row:
                    d[item] += 1
                    colors[item] += 1
            faces[f] = d.items()
        return colors, faces

    def update(self, state):
        """
        Given state, update the faces of the model accordingly.
        """
        self.fUp = deepcopy(state[0])
        self.fDown = deepcopy(state[1])
        self.fLeft = deepcopy(state[2])
        self.fRight = deepcopy(state[3])
        self.fFront = deepcopy(state[4])
        self.fBack = deepcopy(state[5])

    def useDirections(self, directions):
        for f, d in directions:
            state = self.rotate(self.currentState(), f, d)
            self.update(state)

    def prettyPrint(self, state):
        # not actually pretty sorry
        for item, name in zip(state, ['up   ','down ','left ','right','front','back ']):
            for i in range(3):
                text = reduce(lambda x,y: x + ' ' + y, item[i])
                if i == 0:
                    print '\n     ', text
                elif i==1:
                    print name, text
                else:
                    print '     ', text
        print ''

    def prettyPrint2(self, state):
        # not actually pretty sorry
        row1 = '|'
        row2 = '|'
        row3 = '|'
        labels = '   up      down    left   right   front    back'
        for item in state:
            for i in range(3):
                text = reduce(lambda x,y: x+' '+y, item[i])
                if i == 0:
                    row1 += ' ' + text + ' |'
                elif i==1:
                    row2 += ' ' + text + ' |'
                else:
                    row3 += ' ' + text + ' |'
        print row1
        print row2
        print row3
        print labels
        print 'goal state:', self.isGoal(state)

our_cube = Cube()

"""
# test each possible rotation
init = our_cube.currentState()
for f in range(0,4):
    for d in range(1,5):
        print "face:", f, "degree:", 90*d
        our_cube.prettyPrint2(our_cube.rotate(init, f, d))
"""

"""
# THIS JUST PRINTS THE ORDER IN WHICH FACES WERE ROTATED
# THE REVERSE ORDER IS THE SOLUTION TO THE PROBLEM WHICH 
# WILL BE USEFUL FOR CHECKING OUR SOLUTION
"""

our_cube.scramble(4)
scrambled = our_cube.currentState()
soln = [(f, 4-d) for f,d in ultimate_list[::-1]]
bfs = breadthFirstSearch(our_cube)

"""
state = our_cube.currentState()
for f, d in ultimate_list[::-1]:
    print "undoing rotation face {} by {} degrees".format(f, d*90)
    state = our_cube.rotate(our_cube.currentState(), f, 4-d)
"""
