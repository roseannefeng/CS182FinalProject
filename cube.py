from copy import deepcopy, copy
import sys, os
import random 
import heapq
from collections import defaultdict
import time
import numpy as np

ultimate_list = []


def genFace(label, n):
    return np.array([[label for i in range(n)] for i in range(n)])

class Cube:
    def __init__(self, n):

        self.n = n

        self.fUp = genFace('Y', n)
        self.fDown = genFace('W', n)
        self.fLeft = genFace('R', n)
        self.fRight = genFace('O', n)
        self.fFront = genFace('G', n)
        self.fBack = genFace('B', n)
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
            temp = copy(front[0])
            front[0] = right[0]
            right[0] = back[0]
            back[0] = left[0]
            left[0] = temp

        # IF FACE IS DOWN (UP AND DOWN DONT CHANGE)
        elif face == 1:
            temp = copy(front[-1])
            front[-1] = left[-1]
            left[-1] = back[-1]
            back[-1] = right[-1]
            right[-1] = temp

        # IF FACE IS LEFT (LEFT AND RIGHT DONT CHANGE)
        elif face == 2:
            relevant_matrices = [front, down, back, up]
            for i in range(len(relevant_matrices)):
                matr = relevant_matrices[i]
                relevant_matrices[i] = np.transpose(matr)
            front, down, back, up = relevant_matrices

            temp = copy(front[0])
            front[0] = up[0]
            up[0] = back[0]
            back[0] = down[0]
            down[0] = temp

            for ind, i in enumerate(relevant_matrices):
                i = np.transpose(i)

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
            relevant_matrices = [front, down, back, up]
            for i in range(len(relevant_matrices)):
                matr = relevant_matrices[i]
                relevant_matrices[i] = np.transpose(matr)
            front, down, back, up = relevant_matrices
            temp = copy(front[-1])
            front[-1] = down[-1]
            down[-1] = back[-1]
            back[-1] = up[-1]
            up[-1] = temp

            for ind, i in enumerate(relevant_matrices):
                i = np.transpose(i)

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
            relevant_matrices = [left, down, right, up]
            for i in range(len(relevant_matrices)):
                matr = relevant_matrices[i]
                relevant_matrices[i] = np.transpose(matr)
            left, down, right, up = relevant_matrices
            temp = copy(left[-1])
            left[-1] = down[-1]
            down[-1] = right[-1]
            right[-1] = up[-1]
            up[-1] = temp

            for ind, i in enumerate(relevant_matrices):
                i = np.transpose(i)

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
            relevant_matrices = [left, down, right, up]
            for i in range(len(relevant_matrices)):
                matr = relevant_matrices[i]
                relevant_matrices[i] = np.transpose(matr)
            left, down, right, up = relevant_matrices
            temp = copy(left[0])
            left[0] = up[0]
            up[0] = right[0]
            right[0] = down[0]
            down[0] = temp

            for ind, i in enumerate(relevant_matrices):
                i = np.transpose(i)

                if ind == 0:
                    left = i
                if ind == 1:
                    down = i
                if ind == 2:
                    right = i
                if ind == 3:
                    up = i

        newState = np.array([up, down, left, right, front, back])
        return newState

    def rotate(self, state, face, deg):
        """
        Rotates given face on the state by deg, which is an int 1, 2, or 3, by calling
        rotate90 deg times. Returns state', does not modify state or cube.
        """

        for i in range(deg):
            state = self.rotate90(state, face)
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
        for i in range(k):
            face = random.randint(0, 5)
            degree = random.randint(1, 3)
            self.update(self.rotate(self.currentState(), face, degree))
            ultimate_list.append((face, degree))

    # functions to check states
    def isGoal(self, state):
        """ 
        Returns True if all faces are a solid color.
        """
        colors, faces = self.countColors(state)
        # check if all faces are a solid color
        solid = True
        numEachColor = self.n ** 2
        for face in faces.values():
            if len(face) == 1:
                solid = solid and (face[0][1] == numEachColor)
                if not(solid): #if a face is not a solid color
                    return solid
        return (solid
                and [x[0][1] for x in faces.values()] == [numEachColor] * 6
                and len(set(colors.keys())) == 6)

    def cornersSolved(self, state):
        """
        Returns True if all corners are in the correct position.
        """
        match = True
        for face in state:
            center = face[self.n / 2][self.n / 2]
            c1 = center == face[0][0]
            c2 = center == face[0][-1]
            c3 = center == face[-1][0]
            c4 = center == face[-1][-1]
            match = match and (c1 and c2 and c3 and c4)
        return match

    def edges1Solved(self, state):
        """
        Returns True if a subset of the edges are in the correct position.
        """
        match = True
        mid = self.n / 2
        up, down, left, right, front, back = state
        fcenter = front[mid][mid]
        ucenter = up[mid][mid]
        lcenter = left[mid][mid]
        rcenter = right[mid][mid]
        dcenter = down[mid][mid]
        fc1 = fcenter == front[0][mid]
        fc2 = fcenter == front[mid][0]
        fc3 = fcenter == front[mid][-1]
        fc4 = fcenter == front[-1][mid]
        u1 = ucenter == up[mid][0]
        u2 = ucenter == up[mid][-1]
        u3 = ucenter == up[-1][mid]
        l1 = lcenter == left[0][mid]
        l2 = lcenter == left[mid][-1]
        r1 = rcenter == right[0][mid]
        r2 = rcenter == right[mid][0]
        d1 = dcenter == down[0][mid]

        match = ((fc1 and fc2 and fc3 and fc4) and
                (u1 and u2 and u3) and 
                (l1 and l2) and 
                (r1 and r2) and 
                d1)

        return match

    def edges2Solved(self, state):
        """
        Returns True if the edges not in edges1 are in the correct position.
        """
        match = True
        mid = self.n / 2
        up, down, left, right, front, back = state
        bcenter = back[mid][mid]
        ucenter = up[mid][mid]
        lcenter = left[mid][mid]
        rcenter = right[mid][mid]
        dcenter = down[mid][mid]
        bc1 = bcenter == back[0][mid]
        bc2 = bcenter == back[mid][0]
        bc3 = bcenter == back[mid][-1]
        bc4 = bcenter == back[-1][mid]
        u1 = ucenter == up[0][mid]
        l1 = lcenter == left[mid][0]
        l2 = lcenter == left[-1][mid]
        r1 = rcenter == right[-1][mid]
        r2 = rcenter == right[mid][-1]
        d1 = dcenter == down[mid][0]
        d2 = dcenter == down[mid][-1]
        d3 = dcenter == down[-1][mid]

        match = ((bc1 and bc2 and bc3 and bc4) and
                (u1) and 
                (l1 and l2) and 
                (r1 and r2) and 
                (d1 and d2 and d3))

        return match

    # function for sanity check on rotations
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

    # use to set cube to any given state
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

    # give a specific order of actions to cube
    def useDirections(self, directions):
        for f, d in directions:
            state = self.rotate(self.currentState(), f, d)
            self.update(state)

    def prettyPrint(self, state):
        # not actually pretty sorry
        rows = []
        for i in range(self.n):
            rows.append('|')
        labels = [' up  ',' down',' left','right','front',' back']
        spacing = ' ' + (' ' * 2*(self.n-2))
        labelstr = (' ' * (self.n-1)) + spacing.join(labels)
        for item in state:
            for i in range(self.n):
                text = reduce(lambda x,y: x+' '+y, item[i])
                rows[i] += ' ' + text + ' |'
        for row in rows:
            print row
        print labelstr
        print 'goal state:', self.isGoal(state)



# eyyyy hw1, code for Queue and PriorityQueue from UC Berkeley's Pacman problems
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

class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.

      Note that this PriorityQueue does not allow you to change the priority
      of an item.  However, you may insert the same item multiple times with
      different priorities.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        # FIXME: restored old behaviour to check against old results better
        # FIXED: restored to stable behaviour
        entry = (priority, self.count, item)
        # entry = (priority, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        #  (_, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0


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
        for neighbor in problem.getSuccessors(s):
            s_ = neighbor[0]
            p_ = p + [neighbor[1]]
            if problem.isGoal(s_):
                return p_
            s_ = np.array(s_)
            s_.flags.writeable = False
            hashs_ = hash(s_.data)
            if hashs_ not in explored:
                frontier.push((s_, p_))
                explored.add(hashs_)

    return []

def nullHeuristic(position, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    start = problem.currentState()
    explored = set()
    frontier = PriorityQueue()
    frontier.push((start, []), heuristic(start, problem))

    while not(frontier.isEmpty()):
        s, p = frontier.pop()
        s = np.array(s)
        s.flags.writeable = False
        problem.update(s)
        explored.add(hash(s.data))
        for neighbor in problem.getSuccessors(s):
            s_ = neighbor[0]
            p_ = p + [neighbor[1]]
            if problem.isGoal(s_):
                return p_
            s_ = np.array(s_)
            s_.flags.writeable = False
            hashs_ = hash(s_.data)
            if (hashs_ not in explored):
                frontier.push((s_, p_), len(p) + heuristic(s_, problem))
                explored.add(hashs_)

    return []

def manhattanHeuristic(position, problem):
    """The Manhattan distance heuristic for a Rubik's cube."""
    pos = np.array(position)
    pos.flags.writeable = False
    hpos = hash(pos.data)

    # because of the way we've set this up, 
    # all states needed for finding optimum are in the dist dicts
    if hpos not in distToCorners or hpos not in distToEdges1 or hpos not in distToEdges2:
        return float('inf')
    corners = distToCorners[hpos]
    edges1 = distToEdges1[hpos]
    edges2 = distToEdges2[hpos]

    return max(corners, edges1, edges2)

    #return len(bfsHeuristic(deepcopy(problem)))

def idaStarSearch(problem, heuristic=nullHeuristic):
    """Iterative depth A* search."""
    state = np.array(problem.currentState())
    bound = heuristic(state, problem)
    while True:
        t, p = search(state, [], bound, problem, manhattanHeuristic)
        if t is True:
            return p
        if t == float('inf'):
            return []
        bound = t
    return []

def search(node, path, bound, problem, heuristic):
    f = len(path) + heuristic(node, problem)
    if f > bound:
        return f, path
    if problem.isGoal(node):
        return True, path
    m = float('inf')
    for neighbor, action in problem.getSuccessors(node):
        path_ = path + [action]
        problem.update(neighbor)
        t, p = search(neighbor, path_, bound, problem, heuristic)
        if t is True:
            return True, p
        if t < m:
            m = t
    return m, path



distToCorners = {}
distToEdges1 = {}
distToEdges2 = {}

def computeDistances(depth):
    """ Preprocessing distances for use by manhattanHeuristic. """
    distfromgoal = 0
    cube = Cube(3)
    init = cube.currentState()
    init = np.array(init)
    init.flags.writeable = False
    hashinit = hash(init.data)
    distToCorners[hashinit] = distfromgoal
    distToEdges1[hashinit] = distfromgoal
    distToEdges2[hashinit] = distfromgoal

    frontier = Queue()
    frontier.push((init, distfromgoal))
    explored = set()
    explored.add(hashinit)

    while not(frontier.isEmpty()):
        s, d = frontier.pop()
        if d == depth + 1:
            return explored
        cube.update(s)
        for neighbor, _ in cube.getSuccessors(s):
            state = np.array(neighbor)
            state.flags.writeable = False
            hashstate = hash(state.data)
            if hashstate not in distToCorners:
                if cube.cornersSolved(state):
                    distToCorners[hashstate] = 0
                else:
                    distToCorners[hashstate] = d+1
            if hashstate not in distToEdges1:
                if cube.edges1Solved(state):
                    distToEdges1[hashstate] = 0
                else:
                    distToEdges1[hashstate] = d+1
            if hashstate not in distToEdges2:
                if cube.edges2Solved(state):
                    distToEdges2[hashstate] = 0
                else:
                    distToEdges2[hashstate] = d+1
            if hashstate not in explored:
                frontier.push((state, d+1))
            explored.add(hashstate)

    return


depth = 5
dim = 3
cube = Cube(dim)
print "running code for a {}-dimensional cube up to depth {}".format(depth, dim)
init = cube.currentState()
start = time.time()
e = computeDistances(depth)
end = time.time()
print "preprocessing: {} seconds".format(end - start)
print "states explored:", len(e)

for d in range(1,depth+1):
    print "cube scrambled to depth", d
    cube.scramble(d)
    scrambled = cube.currentState()
    cube.prettyPrint(scrambled)
    soln = [(f, 4-de) for f,de in ultimate_list[::-1]]

    start = time.time()
    bfs = breadthFirstSearch(deepcopy(cube))
    end = time.time()
    print "{} seconds to run BFS".format(end - start)

    start = time.time()
    astar = aStarSearch(deepcopy(cube), manhattanHeuristic)
    end = time.time()
    print "{} seconds to run A*".format(end - start)

    start = time.time()
    idastar = idaStarSearch(deepcopy(cube), manhattanHeuristic)
    end = time.time()
    print "{} seconds to run IDA*".format(end - start)

    print "reverse steps:       ", soln
    print "breadth-first search:", bfs
    print "A* search:           ", astar
    print "IDA* search:         ", idastar

    cube.update(init)
    ultimate_list = []
    print ""




"""
# test each possible rotation
init = cube.currentState()
for f in range(0,6):
    for d in range(1,5):
        print "face:", f, "degree:", 90*d
        cube.prettyPrint(cube.rotate(init, f, d))
"""

"""
# THIS JUST PRINTS THE ORDER IN WHICH FACES WERE ROTATED
# THE REVERSE ORDER IS THE SOLUTION TO THE PROBLEM WHICH 
# WILL BE USEFUL FOR CHECKING OUR SOLUTION

print ultimate_list

# uses ultimate_list to reverse the scramble
state = cube.currentState()
for f, d in ultimate_list[::-1]:
    print "undoing rotation face {} by {} degrees".format(f, d*90)
    state = ube.rotate(cube.currentState(), f, 4-d)
"""
