# CS182 Final Project
Final project for CS182: Artificial Intelligence, Rubik's Cube Solver as search.
Roseanne Feng, Nelson Yanes-Nunez, and Zahra Mahmood

# Instructions
To test this code, run cube.py in terminal.
cube.py is configured to generate a 3x3x3 Rubik's cube and scramble it by selecting 5 moves at random. It then runs three search algorithms to find the optimal solution, or list of actions to unscramble the Rubik's cube.

Note: the bulk of the runtime will come from preprocessing, as well as BFS as the depth of the cube's scrambling increases. For a depth of 5, cube.py has been observed to run for half an hour, the bulk of which comes from preprocessing required for the heuristic used in A* and IDA*.
