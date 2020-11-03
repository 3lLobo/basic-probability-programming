"""
The code that implements the programming assignment 1.
"""

import random #we import the random module
import numpy as np


#The following three variable assignments tell us how we encode internal choices in the game
ROCK = 0
PAPER = 1
SCISSORS = 2

game_dict = {0: 'rock', 1: 'paper', 2: 'scissors'}
#Note that these variable assignments are written with capital letters. By convention in Python, capital letters are used for constant variable assignments (these variables are never re-assigned values, e.g., SCISSORS keeps the value 2 throughout the whole program).

def who_won(value1, value2):
    """
    Did value1 beat value2?
    :param value1: choice of player1
    :param value2: choice of player2
    """
    #Delete pass and in its place, specify the function; the function should return 1 if value1 beats value2 in rock-paper-scissors; it should return 2 if value2 beats value1; and it should return 0 if it is a draw; keep in mind that value1 and value2 are internally numbers 0, 1 or 2

    # Condition for draw.
    if value1 == value2:
        return 0

    # We chose to make to separate if statements instead of elif, to avoid computation of the 3 winning conditions in case of a draw.
    # Condition for player1 winning.
    r_vs_s = value1 == 0 and value2 == 2
    p_vs_r = value1 == 1 and value2 == 0
    s_vs_p = value1 == 2 and value2 == 1
    if r_vs_s or p_vs_r or s_vs_p:
        return 1
    # Condition for player2 winning.
    else:
        return 2 
         

#you can test the function by specifying a few values, e,g,:
who_won(ROCK, PAPER)

def what_was_played(value):
    """
    Given the value 0, 1, or 2, what was played?
    :param value: choice of gamemove
    """
    #Delete pass and in its place, specify the function; the function should not return anything; it should only print what was played given the value. We assume that 0 = ROCK, 1 = PAPER, 2 = SCISSORS (see also the variables assigned at the start of this code)
    print('{} was played.'.format(game_dict[value]))

win_dict = {0: 'It\'s a draw.', 1: 'Player1 wins.', 2: 'Player2 wins'}
#you can test the function by specifying a few values, e,g,:
what_was_played(ROCK)

#Here below you should write the code which plays a game; it should randomly assign 0, 1 or 2 to player1; and 0, 1 or 2 to player2; it should then print what player1 chose and what player2 chose (using the function what_was_played); after that, it should specify who won

n = 6 # game iterations
seed = 11
np.random.seed(seed=seed) # set seed for reproducibility

for i in range(n):
    print('\n Playing game {}.'.format(i+1))
    p1, p2 = int(np.random.choice(3,1)), int(np.random.choice(3,1))
    print('Player 1:')
    what_was_played(p1)
    print('Player 2:')
    what_was_played(p2)
    print(win_dict[who_won(p1,p2)])

