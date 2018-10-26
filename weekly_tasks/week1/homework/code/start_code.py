"""
The code that implements the programming assignment 1.
"""

import random #we import the random module

#The following three variable assignments tell us how we encode internal choices in the game
ROCK = 0
PAPER = 1
SCISSORS = 2
#Note that these variable assignments are written with capital letters. By convention in Python, capital letters are used for constant variable assignments (these variables are never re-assigned values, e.g., SCISSORS keeps the value 2 throughout the whole program).

def who_won(value1, value2):
    """
    Did value1 beat value2?
    """
    pass
    #Delete pass and in its place, specify the function; the function should return 1 if value1 beats value2 in rock-paper-scissors; it should return 2 if value2 beats value1; and it should return 0 if it is a draw; keep in mind that value1 and value2 are internally numbers 0, 1 or 2

#you can test the function by specifying a few values, e,g,:
who_won(ROCK, PAPER)

def what_was_played(value):
    """
    Given the value 0, 1, or 2, what was played?
    """
    pass
    #Delete pass and in its place, specify the function; the function should not return anything; it should only print what was played given the value. We assume that 0 = ROCK, 1 = PAPER, 2 = SCISSORS (see also the variables assigned at the start of this code)

#you can test the function by specifying a few values, e,g,:
what_was_played(ROCK)

#Here below you should write the code which plays a game; it should randomly assign 0, 1 or 2 to player1; and 0, 1 or 2 to player2; it should then print what player1 chose and what player2 chose (using the function what_was_played); after that, it should specify who won
    
