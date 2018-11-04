"""
This is the second set of exercises used in class.

The comments (marked by #) explain what you should do. Type your code below each comment.
"""

import random

#In this set of exercises, we will practice working with strings and
#iterables. We will also practice working with random sampling.

#1. Create a deck of cards. Take the variables RANKS and SUITS and
#make a deck of cards out of it, DECK (each element in DECK should
#be a string in which the first element is from RANKS, the second
#element is from SUITS, and DECK exhausts all combinations).
#Note that RANKS are individual ranks, and in suits, C=clubs,
#D=diamonds, H=hearts, S=spades.

RANKS = ['A', '2', '3', '4', '5', '6', '7',
             '8', '9', '10', 'J', 'Q', 'K']
SUITS = ['C', 'D', 'H', 'S']

#2. If everything went fine, you just created a list DECK by looping
#through two lists, RANKS and SUITS. In general if you loop through
#iterables only to create a new one, there is a much more readable and
#preferable way to proceed: using list comprehensions. Read this
#explanation:
#http://www.secnetix.de/olli/Python/list_comprehensions.hawk
#After that, re-write the deck creation, DECK2, by using list
#comprehension. The new way of creating the dect should be just a
#single line.

#3. DECK and DECK2 should be equal (DECK == DECK2 should evaluate to
#True). Check this.

#List comprehensions are very very common. Try to use them throughout
#the whole course.

#4. Test the length of DECK and DECK2. It should consist of 52 elements.

#5. Create a function, called create_deck, which takes two
#arguments, ranks and suits and creates a deck of cards, assigned to
#the variable deck. The deck is 
#shuffled and returned. To make this work, you should use a
#function from the package random. To find the right function, see
#the documentation here:
#https://docs.python.org/3.5/library/random.html

def create_deck(ranks, suits):
    pass

#6. Check that the shuffling works: compare DECK to create_deck(RANKS,
#SUITS). The comparison should return False (it is extremely unlikely
#that shuffling would return the same order of cards as in the
#originally created DECK. Check also that the shuffled deck still has
#52 cards.

#7. Create a new function, deal_hand, that (i) makes a shuffled deck
#out of ranks and suits, (ii) draws the first 5 cards from this deck
#and it returns this hand. Of course, you should reuse the function
#create_deck for this. To get 5 cards, you should use slicing (see Think
#Python).

def deal_hand(ranks, suits):
    pass

#8. Check that the function deal_hand works. Run it a few times
#(e.g., 10 times, as suggested in the code below) and
#check that it returns various hands.

for _ in range(10):
    pass

#9. We now want to analyze whether a dealt hand has the same rank of
#cards and how many of the same cards there are.
#The function, check_rank, will take a hand as its argument, and it
#returns the dictionary with counts of the same ranks (that is, the hand ['QS',
#'3C', '3S', '10S', '8H'] would lead to {'Q': 1, '3': 2, '10': 1,
#'8': 1}; the hand ['QS', '3C', '4S', '10S', '8H'] would lead to the
#return value of {'Q': 1, '3': 1, '4': 1, '10': 1, '8': 1} etc.)
#Note: it is handy to use the method count from lists. Check the
#documentation to see what the method does.

def check_rank(hand):
    pass

#10. Check that the function works by running a few random draws and
#inspecting the resulting dictionary.


#With the mathematical rules from probability theory you can
#compute the probability that an event happens. Unfortunately, such
#computations quickly become impossible if we consider more and more
#complex scenarions, which we usually have to to get to realistic
#cases. There is a simple numerical way of computing probabilities
#that is applicable to problems with uncertainty. The principal ideas
#of this is to run a simulation and record the outcome. Suppose the
#succes happens M times and we run the simulation N times. Then, the
#probability is M/N, and it becomes more and
#more accurate (closer to the true probability) as we increase N. The
#mathematical technique of letting the computer perform lots of
#experiments based on drawing random numbers is commonly called Monte
#Carlo simulation. We will now consider a few Monte Carlo (MC) simulations
#with our deck of cards (in fact, we already did one MC simulation in
#the last class).

#11. What is the probability that there is only one pair (and no other
#ranks are shared)? Write the simulation using the for loop.
#After you debugged the code, run the simulation with N=100,000 (or more). Your
#result should be close to 42.25 percent. See also:
#https://en.wikipedia.org/wiki/Poker_probability


#12. What is the probability of getting a full house (three of a kind
#and a pair)? You can compare your simulation to the answer given on
#the Wikipedia page.

#13. So far, we considered cases that are not that difficult to solve
#analytically. But consider the following one:
#We have a player that cheats. He gets a hand (5 cards) and inspects
#it. If it turns out that the hand has no two or more cards of the
#same rank, he throws away these cards and gets the next 5 cards. He
#then plays with this new hand (no matter what is in it). What is the
#probability of getting a flush in this case?
#Note: first, create a new function, deal_hand_mod, which is like the
#old one but implements this new situation. After that, run the
#simulation using deal_hand_mod. Your result should be around 0.215 percent.
#Note that while calculating this would be far from trivial, creating
#a simulation is not significantly more difficult than running MC
#simulations in previous cases.

