"""
This is the second set of exercises used in class.

The comments (marked by #) explain what you should do. Type your code below each comment.
"""

import random
import collections

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

DECK = []

for i in RANKS:
    for j in SUITS:
        DECK.append(i+j)

#2. If everything went fine, you just created a list DECK by looping
#through two lists, RANKS and SUITS. In general if you loop through
#iterables only to create a new one, there is a much more readable and
#preferable way to proceed: using list comprehensions. Read this
#explanation:
#http://www.secnetix.de/olli/Python/list_comprehensions.hawk
#After that, re-write the deck creation, DECK2, by using list
#comprehension. The new way of creating the dect should be just a
#single line.

DECK2 = [i+j for i in RANKS for j in SUITS]

#3. DECK and DECK2 should be equal (DECK == DECK2 should evaluate to
#True). Check this.

print(DECK == DECK2)

#List comprehensions are very very common. Try to use them throughout
#the whole course.

#4. Test the length of DECK and DECK2. It should consist of 52 elements.

print(len(DECK))
print(len(DECK2))

#5. Create a function, called create_deck, which takes two
#arguments, ranks and suits and creates a deck of cards, assigned to
#the variable deck. The deck is 
#shuffled and returned. To make this work, you should use a
#function from the package random. To find the right function, see
#the documentation here:
#https://docs.python.org/3.5/library/random.html

def create_deck(ranks, suits):
    deck = []
    for i in ranks:
        for j in suits:
            deck.append(i+j)
    random.shuffle(deck)
    return deck

#6. Check that the shuffling works: compare DECK to create_deck(RANKS,
#SUITS). The comparison should return False (it is extremely unlikely
#that shuffling would return the same order of cards as in the
#originally created DECK. Check also that the shuffled deck still has
#52 cards.

print(DECK == create_deck(RANKS, SUITS))

print(len(create_deck(RANKS, SUITS)))

#7. Create a new function, deal_hand, that (i) makes a shuffled deck
#out of ranks and suits, (ii) draws the first 5 cards from this deck
#and it returns this hand. Of course, you should reuse the function
#create_deck for this. To get 5 cards, you should use slicing (see Think
#Python).

def deal_hand(ranks, suits):
    new_deck = create_deck(ranks, suits)
    return new_deck[:5]

#8. Check that the function deal_hand works. Run it a few times and
#check that it returns various hands.

for _ in range(10):
    print(deal_hand(RANKS, SUITS))

print("*********")

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
    #the following line keeps only information about ranks in hand
    hand_ranks = [i[0] if len(i) == 2 else i[:2] for i in hand]
    #if statement is there to keep two letters in case of '10'
    count_rank = {}
    for i in hand_ranks:
        if i in count_rank:
            count_rank[i] += 1 #increase count if the rank is present
        else:
            count_rank[i] = 1 #start count if the rank is not yet present
    return count_rank

#Note: I thought it might be easier for some of you to loop
#through all RANKS and count instances in the hand; the code for
#this alternative is given below
def check_rank(hand):
    hand_ranks = [i[0] if len(i) == 2 else i[:2] for i in hand]
    count_rank = {}
    for i in RANKS:
        if hand_ranks.count(i): 
            count_rank[i] = hand_ranks.count(i)
    return count_rank

#There is another solution, which would probably be the most 
#straightforward one for Python users; it involves counters, which
#are dictionaries used for incremental (one by one) counting

def check_rank(hand):
    hand_ranks = [i[0] if len(i) == 2 else i[:2] for i in hand]
    count_rank = collections.Counter(hand_ranks) #this creates a counter
    #counters are dictionaries; you can supply a list to them
    #they will count all instances in the list
    return count_rank

#10. Check that the function works by running a few random draws and
#inspecting the resulting dictionary.

my_hand = deal_hand(RANKS, SUITS)
print(my_hand)
print(check_rank(my_hand))

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
#ranks are shared)?

#When you debug the code, run at least 100,000 simulations. Your
#result should be close to 42.25 percent. See also:
#https://en.wikipedia.org/wiki/Poker_probability

success = 0

runs = 100000

for _ in range(runs):
    my_hand = deal_hand(RANKS, SUITS)
    vals = check_rank(my_hand)
    if len(vals) == 4:
        success = success + 1

print("Probability of exactly one pair", success/runs)

#12. What is the probability of getting a full house (three of a kind
#and a pair)? You can compare your simulation to the answer given on
#the Wikipedia page.

success = 0

for _ in range(runs):
    my_hand = deal_hand(RANKS, SUITS)
    vals = list(check_rank(my_hand).values())
    if max(vals) == 3 and 2 in vals:
        success = success + 1

print("Probability of a full house", success/runs)

#13. So far, we considered cases that are not that difficult to solve
#analytically. But consider the following one:
#We have a player that cheats. He gets a hand (5 cards) and inspects
#it. If it turns out that the hand has no two or more cards of the
#same rank, he throws away these cards and gets the next 5 cards. He
#then plays with this new hand (no matter what is in it). What is the
#probability of getting a flush in this case?
#Note: first, create a new function, deal_hand_mod, which is like the
#old one but implements this new situation. After that, run the
#simulation. Your result should be around 0.215 percent.
#Note that while calculating this would be far from trivial, creating
#a simulation is not significantly more difficult than running MC
#simulations in previous cases.

def deal_hand_mod(ranks, suits):
    new_deck = create_deck(ranks, suits)
    rank_dictionary = check_rank(new_deck[:5])
    if max(list(rank_dictionary.values())) == 1:
        return new_deck[5:10]
    else:
        return new_deck[:5]

success = 0

for _ in range(runs):
    my_hand = deal_hand_mod(RANKS, SUITS)
    vals = list(check_rank(my_hand).values())
    if max(vals) == 3 and 2 in vals:
        success = success + 1

print("Probability of a full house in cheating", success/runs)
