"""
The code that implements the programming assignment 2. This is the start. You have to fill in the rest.
"""

import random #we import the random module, to be able to randomly select elements
from collections import Counter #this can be useful if you are going to use Counter as a container for word frequencies

#The code below asks user for path to text file; the answer of the user is stored in the variable textFile
textFile = str(input('Please enter the path to the text file you want to read: '))

#the code below reads in the textfile specified and loops through the file line by line
def collect_frequencies(nameoffile):
    """
    Using nameoffile, the function returns frequency counts of every word in the file called nameoffile. What is returned can be a dictionary or a Counter.
    """
    with open(nameoffile) as text:
        for line in text:
            pass
            #your first task is to substitute pass with a real code. You have to split line into words using white spaces, lower its characters. Then, it has to count all the words just found, creating one dictionary with frequencies stored. The dictionary is then outputed. For example, if the text was this:
            #Mary sleeps.
            #John sleeps too.
            #The dictionary would be: {'mary': 1, 'john': 1, 'sleeps': 2, 'too': 1}. For this task, check the documentation about methods of strings, here:
#https://docs.python.org/3.5/library/stdtypes.html

def find_frequent_words(word_frequencies, amount=50):
    """
    Return two lists. The first list is the list of amount-many most frequent words, ordered by frequency (starting with the most frequent). The second list is the list of corresponding frequencies.

    The first argument of this function, word_frequencies, is the dictionary or the counter of word frequencies, created in the function collect_frequencies.
    """
    pass
    #Delete pass and in its place, specify the function; the function should return two lists - one with words, the other one with frequencies; note that you can return two lists by simply writing return list1, list2

def find_word(word_frequencies, word=None):
    """
    Returns two lists, each list consisting of just one element. The first list has the string identical to the specified word, the second list is its frequency. If no word is specified, a random word is picked.

    The first argument of this function, word_frequencies, is the dictionary or the counter of word frequencies, created in the function collect_frequencies.
    """
    pass
    #Delete pass and in its place, specify the function

def print_lists(list1, list2):
    """
    This function returns nothing, but it prints elements of list1 and list2 as follows (suppose list1 = [el1, el2], and list2 = [el3, el4]):
    el1     el3
    el2     el4
    """
    pass
    #Delete pass and specify the function

#Here below, run the functions, to see that the program outputs all the relevant information:

collected_frequencies = collect_frequencies(textFile)
freq1, freq2 = find_frequent_words(collected_frequencies) #we assign the output, list1 and list2 to variables freq1 and freq2, respectively
word_freq1, word_freq2 = find_word(collected_frequencies) #we assign the output, list1 and list2 to variables word_freq1 and word_freq2, respectively
print_lists(freq1, freq2)
print_lists(word_freq1, word_freq2)
