"""
The code that implements the programming assignment 2. This is the start. You have to fill in the rest.
"""

import random #we import the random module, to be able to randomly select elements
from collections import Counter #this can be useful if you are going to use Counter as a container for word frequencies

#The code below asks user for path to text file; the answer of the user is stored in the variable textFile
# textFile = str(input('Please enter the path to the text file you want to read: '))
textFile = 'weekly_tasks/week2/homework/pg2600.txt'

#the code below reads in the textfile specified and loops through the file line by line
def collect_frequencies(nameoffile):
    """
    Using nameoffile, the function returns frequency counts of every word in the file called nameoffile. What is returned can be a dictionary or a Counter.
    """
    with open(nameoffile) as text:
        list_of_words = []
        for line in text:
            words = line.split()
            list_of_words = list_of_words + words
        list_of_words = [word.lower() for word in list_of_words]

        dict = Counter(list_of_words)
        print(dict)
        return dict

def find_frequent_words(word_frequencies, amount=50):
    """
    Return two lists. The first list is the list of amount-many most frequent words, ordered by frequency (starting with the most frequent). The second list is the list of corresponding frequencies.

    The first argument of this function, word_frequencies, is the dictionary or the counter of word frequencies, created in the function collect_frequencies.
    """
    alphabetically_sorted = sorted(word_frequencies.most_common(amount), key=lambda tup: tup[0])
    final_sorted = sorted(alphabetically_sorted, key=lambda tup: tup[1], reverse=True)
    list1 = [i[0] for i in final_sorted]

    list2 = [i[1] for i in final_sorted]
    return list1, list2


def find_word(word_frequencies, word=None):
    """
    Returns two lists, each list consisting of just one element. The first list has the string identical to the specified word, the second list is its frequency. If no word is specified, a random word is picked.

    The first argument of this function, word_frequencies, is the dictionary or the counter of word frequencies, created in the function collect_frequencies.
    """
    if word == None:
        random_word = random.choice(list(word_frequencies))
        list1 = [str(random_word)]
        list2 = [word_frequencies[random_word]]
    elif word in word_frequencies:
        list1 = [str(word)]
        list2 = [word_frequencies[word]]
    else:
        print('Sorry, this word does not occur in the text')
        list1 = []
        list2 = []

    return list1, list2


def print_lists(list1, list2):
    """
    This function returns nothing, but it prints elements of list1 and list2 as follows (suppose list1 = [el1, el2], and list2 = [el3, el4]):
    el1     el3
    el2     el4
    """

    for item1, item2 in zip(list1, list2):
        print(str(item1) + '     ' + str(item2))




#Here below, run the functions, to see that the program outputs all the relevant information:

collected_frequencies = collect_frequencies(textFile)
print('test for: find_frequent_words, when amount is default(50)')
freq1, freq2 = find_frequent_words(collected_frequencies)
print_lists(freq1, freq2)
print('test for: find_frequent_words, when amount is different, e.g. 5')
freq1, freq2 = find_frequent_words(collected_frequencies, 5)
print_lists(freq1, freq2)

print('test for: find_word, when no word is given')
word_freq1, word_freq2 = find_word(collected_frequencies)
print_lists(word_freq1, word_freq2)
print('test for: find_word, when word is in the text')
word_freq1, word_freq2 = find_word(collected_frequencies, 'the')
print_lists(word_freq1, word_freq2)
print('test for: find_word, when word is not in the text')
word_freq1, word_freq2 = find_word(collected_frequencies, 'blahblahblahblah')
print_lists(word_freq1, word_freq2)

