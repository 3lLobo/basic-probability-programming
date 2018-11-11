"""
This is the third set of exercises used in class.

The comments (marked by #) explain what you should do. Type your code below each comment.
"""

#We will continue working with a text file, calculating
#probabilities/frequences of words in the text. However, we will now move on to
#the n-gram analysis of the text.
#Say we have the following sentence:
#The boy kissed the frog.
#In the homework, you had to count occurrencies, which would give the following
#dictionary: {"the": 2, "kissed": 1, "boy": 1, "frog": 1}
#Now, we will want to calculate the frequency of a word given a prefix, where
#by prefix, we understand the previous word or words. That is, we want to count
#how often a word follows a particular word or words. Suppose, we assume a prefix of 1.
#Then we want to get:
#{"the": Counter("boy": 1, "frog": 1), "boy": Counter("kissed": 1), "kissed": Counter("the": 1),
#"frog": Counter()}
#We will later use this ngram to generate sentences which sound (almost)
#English-like.

#We will use the package random and counter in this exercise. Later on,
#itertools will become useful. So we import this first:

import random
import itertools
from collections import Counter

#We will first operate with a fake text (only later will we substitute it
#with a book). We will assign the following list to a variable with a recognizable name:

temp_text = ["The boy kissed a frog.", "After kissing it, nothing happened and the boy got angry and left the frog where he kissed it.", "The frog never saw the boy and the boy kissed no frog since then."]

#1. Write a generator, split_text. The generator takes a list of strings
#(sentences). It splits each sentence by space and lowers all characters.
#It yields a list of words, such that the word list corresponds to one
#sentence.

def split_text(text):
    """
    text: list of strings.
    """
    pass

#2. Test that split_text works. The loop below should print one sentence,
#appearing as a list of words, per iteration.

test = split_text(temp_text)

for sentence in test:
    print(sentence)
    print("*********")

#3. Create a function, update_bigrams, that gets two arguments: dict_to_update (a
#dictionary), and a sentence (a list). It updates the dictionary with the
#sentence as follows: for every sequence of words a b, it takes a as the key in
#the dictionary and updates the counter that is the value of the key a by 1 wrt
#the word b. This a void function (nothing is returned).

def update_bigrams(dict_to_update, sentence):
    """
    dict_to_update: dictionary
    sentence: list

    Return bigram_dict updated with bigrams in the sentence. The prefix is the
    key in bigram_dict, values are encoded as Counter.
    """
    pass

#4. Test the function and generator by creating the generator using  temp_text, called
#preprocessed_text. Then, loop through preprocessed_text and update bigram_dict
#by each sentence. After looping through all sentences, the statement below
#should evaluate as True.

bigram_dict = {}

print(bigram_dict == {'it,': Counter({'nothing': 1}), 'boy': Counter({'kissed':
    2, 'and': 1, 'got': 1}), 'frog': Counter({'never': 1, 'since': 1, 'where':
        1}), 'saw': Counter({'the': 1}), 'happened': Counter({'and': 1}),
    'since': Counter({'then.': 1}), 'angry': Counter({'and': 1}), 'nothing':
    Counter({'happened': 1}), 'and': Counter({'the': 2, 'left': 1}), 'never':
    Counter({'saw': 1}), 'he': Counter({'kissed': 1}), 'kissing':
    Counter({'it,': 1}), 'no': Counter({'frog': 1}), 'after':
    Counter({'kissing': 1}), 'kissed': Counter({'it.': 1, 'no': 1, 'a': 1}),
    'left': Counter({'the': 1}), 'where': Counter({'he': 1}), 'the':
    Counter({'boy': 4, 'frog': 2}), 'a': Counter({'frog.': 1}), 'got':
    Counter({'angry': 1})})
    
#5. We want to generalize our strategy to n-grams. Create a new function,
#update_ngram, with 3 arguments: ngram_dict, sentence, prefix. The function
#returns an updated ngram_dict. The function is like update_bigrams,
#but it allows a sequence of any number of words to appear as the prefix. How
#many words should be used is given in prefix.

def update_ngrams(dict_to_update, sentence, prefix=1):
    """
    dict_to_update: dictionary
    sentence: list

    The prefix encodes how many words should be used as keys in the dictionary.
    """
    pass

#6. Test update_ngrams with prefix=2 by creating a generator out of temp_text, called
#preprocessed_text. Then, loop through preprocessed_text and update ngram_dict
#by each sentence. After looping through all sentences, the statement below
#should evaluate as True. (An ngram with prefix=2 is also called a
#trigram.)

ngram_dict = {}

print(ngram_dict == {('got', 'angry'): Counter({'and': 1}), ('left', 'the'):
    Counter({'frog': 1}), ('boy', 'kissed'): Counter({'no': 1, 'a': 1}), ('boy',
        'and'): Counter({'the': 1}), ('kissed', 'a'): Counter({'frog.': 1}),
    ('kissed', 'no'): Counter({'frog': 1}), ('the', 'frog'): Counter({'never':
        1, 'where': 1}), ('nothing', 'happened'): Counter({'and': 1}), ('frog',
            'since'): Counter({'then.': 1}), ('happened', 'and'):
        Counter({'the': 1}), ('boy', 'got'): Counter({'angry': 1}), ('angry',
            'and'): Counter({'left': 1}), ('saw', 'the'): Counter({'boy': 1}),
        ('kissing', 'it,'): Counter({'nothing': 1}), ('frog', 'where'):
        Counter({'he': 1}), ('frog', 'never'): Counter({'saw': 1}), ('and',
            'the'): Counter({'boy': 2}), ('and', 'left'): Counter({'the': 1}),
        ('no', 'frog'): Counter({'since': 1}), ('it,', 'nothing'):
        Counter({'happened': 1}), ('the', 'boy'): Counter({'kissed': 2, 'and':
            1, 'got': 1}), ('never', 'saw'): Counter({'the': 1}), ('he',
                'kissed'): Counter({'it.': 1}), ('where', 'he'):
            Counter({'kissed': 1}), ('after', 'kissing'): Counter({'it,': 1})})

#7. Create a new function, create_prob_ngrams, which takes an ngram_dict as its
#argument and returns a dictionary in which each prefix shows the
#*probability* that the prefix is followed by particular words. That is, instead
#of Counter and counting occurrencies, you should use a dictionary and store
#probabilities). Note that by doing this part, you create conditional probabilities:
#probabilities that some words appear given the prefix.

def create_prob_ngrams(ngram_dict):
    """
    ngram_dict: dictionary
    sentence: list

    Return ngram_dict updated with bigrams in the sentence. The prefix is the
    key in ngram_dict, values are encoded as Counter.
    """
    pass

#8. Test the function. The following should evaluate to True.

ngram_prob = create_prob_ngrams(ngram_dict)

print(ngram_prob == {('got', 'angry'): {'and': 1.0}, ('kissed', 'a'): {'frog.': 1.0}, ('the', 'frog'): {'where': 0.5, 'never': 0.5}, ('nothing', 'happened'): {'and': 1.0}, ('boy', 'and'): {'the': 1.0}, ('angry', 'and'): {'left': 1.0}, ('no', 'frog'): {'since': 1.0}, ('frog', 'where'): {'he': 1.0}, ('happened', 'and'): {'the': 1.0}, ('left', 'the'): {'frog': 1.0}, ('after', 'kissing'): {'it,': 1.0}, ('frog', 'never'): {'saw': 1.0}, ('boy', 'got'): {'angry': 1.0}, ('frog', 'since'): {'then.': 1.0}, ('saw', 'the'): {'boy': 1.0}, ('kissed', 'no'): {'frog': 1.0}, ('the', 'boy'): {'kissed': 0.5, 'and': 0.25, 'got': 0.25}, ('never', 'saw'): {'the': 1.0}, ('it,', 'nothing'): {'happened': 1.0}, ('and', 'the'): {'boy': 1.0}, ('boy', 'kissed'): {'a': 0.5, 'no': 0.5}, ('and', 'left'): {'the': 1.0}, ('where', 'he'): {'kissed': 1.0}, ('he', 'kissed'): {'it.': 1.0}, ('kissing', 'it,'): {'nothing': 1.0}})

#8. Finally, we will consider the function select_word. This function takes a
#dictionary and randomly selects a word. The random selection is based on
#probability: words with a higher probability are selected more likely
#(Hint: to do this, select a random number between 0 and 1, and match it
#against the *cumulative* probability of words).
#(Another hint: to select the random number, consult the package random and
#the documentation on Python:
#https://docs.python.org/3/library/random.html
#To get to the cumulative probability, you have to cumulate
#probabilities of every word. There are various tools in itertools for tasks
#like that. Find a handy function here:
#https://docs.python.org/3/library/itertools.html

def select_word(words):
    """
    words: dictionary, each word having a probability.
    """
    pass

#9. Now, it is the time to use the actual text file. Create a new generator,
#load_and_split_text. The generator opens a file and for every line, it yields
#it, split by spaces and with lower-case characters.

def load_and_split_text(textfile):
    """
    textfile: name of textfile
    """
    with open(textfile) as text:
        pass

#10. Create an ngram_dict and ngram_prob that is based on the text file.

#11. Finally, you should be able to generate a sentence that is somewhat
#English-like, using the created ngram_prob. Start with the word "the"
#and let select_word select the next word. Then input the selected word
#as the following prefix, and so on. Create a "sentence" of length 10.

target_word = "the"

#11. As an extra, we can calculate a second dictionary, for trigram (specifying
#prefix=2). We can then proceed as follows:
#let trigram prob. dictionary select the next word, and if it fails, we fall
#back on bigram. This way, the generated sentence becomes much more natural.

print("******BIGRAM AND TRIGRAM******")

target_words = ("napoleon", "said")
