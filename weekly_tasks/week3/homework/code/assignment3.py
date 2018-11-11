from collections import Counter
import json
print_steps = True
def step(message):
    if print_steps:
        print("\n" + "*"*70 + '\n{:*^70}\n'.format(message))


#####################################################################
# In this assignment we build a simple information retrieval tool,
# that will search through a corpus of around 80 documents for the
# document most similar to a given 'query'.
#
# We prepared a dictionary `corpus` with info about the corpus we will
# work with. Throughout the homework, we add more info to it, such as
# the word frequencies per document. Every document has a unique
# identifier such as '1789-Washington'.
#
# To speed up things, use a subset of documents during development:
corpus = json.load(open('corpus/corpus-subset.json', 'r')) # Subset
# corpus = json.load(open('corpus/corpus.json', 'r'))   # All docs

# Play around with `corpus`; make sure you understand its structure
# e.g. print(corpus["1789-Washington"])

### Split text and load frequencies
# We've included two functions to split text and extract word frequencies
# from a given text file. You've seen how to make those last week.

def split_text(text):
    """
    Split the string in words (tokens)
    :param text: string
    :return: list
    """
    return text.lower().split(' ')

def get_file_freqs(filename):
    """
    Get the word frequencies in a file
    :param filename:
    :return: Counter
    """
    freqs = Counter()
    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            words = split_text(line)
            freqs.update(words)

    return freqs


#####################################################################

####
step("1. Collect the corpus frequencies")
# Make a Counter object `corpus_freqs` with the corpus (word) freqs,
# the freqs of words in all documents combined. You can use the
# function get_file_freqs and Counter.update() might be useful.
corpus_freqs = Counter()
for docid, info in corpus.items():
    pass

print('Number of unique words in corpus:', len(corpus_freqs))


####
step("2. Make vocabulary")
# To scale things down, we only consider some of the most frequent
# words in the corpus. Use the method Counter.most_common() to make
# a list (!) called `vocabulary` which contains the voc_size=100 most
# common words in the corpus
voc_size = 100
vocabulary = [] # <-- change this


####
step("# 3. Collect vocabulary word frequencies")
# We are only interested in the frequency of words in the vocabulary.
# Write a function that, given a frequency counter and the vocabulary
# returns a list (!) of frequencies of the words in the vocabulary.
# So if freqs['book'] = 10 and 'book' is the 3rd word in the vocabulary,
# then the function should output a list with 10 as the 3rd item.
def freqs_to_vector(freqs, vocabulary):
    """
    Turns a frequency Counter into a list (!) of frequencies, in the
    same order as the words in vocabulary.
    :param freqs: a Counter with word frequencies
    :param vocabulary: a list of vocabulary words
    :return: a list with the frequencies of each of the voc. words
    """
    pass


####
step("4. Collect vocabulary word frequencies")
# Store the frequency vector of every document in the `corpus` object
# as `corpus[doc_id]['freqs']` (remember `corpus` is a dictionary of
# dictionaries). You first have to read out the frequencies again
# using `get_file_freqs`, and then you can use `freq_to_vector`.

# ...


####
step("5. Norm")
# Define a function that returns the norm of a vector (list of numbers).
# So e.g. norm([3, 4]) = sqrt( 3^2 + 4^2 ) = 5. You can use the function
# `math.sqrt` for the square root, and `sum(my_list)` is also useful
def norm(vector):
    """
    Computes the norm of a list of numbers
    :param vector: a list of numbers
    :return: a number
    """
    pass

#Here are some tests that your norm function should pass
print("\nTest 5: norm:")
print( norm([3, 4]) ) # Should be 5.0
print( norm([1, 1, 1, 1])) # Should be 2.0
print( norm([5, 20, 102, 9, 1])) # Should be 104.45...

####
step("6. Cosine similarity")
# Write a function that computes the cosine similarity of two vectors
# A = [a_1, ..., a_n] and B = [b_1, ..., b_n]. Recall that the cosine
# similarity is defined as
#   sim = (a_1 * b_1 + ... + a_n * b_n) / ( norm(A) * norm(B) )
def similarity(A, B):
    """
    Computes the cosine similarity between two vectors (of equal length)
    :param A: a vector (list of numbers)
    :param B: another vector (list of numbers)
    :return: the cosine similarity (a number between -1 and 1)
    """
    pass

#Here are some tests that your similiary function should pass
print("\nTest 6: cosine similarity:")
print( similarity([1,0,0], [1,0,0]) ) # Should be 1.0
print( similarity([1,0,0], [0,1,0]) ) # Should be 0.0
print( similarity([1,0,0], [-1,0,0]) )# Should be -1.0
print( similarity([1,1,.5], [0.5,2,0]))# Should be 0.808...

####
step("7. Compute cosine similarities")
# Compute the cosine similarity between three documents:
# the first and second inaugural address by Washington
# (id's "1789-Washington" and "1793-Washington") and the
# poetry collection 'Leaves of grass' by Walt Whitman
# (id "whitman-leaves").
# washington1 = ...
# washington2 = ...
# whitman = ...

# print("\nCosine similarities")
# print("Washington 1 vs Washington 2:", ... )
# print("Washington 1 vs Whitman:", ... )
# print("Washington 2 vs Whitman:", ... )


####
step("8. Arbitrary queries")
# We want to use the cosine similarity to automatically find
# the document most similar to a 'query', that is, we want
# to build a kind of search engine. Write a function that turns
# a query string into a frequency vector. You probabl want to use
# the functions split_text and freqs_to_vector we defined before.
def text_to_vector(text, vocabulary):
    """
    Turns a string into a vector (list) of word frequencies for those
     words in the vocabulary
    :param text: the input string
    :param vocabulary: the list of vocabulary words
    :return: a list of word-frequencies
    """
    pass

# We have already written the function that ranks the documents for you
def rank_documents(query_vector, corpus, num=100):
    """
    Ranks the documents according to their cosine similarity to a query vector.
    :param query_vector: list
    :param num: only return the `num` top ranking documents
    :return: two lists: a list of ranked document ids (most similar first) and a
        list with the corresponding similarity scores
    """
    similarities = {}
    for doc_id, info in corpus.items():
        freq_vect = corpus[doc_id]['freq_vect']
        similarities[doc_id] = similarity(query_vector, freq_vect)

    ranked_ids = sorted(similarities, key=lambda i: similarities[i], reverse=True)
    ranked_sims = [similarities[id] for id in ranked_ids]
    return ranked_ids[:num], ranked_sims[:num]


####
step("9. Rank documents")
# Use the functions text_to_vector and rank_documents to find the document
# closest to the following excerpt from Adams inaugural address.

adams_txt = "When it was first perceived, in early times, that no middle \
course for America remained between unlimited submission to a foreign \
legislature and a total independence of its claims, men of reflection \
were less apprehensive of danger from the formidable power of fleets \
and armies they must determine to resist than from those contests and \
dissensions which would certainly arise concerning the forms of government \
to be instituted over the whole and over the parts of this extensive country."

#...

# Do play around with our querying system. To use the full collection,
# rather than the 3 corpus we used so far, uncomment the line
# `corpus = json.load(open('documents.json', 'r'))` at the top of this file.
# You note that our system isn't very reliable, and can be improved in
# many ways. The first thing you would want to do is tackle stop-words.
#
# If you look at the vocabulary (print it!) it contains many words like
# 'and', 'of', 'it', and so on. These are highly frequent in all texts,
# and not informative for a document's content. There are at least two
# improvements: (1) remove such words from the vocabulary, or (2) adjust
# our vectors to be less 'sensitive' to those words.
#
# A common approach to (2) is to use so called tf-idf vectors, which stands
# for (term frequency)-(inverse document frequency). The inverse document
# frequency roughly punishes words that occur in many of the documents
# in the corpus. It is fairly easy to extend this assignment to use
# tf-idf scores instead. If you're interested, look up the Wikipedia page
# https://en.wikipedia.org/wiki/Tf%E2%80%93idf or ask Bas if you need help.
