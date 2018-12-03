import numpy as np

def EXERCISE(number, ex=None):
    print("\n======== Exercise {} =========".format(number))


############# VECTORS #############

# Working with vectors is easy in NumPy. The only thing is that they're called 'arrays'.
# Let's build some:
a = np.array([10, 10, 10, 10])
b = np.array([1, 2, 3, 4])
print("a:", a)
print("b:", b)

# You can manipulate arrays with so called 'entry-wise' operations. These are operations
# applied to all entries (numbers) in an array separately, and the result is always an array of
# the same size. Let's for example raise b to the power 2 and 3:
print("Squares:", b ** 2)
print("Cubes:", b ** 3)

# Most basic mathematical operations are built into numpy. For example:
print("Cosine:", np.cos(b))
print("Exponential:", np.exp(b))


EXERCISE(1, "Can you use numpy to compute the square roots of numbers in b?")
#...

EXERCISE(2, "Try multiplying b by a number. What happens What if you add a number?")
# ...

# Not all operations are entry-wise. The sum or product of an array are not, for example:
print("Sum:", np.sum(a))
print("Product:", np.prod(a))

# Instead of using the function `np.sum`, we can also use the method `.sum()`.
print("Sum:", a.sum())
print("Product:", a.prod())

EXERCISE(3, "Not all functions are implemented as methods: try running `a.cos()`")
# ...

EXERCISE(4, " Suppose b contains word frequencies. Turn it into a probability distribution, \
    described by log-probabilities. (Can you do the computations using 17 characters of code?)")
# logprobs = ...

# print("Test:", np.exp(logprobs).sum() == 1)

# You can also add vectors:
c = a + b
# print("c:", c)

EXERCISE(5, "Try multiplying a and b. Is this operation entry-wise? \
    How about the so called dot product `np.dot(a, b)`?")


############# MATRICES #############

# Vectors are lists of numbers, matrices tables of numbers. You can also think of vectors
# as tables, but with only one row/column. That is precisely what numpy does, and that's
# why it calls all these things 'arrays'.
my_matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
# print("my_matrix:\n", my_matrix)

# You can also construct a matrix from other arrays:
my_other_matrix = np.array([a, b, c])
# print("my_other_matrix:\n", my_matrix)

# Entry-wise operations that worked for vectors, also work for matrices.
EXERCISE(6, "Compute the cosine of the square of my_matrix")

# Operations that are not entry-wise can be a bit more tricky for matrices. Take the sum.
# You can  sum *all*  entries in the matrix (which gives a number), sum all rows (which gives
# a vector with the same number of columns), or sum all columns. Compare the following:
# print(my_matrix.sum())
# print(my_matrix.sum(axis=0))
# print(my_matrix.sum(axis=1))

# To get individual entries from an array (this is called indexing) you use square brackets,
# just as with lists. But there is a difference: a matrix for example has both rows and columns,
# and you need to specify both. So `my_matrix[i, j]` selects the entry in row i, column j
# (the first row/column is numbered 0):
# print("Entry (1, 1):", my_matrix[1, 1])

EXERCISE(7, "can you select the 6 in my_matrix?")
#...

# You can also select a slice of the array such as the second row, or the last two columns.
# This is called slicing, and it's super useful, but can also be confusing. The most important
# thing is the colon ":" which is perhaps best explained by some examples.
#
# For more info: https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html

# Can you make sense of this?
# print("First row:",                      my_matrix[0, :])
# print("Second row:",                     my_matrix[1, :])
# print("Third row:",                      my_matrix[2, :])
# print("First column:",                   my_matrix[:, 0])
# print("Last column:",                    my_matrix[:, 2])
# print("Last column, again:",             my_matrix[:, -1])
#
# # And how about these?
# print("\nFirst two rows:\n",             my_matrix[:2, :])
# print("Last two rows:\n",                my_matrix[1:, :])
# print("Last two rows, first column:\n",  my_matrix[1:, 0])

EXERCISE(8, "Select the first two rows of the second column: (2,5)")

# Here's another useful one: you can also slice by indicating which rows/columns
# to select using a list of Booleans. Can you make sense of this?
# select = [True, False, True]
# print(my_matrix[select, :])
# print(my_matrix[:, select])

# We will need this later today.
