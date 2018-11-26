"""
This is the fifth set of exercises used in the class.

This class is somewhat different. You will create two files.
First, you will create a class which implements vectors. Second, you will create
unittests. unittests establish whether the class works properly or not.

Below you find a skeleton for the code and the instructions on the exercises.

"""

#we will use unittests: to get an idea how that works, you can read about it
#here:
#https://docs.python.org/3/library/unittest.html
import unittest

#here specify the file in which you create the code
from inclass5 import Vector
from inclass5 import ProbVector

#############################################################################
############################# E X E R C I S E S #############################
#############################################################################

#class Vector

class TestVector(unittest.TestCase):

# Exercise 1
# We will start by filling in the init method of the class Vector.
# The init method is partly pre-specified (it raises an exception if we try to
# input something else than a number). What needs to be done is to store all the
# values passed to the init method in the instance attribute vector. Note that
# the values should be stored as an iterable. Afterwards, you should be able to
# pass the test test_setup

    def test_setup(self):
        self.assertEqual((2, 3, 4), tuple(Vector(2, 3, 4).vector))
        self.assertEqual((2.2, 3.3, 4.4), tuple(Vector(2.2, 3.3, 4.4).vector))
        self.assertNotEqual((2.2, 3.3), tuple(Vector(2.2, 3.3, 4.4).vector))
        self.assertEqual((2.2, ), tuple(Vector(2.2).vector))
        self.assertRaises(ValueError, lambda x: Vector(x), 'a')
        self.assertEqual(tuple(), tuple(Vector().vector))

### Exercise 2
# The __len__ method should return the length of the attribute vector. Note
# that this method is called by function len().

    def test_length(self):
        for i in range(10):
            self.assertEqual(i, len(Vector(*[x for x in range(i)])))

# Exercise 3
# The __add__ method should add two instances of Vector pointwise. It proceeds
# as follows: first, it raises an error (TypeError) if the two vectors are of different
# lengths. If they are of the same length, it adds them up position by position.
# If the second element is not a vector (it lacks the attribute vector), the
# method raises TypeError. Note that we only test basic cases. You might want to
# add more tests (for example, what should addition of two zero vectors return?)

    def test_add(self):
        self.assertEqual(Vector(5, 5), Vector(2, 3) + Vector (3, 2))
        self.assertRaises(TypeError, lambda x, y, z: Vector(x) + Vector(y, z),
        1, 2, 3)
        self.assertRaises(TypeError, lambda x: Vector(x) + 7, 1)

# Exercise 4
# The __eq__ method should compare two vectors (it is used when we compare
# elements, by '==' and '!='). It should return True if the two
# vectors match (e.g., Vector(4, 4, 3) == Vector(4, 4, 3) is True). It should
# return False if the two vectors do not match. It should also return False if
# we try to compare a vector with something else (for instance, a tuple with the
# same elements). Note that we only have one test here and we do not test all
# possibilities. Add more tests to make sure you cover all possibilities!
# 

    def test_eq(self):
        self.assertTrue(Vector(2, 3, 4) == Vector(2, 3, 4))

# Exercise 5
# The scale method should scale a vector by a scalar (a scalar is a number),
# e.g., Vector(2, 3, 4).scale(10) = Vector(20, 30, 40). It should raise
# ValueError if we try to scale a vector by other things than a scalar, e.g., a
# string.
# Note that you should add tests here to cover all your bases.

    def test_scale(self):
        self.assertEqual(Vector(20, 30, 40), Vector(2, 3, 4).scale(10))

# Exercise 6
# If you get to this point, share now your unittest with your colleague! You
# should be able to pass their tests (for the class Vector) and they should be able to
# pass your tests (for the class Vector).


#class ProbVector

class TestProbVector(unittest.TestCase):

# Exercise 7
# The instance of ProbVector should be initialized in the same way as Vector.
# However, rather than storing just the sequence of numbers in the attribute
# vector, it stores the sequence of normalized numbers (they have to sum up to
# 1 if the vector is non-zero).
#
    
    def test_setup(self):
        self.assertEqual(1, sum(ProbVector(3, 7, 2).vector))
        self.assertEqual(Vector(0.2, 0.3, 0.5), ProbVector(2, 3, 5))
        self.assertRaises(ValueError, lambda x: ProbVector(x), 'a')

# Exercise 8
# The __add__ method adds two vectors together and returns a new prob. vector
# (that is, a vector that is normalized). Note that only one basic case is
# tested. You should add more tests! (For example, an error should be raised in
# cases we try to add different classes, as in the case of Vector; also, we only
# test right now that the addition is normalized to 1, but we do not test that
# the addition returns a prob. vector we would expect - this should be tested!).
    
    def test_add(self):
        simple_probvector = ProbVector(2, 3) + ProbVector (3, 2)
        self.assertEqual(1, sum(simple_probvector.vector))

# Exercise 9
# Finally, we turn the ProbVector instance into a Vector instance in which each
# value in the vector is log-transformed. Do that in the function logprob. (That
# is, the function should return an instance of class Vector with
# log-transformed probabilities.)

    def test_logprob(self):
        pass


# Exercise 10
# If you get to this point, share now your unittest with your colleague! You
# should be able to pass their tests (for the class ProbVector) and they should be able to
# pass your tests (for the class ProbVector).

if __name__ == '__main__':
    unittest.main()
