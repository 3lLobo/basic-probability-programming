"""
This is the fifth set of exercises used in the class.

This class is somewhat different. You will create two files.
First, you will create a class which implements vectors. Second, you will create
unittests. unittests establish whether the class works properly or not.

Below you find a skeleton for the code. At the end of the file you find the 
exercises that fill in the details step by step.

"""

import numbers
import math

class Vector(object):
    """
    Vector class.
    """

    def __init__(self, *args):
        for i in args:
            if not isinstance(i, numbers.Number):
                raise ValueError("vectors can only consist of numbers")
        self.vector = tuple(args)

    def __len__(self):
        return len(self.vector)

    def __str__(self):
        return "".join(["Vector", str(self.vector)])

    def __add__(self, other):
        if len(self) != len(other):
            raise TypeError("vectors of different lengths cannot be added together")
        new_vector = []
        try:
            for i in range(len(self.vector)):
                new_vector.append(self.vector[i] + other.vector[i])
        except AttributeError:
            raise TypeError("can only add a vector to a vector")
        else:
            return Vector(*new_vector)

    def __eq__(self, other):
        try:
            if self.vector == other.vector:
                return True
            else:
                return False
        except AttributeError:
            return False

    def scale(self, scalar):
        return Vector(*[i * scalar for i in self.vector])

class ProbVector(Vector):
    """
    Class of probability vector.

    The class normalizes the vector, so that its sum is 1.
    """

    def __init__(self, *args):

        Vector.__init__(self, *args)

        total = sum(self.vector)

        new_vector = []

        for i in args:
            new_vector.append(i/total)

        self.vector = tuple(new_vector)


    def __add__(self, other):
        vector = super().__add__(other)
        return ProbVector(*vector.vector)

    def logprob(self):
        return Vector(*[math.log(i) for i in self.vector])


