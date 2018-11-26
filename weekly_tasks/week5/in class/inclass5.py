"""
This is the fifth set of exercises used in the class.

This class is somewhat different. You will create two files.
First, you will create a class which implements vectors. Second, you will create
unittests. unittests establish whether the class works properly or not.

Below you find a skeleton for the code. At the other file you find the 
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
        #Ex 1: ...
        pass

    def __len__(self):
        #Ex 2: ...
        pass

    def __str__(self):
        return "".join(["Vector", str(self.vector)])

    def __add__(self, other):
        #Ex 3:...
        pass

    def __eq__(self, other):
        #Ex 4:...
        pass

    def scale(self, scalar):
        #Ex 5:...
        pass

class ProbVector(Vector):
    """
    Class of probability vector.

    The class normalizes the vector, so that its sum is 1.
    """

    def __init__(self, *args):

        #Ex 7:...

        #The line below calls the init method of the parent. This is useful,
        #as you create all the attributes of the parent class this way.
        Vector.__init__(self, *args)

    def __add__(self, other):
        #Ex 8:...
        pass

    def logprob(self):
        #Ex 9:...
        pass

