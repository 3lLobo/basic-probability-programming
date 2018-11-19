"""
This is the fourth set of in class exercises. We will implement two classes for 
Bernoulli and Binomial random variables and a possiblity of adding those. 
At the end, the following code should do precisely what you expect:

```
X = Bernoulli(0.6)
Y = Bernoulli(0.6)
A = X + Y                   # --> A ~ Binomial(2, 0.6)
B = Binomial(5, 0.6) + A    # --> B ~ Binomial(7, 0.6)
C = Y + B                   # --> C ~ Binomial(8, 0.6)
```

Below you find a skeleton for the code. At the end of the file you find the 
exercises that fill in the details step by step. Exercises 11, 12 and 13 are
most important, so try to work through the first part quickly (you can skip
ex 9 and 10 if you're running out of time).
"""

def is_probability(p):
  """
  Tests if p is a probability, that is, it tests that p is a number in the interval [0, 1].
  The function returns True if this is the case and False otherwise.
  """
  # Ex 1: ...
  return True

def is_natural_number(n):
  """Test if n is a natural number"""
  
  # Ex 7: ...
  return True


#############################################################################


class Bernoulli(object):
  """A Bernoulli random variable"""

  def __init__(self, p):
    # Test if p is a valid parameter. If not, print a error message and exit the program.
    # We will discuss error handling next week. For now, it's enough to know that 
    # raise ValueError("my message") prints the error message "my message" and exits.
    if not is_probability(p):
      raise ValueError("The parameter p should be a number between 0 and 1")

    self.p = p

  def __str__(self):
    """Returns a string of the form "Bernoulli(p)", where p is the value of the parameter."""
    # Ex 6: ...
    pass

  def __add__(self, other):
    """Adds""" 
    if isinstance(other, Bernoulli):
      # Ex 11: Check whether the parameter p of `other` is the same as the parameter of `self`.
      # If so, return a Bernoulli(2, p) object. If not, raise an appropriave ValueError.
      # ...
      pass

    # Ex 13: ...
    else:  
      pass

  def support(self):
    """Returns the support of the random variable"""
    return [0, 1]

  def mean(self):
    """Returns the mean $p$ of this Bernoulli random variable"""
    return self.p

  # Ex 5: ...


class Binomial(object):
  """A Binomial random variable"""

  def __init__(self, n, p):

    # Check the argument p
    if not is_probability(p):
      raise ValueError("The parameter p should be a number between 0 and 1")

    # Check if n is an integer
    if not is_natural_number(n):
      raise ValueError("The parameter n should be an integer")
    
    # Ex 8: Store parameters as attributes n and p
    # ...
    
    pass

  def __str__(self):
    """Returns a string of the form "Binomial(n, p)", where n and p are the parameter values"""
    # Ex 8: ...
    pass

  def support(self):
    """Returns the support of the random variable"""
    return list(range(0, n+1))

  def mean(self):
    """Returns the mean $np$ of this Binomial random variable"""
    # Ex 9: ...
    pass

  # Ex 10: ...


#############################################################################
############################# E X E R C I S E S #############################
#############################################################################


### Exercise 1
# Before we can create an instance of the Bernoulli class, we need to check
# if the parameter p is valid (this is done in the `__init__` methdod).
# Complete the function `is_probability`.
# --> See above

# Tests for exercise 1
# print("Test 1.1", is_probability(0.5) == True)
# print("Test 1.2", is_probability(0) == True)
# print("Test 1.3", is_probability(1) == True)
# print("Test 1.4", is_probability(10) == False)


### Exercise 2
# Create a Bernoulli random variable X with parameter p = .6 and 
# print the attribute p of X

# X = ...


### Exercise 3
# Update the parameter of X to 0.3. Also try 1.4. What happens?

#...


### Exercise 4
# We have already implemented a method `mean` that returns the mean
# of a Bernoulli random variable. Print the mean of X.

# ...


### Exercise 5
# Write a new method `variance` for the class Bernoulli that returns
# the variance p * (1 - p) of the random variable
# --> See above

# print("Test 5.1", X.variance() == X.p * (1 - X.p))


### Exercise 6 
# When we print a random variable, we want to get readable output. However,
# when you print X now, the output is not that readable. Try that. After that,
# complete the method __str__ of the Bernoulli class, and try what happens
# when you print X afterwards.
# --> See above

# print("Test 6.1", str(Bernoulli(0.5)) == "Bernoulli(0.5)") 


### Exercise 7 
# We now turn to the Binomial class. Recall that when initializing a 
# Bernoulli RV, we tested whether the parameter p was a valid probability. We
# now need to test if the other parameter, n, is natural number.
# Complete the function `is_natural_number`. Hint: the function `isinstance` can help.
# --> See above

# print("Test 7.1", is_natural_number(0.5) == False)
# print("Test 7.2", is_natural_number(-1) == False)
# print("Test 7.3", is_natural_number(0) == False)
# print("Test 7.4", is_natural_number(1) == True)
# print("Test 7.5", is_natural_number('asdf') == False)


### Exercise 8
# We also want readable string representations of Binomial RVs. Store the parameters
# passed to the `__init__` method as p and n, and then complete the `__str__` method of 
# Binomial. Finally, create a Binomial random variable called `Y` with parameters n=4 
# and p=0.2 and print it.
# --> See above

# print("Test 8.1", str(Binomial(0.5, 5)) == "Binomial(0.5, 5)") 


### Exercise 9
# Finish the method `mean` of the Binomial class, and test if it works correctly.
# --> See above


### Exercise 10
# Add a method `variance` to the Binomial class, that returns the variance 
# np(1 - p) of the random variable
# --> see above

### Exercise 11
# Finally, the fun part: we want to be able to add i.i.d. random variables.
# So if X and Y are both Bernoulli's with parameter p, then we want the
# statement `X + Y` to return a Binomial RV with parameters n=2 and p=p.
# Complete the first part of the `__add__` method of the Bernoulli class.
# --> See above


### Exercise 12
# If X ~ Binomial(n, p) and Y an independent Bernoulli(p), then Z = X + Y
# is a Binomial(n+1, p) variable. We want to implement this.
# Add a method `__add__`` to the Binomial class that takes arguments `self`
# and `other`. 
# Tests the type of `other`.
# - If `other` is a Bernoulli, test if the parameters match and return the 
#   right Binomial. If the parameters don't match, raise an appropriate ValueError.
# - If `other` is a Binomial, check the parameters and return the appropriate
#   Binomial. Again, raise a ValueError with mismatching parameters.
# - Otherwise, raise a ValueError.


# Uncomment the following and check if A, B and C are correct.
# p = 0.5
# X = Binomial(p)
# Y = Binomial(p)
# A = X + Y
# B = A + X
# C = A + B


### Exercise 13
# There is still one problem. We can compute A + X (Binomial + Bernoulli), but
# we cannot yet compute X + A (Bernoulli + Binomial), since `Bernoulli.__add__`
# doesn't yet know how to handle Binomials. Fix this by reusing your prior work.
# --> See above

# Check if X + A works afterwards.


### Extra exercises
# - Add a method `pmf` to both the Bernoulli and Binomial classes that takes 
#   a number `x` as input, and returns the probability mass at `x`. If `x` is not
#   in the support, you raise a ValueError.
# - Add a method `cdf` for both classes that computes the cumulative distribution
#   function for every input `x`. 
# - Add a method `sample` that returns a sample from the distribution. How can you 
#   do this for the Binomial?

