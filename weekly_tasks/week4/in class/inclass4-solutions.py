"""
This is the third set of exercises used in class.

The comments (marked by #) explain what you should do. Type your code below each comment.
"""

def is_probability(p):
  return 0 <= p and p <= 1


class Bernoulli(object):

  def __init__(self, p):
    if not is_probability(p):
      raise ValueError("The parameter p should be a probability between 0 and 1")
    self.p = p

  def mean(self):
    return self.p

  def variance(self):
    return self.p * (1 - self.p)

  def support(self):
    return [0, 1]

  def __str__(self):
    return "Bernoulli(" + str(self.p) + ")"

  def __add__(self, other):
    if isinstance(other, Bernoulli):
      if self.p == other.p:
        return Binomial(2, self.p)

      else:
        raise ValueError("Can only add random variables with the same parameter")

    elif isinstance(other, Binomial):
      return other + self


class Binomial(object):

  def __init__(self, n, p):
    if not is_probability(p):
      raise ValueError("The parameter p should be a probability between 0 and 1")
    self.n = n
    self.p = p

  def mean(self):
    return self.n * self.p

  def variance(self):
    return self.n * self.p * (1 - self.p)

  def support(self):
    return list(range(self.n+1))

  def __str__(self):
    return "Binomial(" + str(self.n) + ", " + str(self.p) + ")"

  def __add__(self, other):
    if not self.p == other.p:
      raise ValueError("Can only add random variables with the same parameter p")
    
    elif isinstance(other, Binomial):
      return Binomial(self.n + other.n, self.p)
        
    elif isinstance(other, Bernoulli):
      return Binomial(self.n + 1, self.p)
    
    else:
      raise Error("Don't know what to do...")



p = .8
X = Bernoulli(p)
Y = Bernoulli(p)
Z = X + Y
print(X, Y, Z)
print(X + Z)

# Z = X + Y
# W = Z + Bernoulli(p)
# print(X, Y, Z, W, W+Z, X+X+X )