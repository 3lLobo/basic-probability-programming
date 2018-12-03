import numpy as np
import matplotlib.pyplot as plt

# Fix random seed, so we get the same randomness every time we run the program
np.random.seed(0)

def plot_unit_circle():
    """Plots the unit circle C(0, 1)"""
    degs = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(degs), np.sin(degs), 'k--', lw=.5)
    plt.gca().set_aspect('equal', 'datalim')

####################################################################

# Ex 1: Define your plot_points function here...

# Ex 2: Define your sample_from_square function here...

# Ex 3: Define your function test_if_in_circle here

# Ex 4: Define your function show_monte_carlo_pi here

####################################################################

def exercise1():
    """
    Exercise 1

    Write a function plot_points with two arguments:
    -   coordinates: a matrix with N rows and 2 columns. Rows correspond to
        points, the two columns to the x- and y-coords of the points respectively.
    -   color: specifies the color of the dots. This argument is optional and
        should default to 'k'
    -   markersize: the marker size. The argument is optional and defaults to 1.
    The function should plot the points in the coordinates matrix, using dots as
    markers and no connecting lines. The color and size of the dots should correspond
    to the arguments `color` and `markersize`.
    """

    # This code helps you debug your function
    my_coords = np.array([
        [0, 0],
        [1, 2],
        [-2, 1]
    ])
    plot_points(my_coords , color='red', markersize=4)
    plt.show()


def exercise2():
    """
    Exercise 2

    Write a function sample_from_square that takes one argument:
    -   num_samples: the number of samples to draw from the square.
    The function should return a (num_sample x 2)-coordinate matrix with points that
    are sampled uniformly from the square [-1, 1] x [-1, 1].

    You can use the numpy function `np.random.rand(n, m)` which returns a matrix of
    size n x m whose entries are random numbers between 0 and 1.
    """

    # This code visualizes the result and can be helpful for debugging:
    samples = sample_from_square(100)
    plot_points(samples, color="red", markersize=4)
    plt.axis([-1.5, 1.5, -1.5, 1.5])
    plt.show()


def exercise3():
    """
    Exercise 3

    Define a function test_if_in_circle that takes one argument:
    -   coordinates: a (num_samples x 2)-matrix with coordinates.
    The function should return an array of length num_samples with values True
    and False indicating which of the points lie inside the unit circle. So
    if point i (i.e. row i) is inside the circle, the i-the entry of the output
    array should be True.

    You can use that a point (x, y) is inside the unit circle if x^2 + y^2 <= 1.
    Implement this without a for loop, but manipulate the coordinate array. Final hint:
    what does np.array([1,2,3]) > np.array([-1,0,4]) return?
    """

    # Some code that helps you test your function
    my_coords = np.array([
        [0, 0],
        [0, 1],
        [2, 0]
    ])

    is_inside = test_if_in_circle(my_coords)
    print("Test Ex 3:", np.all(is_inside == [True, True, False]))

    # Plot points
    plot_points(my_coords, color='red', markersize=4)
    plot_unit_circle()
    plt.show()


def exercise4():
    """
    Exercise 4

    Now we put everything together. Define a function show_monte_carlo_pi that
    estimates pi = 3.1415... using a Monte Carlo technique (see below). It takes
    two arguments:
    -   num_samples: the number of samples
    -   show_circle (optional, default to True)

    The idea is this. We sample num_samples points from the square [-1, 1] x [-1, 1],
    and test whether each of these points is inside the unit circle. We visualze this
    by plotting the 'hits' (points inside the circle) in green, and 'misses' (outside it)
    in red.

    To estimate pi, you only need the proportion of hits (i.e. len(hits) / num_samples).
    After all, the area of a circle with radius R is A(r) = pi * r^2. For the unit circle,
    A(1) = pi. But we also know that the square [-1, 1] x [-1, 1] has area 4, and the
    proportion of that that lies inside the square is len(hits) / num(samples). So,
    we can approximate pi as pi = len(hits) / num_samples * 4.

    Your function should do this and print the result (rounded to a 5 decimals) as the
    title of the plot. You can turn hide the axes.
    """

    show_monte_carlo_pi(10)


if __name__ == '__main__':
    pass
    # exercise1()
    # exercise2()
    # exercise3()
    # exercise4()
