import numpy as np
import matplotlib.pyplot as plt


# Matplotlib is the most used plotting library for Python. Let's dive straight in:
# we want to plot the function f(x) =  x^3 + 1. To do so, we give matplotlib the
# x-coordinates and the y-coordinates it has to plot.

def plot_cube():
    xs = np.array([-3, -2, -1, 0, 1, 2, 3]) # x-coords
    ys = xs ** 3 + 1                        # y-coords, this is easy using numpy
    plt.plot(xs, ys)                        # Plot!
    plt.show()                              # Show the plot in a new window

# This is not a nice smooth plot. To fix that we need to add more x-coordinates.
# Numpy has a convenient function for this: `np.linspace`. Figure out what it does
# and change `plot_cube` so that `xs` is an array with 100 points between -3 and 3.

# Next, we plot two functions. Try to understand the following function, and change it
# so that quadratic polynomial get a thin dashed, red line with square markers with
# appropriate label.

def plot_square_cube():
    """
    Plots two functions: a cubic polynomial f(x) = x^3 + 1 and a quadratic polynomial g(x) = x^2 - 2
    """

    xs = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    cubic_values = xs ** 3 + 1
    quadratic_values = xs ** 2 - 2

    # You can pretty much everything in Matplotlib. This demonstrates the most important
    # properties. See https://matplotlib.org/users/pyplot_tutorial.html for more.

    plt.plot(xs, cubic_values,
             color="r",      # "r"=red, "b"=blue, "k"=black, "y"=yellow, ... (even shorter: c='r')
             linestyle="-",  # "--"=dashed, "-."=dash-dot, ":"=dotted, ... (even shorter: ls='-')
             marker="o",     # "o"=dot, "s"=square, "+"=cross, ...
             linewidth=3,    # (even shorter: lw=4)
             markersize=8,   # (even shorter: ms=10)
             label="Cubic: $f(x) = x^3 + 1$")

    #---------
    # PRO TIP: there's a super short syntax for basic color, line style and marker options:
    # plt.plot(xs, cubic_values, 'or--') gives a red dashed line with dots as markers
    # Matplotlib will try hard to make sense of the string in the third argument
    #---------

    plt.plot(xs, quadratic_values)

    # Figure out what the following lines do:
    # plt.title('Two functions')
    # plt.xlabel('Input $x$')
    # plt.ylabel('Ouput $y$')
    # plt.axis('off')
    # plt.axis([-2, 1, -10, 10])
    # plt.legend()

    plt.show()

if __name__ == '__main__':

    plot_cube()

    # plot_square_cube()
