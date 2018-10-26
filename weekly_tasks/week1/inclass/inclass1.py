"""
This is the set of first exercises used in class.

The comments (marked by #) explain what you should do. Type your code below each comment.
"""

#1. Assign values 5 and 3 to variables named first and second;\
#print the values of the variables, one per line\
#(to help you get going, this question is already answered)

first = 5
second = 3
print("first:", first)
print("second:", second)

#2. Add the values of first and second and store the result\
#as the variable total_sum. Print the value of total_sum

#3. Concatenate the values of first and second and store the result as the variable first_and_second (that is, 53 should be stored). Print first_and_second. Subtract 3 from first_and_second and assign this to the variable final_count.

#4. Create a new variable, third, which has the value of 50. Print the variable.

#5. Create a conditional statement that prints 'The values match'\
#in case third and final_count are equal, and it prints 'The values do not match' otherwise.\
#We want the two variables to be equal, that is, 'The values match' should be printed.\
#You might need to do type conversion at this step.

#6. Create a conditional statement that checks that first_and_second\
#is greater than total_sum. If so, it prints 'Concatenation is greater\
#than addition.' Otherwise, it prints 'Addition is greater than concatenation.'\
#Of course, if everything is correct we would expect that the first message is printed.

# We will now use one module, random. We will do so to start learning\
#how solutions we discussed in the first part of the course can be achieved\
#brute force (by random sampling). The documentation about Python and its modules\
#can be found here: https://docs.python.org/3.5/. The documentation about\
#this particular module is here: https://docs.python.org/3.5/library/random.html.\
#First, we import the module as follows:

import random

#7. Now, we want to randomly pick two letters from the string 'abcdefg'.\
# We can do it as shown below. Check the documentation on https://docs.python.org/3.5/library/random.html\
#to understand what random.sample does.

start_alphabet = 'abcdefg'

random_pick = random.sample(start_alphabet, 2)

print("".join(random_pick)) #ignore the part about join for now

#8. Let us encapsulate the behavior in the last piece of code.\
#Write a function called two_letter_sample that will take any string\
#as its argument and return a string with two randomly sampled letters\
#if the original string is at least of length two. Check that the function\
#gives a sensible output using a few strings as an example.

#9. Create a second function, called check_if_vowels. The function will\
#take a (two-letter) string and will check that the string is either 'ae' or 'ea'.\
#If this is so, the function returns True, otherwise it returns False.\
#Test that the function works.

#10. Create a third function, recursive_call. This function takes\
#three arguments: input_string, repetitions, successes. The function\
#repeatedly samples two letters on the input string\
#(and it does so as many times as the number of repetitions).\
#Whenever the sample consists of two vowels ('ae' or 'ea')\
#it increments successes. Finally, it returns the number of successes.

#Note: this is a recursive function and some care has to be taken here.\
#It usually helps to add print statemements to the function.\
#For your convenience, parts of the function are already in place.

def recursive_call(input_string, repetitions, successes):
    """
    Repeatedly sample two letters from input_string, count the number of times that the sample is the string 'ae' or 'ea' (# of successes)

    :param input_string: the string used for sampling
    :param repetitions: how many times the process should be repeated
    :param successes: the current number of successes
    """
    if repetitions > 0:
        pass
        #write your code instead of pass; in your code, you should sample from input_string;\
        #then if the sample is 'ae' or 'ea', the count in successes should\
        #be increased by 1; after that, recursive_call should be called again
    return successes

#11. Run the recursive_call function with 900 repetitions\
#three times and store these results as output1, output2, output3.\
#Then print the frequency of successes. 

#12. The number will vary, but it should be close (up to 1 percent)\
#to the probability arrived at analytically. Write the formula, compare the two values
