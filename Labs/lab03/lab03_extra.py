""" Optional problems for Lab 3 """

from math import factorial
from math import sqrt

from lab03 import *


## Higher order functions

def cycle(f1, f2, f3):
    """Returns a function that is itself a higher-order function.

    >>> def add1(x):
    ...     return x + 1
    >>> def times2(x):
    ...     return x * 2
    >>> def add3(x):
    ...     return x + 3
    >>> my_cycle = cycle(add1, times2, add3)
    >>> identity = my_cycle(0)
    >>> identity(5)
    5
    >>> add_one_then_double = my_cycle(2)
    >>> add_one_then_double(1)
    4
    >>> do_all_functions = my_cycle(3)
    >>> do_all_functions(2)
    9
    >>> do_more_than_a_cycle = my_cycle(4)
    >>> do_more_than_a_cycle(2)
    10
    >>> do_two_cycles = my_cycle(6)
    >>> do_two_cycles(1)
    19
    """
    "*** YOUR CODE HERE ***"

    def cycle_one(n):
        def cycle_two(x):
            i, f = 1, x
            while i <= n:
                if i % 3 == 1:
                    f = f1(f)
                elif i % 3 == 2:
                    f = f2(f)
                elif i % 3 == 0:
                    f = f3(f)
                i += 1
            return f

        return cycle_two

    return cycle_one


## Lambda expressions

def is_palindrome(n):
    """
    Fill in the blanks '_____' to check if a number
    is a palindrome.

    >>> is_palindrome(12321)
    True
    >>> is_palindrome(42)
    False
    >>> is_palindrome(2015)
    False
    >>> is_palindrome(55)
    True
    """
    x, y = n, 0
    f = lambda: y * 10 + (x % 10)
    while x > 0:
        x, y = x // 10, f()
    return y == n


## More recursion practice

def skip_mul(n):
    """Return the product of n * (n - 2) * (n - 4) * ...

    >>> skip_mul(5) # 5 * 3 * 1
    15
    >>> skip_mul(8) # 8 * 6 * 4 * 2
    384
    """
    if n == 1:
        return 1
    if n == 2:
        return 2
    else:
        return n * skip_mul(n - 2)


def is_prime(n):
    """Returns True if n is a prime number and False otherwise.

    >>> is_prime(2)
    True
    >>> is_prime(16)
    False
    >>> is_prime(521)
    True
    """
    "*** YOUR CODE HERE ***"

    def is_prime_two(x, y):
        if y < 2:
            return True
        if x % y == 0:
            return False
        else:
            return is_prime_two(x, y - 1)

    return is_prime_two(n, n // sqrt(n))


def interleaved_sum(n, odd_term, even_term):
    """Compute the sum odd_term(1) + even_term(2) + odd_term(3) + ..., up
    to n.

    >>> # 1 + 2^2 + 3 + 4^2 + 5
    ... interleaved_sum(5, lambda x: x, lambda x: x*x)
    29
    """
    "*** YOUR CODE HERE ***"
    if n == 0:
        return 0

    def product(x, y, z):
        if x % 2 == 1:
            return y(x)
        elif x % 2 == 0:
            return z(x)

    return product(n, odd_term, even_term) + interleaved_sum(n - 1, odd_term, even_term)


def ten_pairs(n):
    """Return the number of ten-pairs within positive integer n.

    >>> ten_pairs(7823952)
    3
    >>> ten_pairs(55055)
    6
    >>> ten_pairs(9641469)
    6
    """
    "*** YOUR CODE HERE ***"

    def num_times(num, n):
        times, i = 0, n
        while i > 0:
            if i % 10 == num:
                times += 1
            i = i // 10
        return times

    j, sum_r = 1, 0
    while j < 5:
        sum_r += num_times(j, n) * num_times(10 - j, n)
        j += 1

    if num_times(5, n) > 2:
        sum_5 = factorial(num_times(5, n)) // (2 * factorial(num_times(5, n) - 2))
    elif num_times(5, n) == 2:
        sum_5 = 1
    else:
        sum_5 = 0

    return sum_r+sum_5
