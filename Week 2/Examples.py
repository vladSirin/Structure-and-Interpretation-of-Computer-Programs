"""Examples for the book"""


def improve(update, close, guess=1):
    """When a higher order function as this is called, it first creates the frame with the function and assign
    the name of arguments with the actual functions are called.
        Then in this case, resolve each of the assigned functions first before doing the computations inside
    the higher order function."""
    while not close(guess):
        guess = update(guess)
    return guess


def approx_eq(x, y, tolerance=1e-15):
    return abs(x - y) < tolerance


def newton_update(f, df):
    """When this function is used as an argument, it is executed so that
    actually this is a function takes 1 argument as update(x) instead of 2 arguments as it was"""
    def update(x):
        return x - f(x) / df(x)
    return update


def find_zero(f, df):
    def near_zero(x):
        return approx_eq(f(x), 0)
    return improve(newton_update(f, df), near_zero)


"""We first implement square_root by defining f and its derivative df. 
We use from calculus the fact that the derivative of f(x)=x^2 - a 
is the linear function df(x) = 2x."""


def square_root_newton(a):
    def f(x):
        return x * x - a

    def df(x):
        return x * 2
    return find_zero(f, df)


print(square_root_newton(64))
