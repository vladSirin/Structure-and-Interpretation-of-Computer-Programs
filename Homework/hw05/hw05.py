def tree(label, branches=[]):
    """Construct a tree with the given label value and a list of branches."""
    for branch in branches:
        assert is_tree(branch), 'branches must be trees'
    return [label] + list(branches)


def label(tree):
    """Return the label value of a tree."""
    return tree[0]


def branches(tree):
    """Return the list of branches of the given tree."""
    return tree[1:]


def is_tree(tree):
    """Returns True if the given tree is a tree, and False otherwise."""
    if type(tree) != list or len(tree) < 1:
        return False
    for branch in branches(tree):
        if not is_tree(branch):
            return False
    return True


def is_leaf(tree):
    """Returns True if the given tree's list of branches is empty, and False
    otherwise.
    """
    return not branches(tree)


def print_tree(t, indent=0):
    """Print a representation of this tree in which each node is
    indented by two spaces times its depth from the root.

    >>> print_tree(tree(1))
    1
    >>> print_tree(tree(1, [tree(2)]))
    1
      2
    >>> numbers = tree(1, [tree(2), tree(3, [tree(4), tree(5)]), tree(6, [tree(7)])])
    >>> print_tree(numbers)
    1
      2
      3
        4
        5
      6
        7
    """
    print('  ' * indent + str(label(t)))
    for b in branches(t):
        print_tree(b, indent + 1)


def copy_tree(t):
    """Returns a copy of t. Only for testing purposes.

    >>> t = tree(5)
    >>> copy = copy_tree(t)
    >>> t = tree(6)
    >>> print_tree(copy)
    5
    """
    return tree(label(t), [copy_tree(b) for b in branches(t)])


#############
# Questions #
#############

def replace_leaf(t, old, new):
    """Returns a new tree where every leaf value equal to old has
    been replaced with new.

    >>> yggdrasil = tree('odin',
    ...                  [tree('balder',
    ...                        [tree('thor'),
    ...                         tree('loki')]),
    ...                   tree('frigg',
    ...                        [tree('thor')]),
    ...                   tree('thor',
    ...                        [tree('sif'),
    ...                         tree('thor')]),
    ...                   tree('thor')])
    >>> laerad = copy_tree(yggdrasil) # copy yggdrasil for testing purposes
    >>> print_tree(replace_leaf(yggdrasil, 'thor', 'freya'))
    odin
      balder
        freya
        loki
      frigg
        freya
      thor
        sif
        freya
      freya
    >>> laerad == yggdrasil # Make sure original tree is unmodified
    True
    """
    "*** YOUR CODE HERE ***"
    if is_tree(t):
        if is_leaf(t):
            if label(t) == old:
                return tree(new)
            return t

        else:
            new_list = [replace_leaf(t_branch, old, new) for t_branch in t]
            return new_list

    else:
        return t


def print_move(origin, destination):
    """Print instructions to move a disk."""
    print("Move the top disk from rod", origin, "to rod", destination)


def move_stack(n, start, end):
    """Print the moves required to move n disks on the start pole to the end
    pole without violating the rules of Towers of Hanoi.

    n -- number of disks
    start -- a pole position, either 1, 2, or 3
    end -- a pole position, either 1, 2, or 3

    There are exactly three poles, and start and end must be different. Assume
    that the start pole has at least n disks of increasing size, and the end
    pole is either empty or has a top disk larger than the top n start disks.

    >>> move_stack(1, 1, 3)
    Move the top disk from rod 1 to rod 3
    >>> move_stack(2, 1, 3)
    Move the top disk from rod 1 to rod 2
    Move the top disk from rod 1 to rod 3
    Move the top disk from rod 2 to rod 3
    >>> move_stack(3, 1, 3)
    Move the top disk from rod 1 to rod 3
    Move the top disk from rod 1 to rod 2
    Move the top disk from rod 3 to rod 2
    Move the top disk from rod 1 to rod 3
    Move the top disk from rod 2 to rod 1
    Move the top disk from rod 2 to rod 3
    Move the top disk from rod 1 to rod 3
    """
    assert 1 <= start <= 3 and 1 <= end <= 3 and start != end, "Bad start/end"
    "*** YOUR CODE HERE ***"
    # Base case
    if n == 1:
        print_move(start, end)
    elif n == 2:
        print_move(start, 6 - start - end)
        print_move(start, end)
        print_move(6 - start - end, end)

    # Recursive solve
    else:
        move_stack(n - 1, start, 6 - start - end)
        move_stack(1, start, end)
        move_stack(n - 1, 6 - start - end, end)


###########
# Mobiles #
###########

def mobile(left, right):
    """Construct a mobile from a left side and a right side."""
    return tree('mobile', [left, right])


def is_mobile(m):
    return is_tree(m) and label(m) == 'mobile'


def sides(m):
    """Select the sides of a mobile."""
    assert is_mobile(m), "must call sides on a mobile"
    return branches(m)


def is_side(m):
    return not is_mobile(m) and not is_weight(m) and type(label(m)) == int


def side(length, mobile_or_weight):
    """Construct a side: a length of rod with a mobile or weight at the end."""
    return tree(length, [mobile_or_weight])


def length(s):
    """Select the length of a side."""
    assert is_side(s), "must call length on a side"
    return label(s)


def end(s):
    """Select the mobile or weight hanging at the end of a side."""
    assert is_side(s), "must call end on a side"
    return branches(s)[0]


def weight(size):
    """Construct a weight of some size."""
    assert size > 0
    "*** YOUR CODE HERE ***"
    return tree('weight', [tree(x) for x in range(size)])


def size(w):
    """Select the size of a weight."""
    "*** YOUR CODE HERE ***"
    return len(branches(w))


def is_weight(w):
    """Whether w is a weight, not a mobile."""
    "*** YOUR CODE HERE ***"
    return w[0] == 'weight'


def examples():
    t = mobile(side(1, weight(2)),
               side(2, weight(1)))
    u = mobile(side(5, weight(1)),
               side(1, mobile(side(2, weight(3)),
                              side(3, weight(2)))))
    v = mobile(side(4, t), side(2, u))
    return (t, u, v)


def total_weight(m):
    """Return the total weight of m, a weight or mobile.

    >>> t, u, v = examples()
    >>> total_weight(t)
    3
    >>> total_weight(u)
    6
    >>> total_weight(v)
    9
    """
    if is_weight(m):
        return size(m)
    else:
        assert is_mobile(m), "must get total weight of a mobile or a weight"
        return sum([total_weight(end(s)) for s in sides(m)])


def balanced(m):
    """Return whether m is balanced.

    >>> t, u, v = examples()
    >>> balanced(t)
    True
    >>> balanced(v)
    True
    >>> w = mobile(side(3, t), side(2, u))
    >>> balanced(w)
    False
    >>> balanced(mobile(side(1, v), side(1, w)))
    False
    >>> balanced(mobile(side(1, w), side(1, v)))
    False
    """
    "*** YOUR CODE HERE ***"
    side_a_end = end(sides(m)[0])
    side_b_end = end(sides(m)[1])

    def balanced_weight():
        return total_weight(side_a_end) * length(sides(m)[0]) == total_weight(side_b_end) * length(sides(m)[1])

    if is_weight(side_a_end) and is_weight(side_b_end):
        return balanced_weight()

    elif is_weight(side_a_end) and is_mobile(side_b_end):
        return balanced_weight() and balanced(side_b_end)

    elif is_mobile(side_a_end) and is_weight(side_b_end):
        return balanced_weight() and balanced(side_a_end)

    elif is_mobile(side_a_end) and is_mobile(side_b_end):
        return balanced_weight() and balanced(side_a_end) and balanced(side_b_end)


#######
# OOP #
#######

class Account:
    """An account has a balance and a holder.

    >>> a = Account('John')
    >>> a.deposit(10)
    10
    >>> a.balance
    10
    >>> a.interest
    0.02

    >>> a.time_to_retire(10.25) # 10 -> 10.2 -> 10.404
    2
    >>> a.balance               # balance should not change
    10
    >>> a.time_to_retire(11)    # 10 -> 10.2 -> ... -> 11.040808032
    5
    >>> a.time_to_retire(100)
    117
    """

    interest = 0.02  # A class attribute

    def __init__(self, account_holder):
        self.holder = account_holder
        self.balance = 0

    def deposit(self, amount):
        """Add amount to balance."""
        self.balance = self.balance + amount
        return self.balance

    def withdraw(self, amount):
        """Subtract amount from balance if funds are available."""
        if amount > self.balance:
            return 'Insufficient funds'
        self.balance = self.balance - amount
        return self.balance

    def time_to_retire(self, amount):
        """Return the number of years until balance would grow to amount."""
        assert self.balance > 0 and amount > 0 and self.interest > 0
        "*** YOUR CODE HERE ***"
        years_to_retire, balance_to_be = 0, self.balance
        while balance_to_be < amount:
            balance_to_be += balance_to_be * self.interest
            years_to_retire += 1
        return years_to_retire


class FreeChecking(Account):
    """A bank account that charges for withdrawals, but the first two are free!

    >>> ch = FreeChecking('Jack')
    >>> ch.balance = 20
    >>> ch.withdraw(100)  # First one's free
    'Insufficient funds'
    >>> ch.withdraw(3)    # And the second
    17
    >>> ch.balance
    17
    >>> ch.withdraw(3)    # Ok, two free withdrawals is enough
    13
    >>> ch.withdraw(3)
    9
    >>> ch2 = FreeChecking('John')
    >>> ch2.balance = 10
    >>> ch2.withdraw(3) # No fee
    7
    >>> ch.withdraw(3)  # ch still charges a fee
    5
    >>> ch.withdraw(5)  # Not enough to cover fee + withdraw
    'Insufficient funds'
    """
    withdraw_fee = 1
    free_withdrawals = 2

    "*** YOUR CODE HERE ***"

    def withdraw(self, amount):
        if self.free_withdrawals != 0:
            self.free_withdrawals -= 1
            return Account.withdraw(self, amount)
        else:
            return Account.withdraw(self, amount + self.withdraw_fee)


############
# Mutation #
############

def make_counter():
    """Return a counter function.

    >>> c = make_counter()
    >>> c('a')
    1
    >>> c('a')
    2
    >>> c('b')
    1
    >>> c('a')
    3
    >>> c2 = make_counter()
    >>> c2('b')
    1
    >>> c2('b')
    2
    >>> c('b') + c2('b')
    5
    """
    "*** YOUR CODE HERE ***"
    dict_strings = {}

    def counter(s):
        if s not in dict_strings.keys():
            dict_strings[s] = 0
        dict_strings[s] += 1
        return dict_strings[s]

    return counter


def make_fib():
    """Returns a function that returns the next Fibonacci number
    every time it is called.

    >>> fib = make_fib()
    >>> fib()
    0
    >>> fib()
    1
    >>> fib()
    1
    >>> fib()
    2
    >>> fib()
    3
    >>> fib2 = make_fib()
    >>> fib() + sum([fib2() for _ in range(5)])
    12
    """
    "*** YOUR CODE HERE ***"
    fib_0, fib_1 = 0, 1
    is_first = True

    def fib():
        nonlocal is_first
        if is_first:
            is_first = False
            return 0
        nonlocal fib_0, fib_1
        temp = fib_1
        fib_1 = fib_0 + fib_1
        fib_0 = temp
        return fib_0

    return fib


def make_withdraw(balance, password):
    """Return a password-protected withdraw function.

    >>> w = make_withdraw(100, 'hax0r')
    >>> w(25, 'hax0r')
    75
    >>> error = w(90, 'hax0r')
    >>> error
    'Insufficient funds'
    >>> error = w(25, 'hwat')
    >>> error
    'Incorrect password'
    >>> new_bal = w(25, 'hax0r')
    >>> new_bal
    50
    >>> w(75, 'a')
    'Incorrect password'
    >>> w(10, 'hax0r')
    40
    >>> w(20, 'n00b')
    'Incorrect password'
    >>> w(10, 'hax0r')
    "Your account is locked. Attempts: ['hwat', 'a', 'n00b']"
    >>> w(10, 'l33t')
    "Your account is locked. Attempts: ['hwat', 'a', 'n00b']"
    >>> type(w(10, 'l33t')) == str
    True
    """
    "*** YOUR CODE HERE ***"
    p1, p2, p3 = '', '', ''
    wrong_count = 0

    def withdraw(amount, code):
        nonlocal p1, p2, p3, wrong_count, balance, password
        if wrong_count >= 3:
            return 'Your account is locked. Attempts: ' + str([p1, p2, p3])
        if code != password:
            wrong_count += 1
            if wrong_count == 1:
                p1 = code
            elif wrong_count == 2:
                p2 = code
            elif wrong_count == 3:
                p3 = code
            return 'Incorrect password'
        else:
            if amount > balance:
                return 'Insufficient funds'
            balance = balance - amount
            return balance

    return withdraw


def make_joint(withdraw, old_password, new_password):
    """Return a password-protected withdraw function that has joint access to
    the balance of withdraw.

    >>> w = make_withdraw(100, 'hax0r')
    >>> w(25, 'hax0r')
    75
    >>> make_joint(w, 'my', 'secret')
    'Incorrect password'
    >>> j = make_joint(w, 'hax0r', 'secret')
    >>> w(25, 'secret')
    'Incorrect password'
    >>> j(25, 'secret')
    50
    >>> j(25, 'hax0r')
    25
    >>> j(100, 'secret')
    'Insufficient funds'

    >>> j2 = make_joint(j, 'secret', 'code')
    >>> j2(5, 'code')
    20
    >>> j2(5, 'secret')
    15
    >>> j2(5, 'hax0r')
    10

    >>> j2(25, 'password')
    'Incorrect password'
    >>> j2(5, 'secret')
    "Your account is locked. Attempts: ['my', 'secret', 'password']"
    >>> j(5, 'secret')
    "Your account is locked. Attempts: ['my', 'secret', 'password']"
    >>> w(5, 'hax0r')
    "Your account is locked. Attempts: ['my', 'secret', 'password']"
    >>> make_joint(w, 'hax0r', 'hello')
    "Your account is locked. Attempts: ['my', 'secret', 'password']"
    """
    "*** YOUR CODE HERE ***"
    x = withdraw(0, old_password)
    if type(x) == str:
        return x

    else:
        def withdraw_r(amount, code):
            if code == new_password:
                # print('password is new')
                return withdraw(amount, old_password)
            elif code != new_password:
                return withdraw(amount, code)

        return withdraw_r


###################
# Extra Questions #
###################

def interval(a, b):
    """Construct an interval from a to b."""
    return [a, b]


def lower_bound(x):
    """Return the lower bound of interval x."""
    "*** YOUR CODE HERE ***"
    return x[0]


def upper_bound(x):
    """Return the upper bound of interval x."""
    "*** YOUR CODE HERE ***"
    return x[1]


def str_interval(x):
    """Return a string representation of interval x."""
    return '{0} to {1}'.format(lower_bound(x), upper_bound(x))


def add_interval(x, y):
    """Return an interval that contains the sum of any value in interval x and
    any value in interval y."""
    lower = lower_bound(x) + lower_bound(y)
    upper = upper_bound(x) + upper_bound(y)
    return interval(lower, upper)


def mul_interval(x, y):
    """Return the interval that contains the product of any value in x and any
    value in y."""
    p1 = lower_bound(x) * lower_bound(y)
    p2 = lower_bound(x) * upper_bound(y)
    p3 = upper_bound(x) * lower_bound(y)
    p4 = upper_bound(x) * upper_bound(y)
    return interval(min(p1, p2, p3, p4), max(p1, p2, p3, p4))


def sub_interval(x, y):
    """Return the interval that contains the difference between any value in x
    and any value in y."""
    "*** YOUR CODE HERE ***"
    p1 = lower_bound(x) - lower_bound(y)
    p2 = lower_bound(x) - upper_bound(y)
    p3 = upper_bound(x) - lower_bound(y)
    p4 = upper_bound(x) - upper_bound(y)
    return interval(min(p1, p2, p3, p4), max(p1, p2, p3, p4))


def div_interval(x, y):
    """Return the interval that contains the quotient of any value in x divided by
    any value in y. Division is implemented as the multiplication of x by the
    reciprocal of y."""
    "*** YOUR CODE HERE ***"
    assert not (lower_bound(y) <= 0 <= upper_bound(y)), "Interval Y including 0, cannot be divided."
    reciprocal_y = interval(1 / upper_bound(y), 1 / lower_bound(y))
    return mul_interval(x, reciprocal_y)


def par1(r1, r2):
    return div_interval(mul_interval(r1, r2), add_interval(r1, r2))


def par2(r1, r2):
    one = interval(1, 1)
    rep_r1 = div_interval(one, r1)
    rep_r2 = div_interval(one, r2)
    return div_interval(one, add_interval(rep_r1, rep_r2))


def check_par():
    """Return two intervals that give different results for parallel resistors.

    >>> r1, r2 = check_par()
    >>> x = par1(r1, r2)
    >>> y = par2(r1, r2)
    >>> lower_bound(x) != lower_bound(y) or upper_bound(x) != upper_bound(y)
    True
    """
    r1 = interval(1, 1)  # Replace this line!
    r2 = interval(0.25, 9)  # Replace this line!
    return r1, r2


def multiple_references_explanation():
    return """The multiple reference problem...
            The problem exists in the case of Par1,
            The TRUE value within the interval, despite unknown to us, is a fixed value.
            Using nested combinations that refer to the same interval twice might 
            assume more than one TRUE values for the same interval. This assume could lead
            to wrong intervals than they should have been.
            
            For example, as the value of i within the interval of [-1, 1], no value with i*i
            will have a result of negative result, but by our mul_interval function, we will have
            the interval of [-1, 1]. This is because of the wrong assumption of the ONE TRUE value
            is allowed to be both -1 and 1 in different references of the same interval.
            
            Hence the program of par2 is better than par 1 because it never combines the same
            interval more than once.
            """


def quadratic(x, a, b, c):
    """Return the interval that is the range of the quadratic defined by
    coefficients a, b, and c, for domain interval x.

    >>> str_interval(quadratic(interval(0, 2), -2, 3, -1))
    '-3 to 0.125'
    >>> str_interval(quadratic(interval(1, 3), 2, -3, 1))
    '0 to 10'
    """
    "*** YOUR CODE HERE ***"
    # A quadratic function graph is a parabola
    # So the extreme point is either the lowest or the highest
    # This is based on the value of 'a', which is the first coefficient

    point_0 = a * lower_bound(x) * lower_bound(x) + b * lower_bound(x) + c
    point_1 = a * upper_bound(x) * upper_bound(x) + b * upper_bound(x) + c
    if a == 0:
        return interval(min(point_1, point_0), max(point_1, point_0))
    ext_x = -b / (2 * a)
    ext_point = a * ext_x * ext_x + b * ext_x + c

    # check if the first coefficient is positive
    if a < 0:  # if so check if the extreme point is inside the interval of 'x'
        if lower_bound(x) <= ext_point <= upper_bound(x):  # extreme point is the upper bound of the return interval
            return interval(min(point_0, point_1), ext_point)
        else:
            return interval(min(point_0, point_1), max(point_1, point_0))
    else:  # means the first coefficient is negative
        if lower_bound(x) <= ext_point <= upper_bound(x):  # extreme point is the lower bound of the return interval
            return interval(ext_point, max(point_0, point_1))
        else:
            return interval(min(point_1, point_0), max(point_0, point_1))


def polynomial(x, c):
    """Return the interval that is the range of the polynomial defined by
    coefficients c, for domain interval x.

    >>> str_interval(polynomial(interval(0, 2), [-1, 3, -2]))
    '-3 to 0.125'
    >>> str_interval(polynomial(interval(1, 3), [1, -3, 2]))
    '0 to 10'
    >>> str_interval(polynomial(interval(0.5, 2.25), [10, 24, -6, -8, 3]))
    '18.0 to 23.0'
    """
    "*** YOUR CODE HERE ***"
    # The hint is using the newton update method 'update(x): x - f(x)/df(x)'
    # First we need to understand that a polynomial graph is a curve with several turning points
    # Then we need to find those turning points and see whether the interval 'x' includes any turning points
    x_zeros = find_zeros(d_poly_f, dd_poly_f, c)
    p_lower = poly_f(lower_bound(x), c)
    p_upper = poly_f(upper_bound(x), c)
    num, zeros_in = 0, []
    for i in x_zeros:
        if lower_bound(x) <= i <= upper_bound(x):
            num += 1
            zeros_in.append(i)
    # If it doesn't, then p(x) is proportional to 'x', thus comparing lower and upper bounds would solve it
    if num == 0:
        return interval(min(p_lower, p_upper), max(p_lower, p_upper))
    # If it includes only one, then solve it use the quadratic way
    elif num == 1:
        p_t = poly_f(zeros_in[0], c)
        return interval(min(p_lower, p_upper, p_t), max(p_lower, p_upper, p_t))
    # If it includes more than one, then max and min must in those points, just comparing the p(x) of them.
    elif num > 1:
        min_pt = float("inf")
        max_pt = -float("inf")
        for zero in zeros_in:
            if poly_f(zero, c) > max_pt:
                max_pt = poly_f(zero, c)
            if poly_f(zero, c) < min_pt:
                min_pt = poly_f(zero, c)
        return interval(min_pt, max_pt)


# HELP FUNCTIONS FOR LAST QUESTION: POLYNOMIAL

def find_zeros(f, df, c, guess=1.0):
    """Keep using update(guess) function until f(x) = 0 is true"""
    max_zeros = len(c) - 2
    zeros = []
    while len(zeros) < max_zeros:
        if abs(f(guess, c)) < 1e-10:
            if guess not in zeros:
                zeros.append(guess)
        if newton_update(f, df)(guess, c) == guess:
            guess = df(guess, c)
        else:
            guess = newton_update(f, df)(guess, c)
    return zeros


def newton_update(f, df):
    """Newton method to update the value by the differentiation"""

    def update(x, c):
        return x - f(x, c) / df(x, c)

    return update


def poly_f(x, c):
    """Calculate the value of the f(x) based on the polynomial method provided by the question:
    f(t) = c[k-1] * pow(t, k-1) + c[k-2] * pow(t, k-2) + ... + c[0] * 1
    """
    f_x = 0
    for i in range(len(c)):
        f_x += pow(x, i) * c[i]
    return f_x


def d_poly_f(x, c):
    """Derivative function of poly_f"""
    df_x = 0
    for i in range(1, len(c)):
        df_x += pow(x, i - 1) * c[i] * i
    return df_x


def dd_poly_f(x, c):
    """Derivative of d_poly_f"""
    ddf_x = 0
    for i in range(2, len(c)):
        ddf_x += pow(x, i - 2) * c[i] * i * (i - 1)
    return ddf_x
