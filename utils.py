from math import inf, log2

"""
Function that count the elements of an array and return a dictionary
"""
def count_vals(array):
    count = {}
    for el in array:
        if el in count:
            count[el] += 1
        else: 
            count[el] = 1

    return count

"""
Function that returns the highest entry in a dictionary
"""
def dict_max(dict):
    max_val = -inf
    max_el = None
    for [k, v] in dict.items():
        if v > max_val:
            max_val = v
            max_el = k

    return [max_el, max_val]

"""
Compute the entropy of a set of data
"""
def entropy(set):
    total = len(set)
    counts = count_vals(set)
    entropy = 0
    for [k, v] in counts.items():
        p = v/total
        entropy += -p * log2(p)
    return entropy

"""
Compute the gini impurity of a set of data
"""
def gini_impurity(set):
    total = len(set)
    counts = count_vals(set)
    sum = 0
    for [k, v] in counts.items():
        p = v/total
        sum + p^2
    return 1 - sum