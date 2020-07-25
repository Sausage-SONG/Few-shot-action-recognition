import scipy.stats
import numpy as np

def print_stage(name, length=65, character="-"):
    Llength = int((length - len(name)) / 2)
    Rlength = length - Llength - len(name)
    print("\n" + character*Llength + name + character*Rlength)

def print_debug(content, character="*"):
    length = len(content) + 2
    print(character*length)
    print("{} DEBUG {}".format(character, character), end='')
    print(character*(length - 9))
    print(character*length)
    print("{}{}{}".format(character, content, character))
    print(character*length)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

def compute_score(prediction, truth, length):
    if truth in prediction:
        zeros = np.count_nonzero(prediction == 0)
        return zeros/(length-1)
    else:
        return 0
    