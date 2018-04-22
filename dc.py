import math

def dc(x, abs_C):
    fracs = []

    for i in x:
        fracs.append(i/sum(x))

    sigma = sum([i*math.log2(i) for i in fracs])
    dc_result = 1 + 1/math.log2(abs_C)*sigma

    return dc_result

def bdc(x, categories):
    abs_C = len(categories)
    fracs = []

    for i in x:
        fracs.append(i/sum(x))

    sigma = sum([i*math.log2(i) for i in fracs])
    bdc_result = 1 + 1/math.log2(abs_C)*sigma

    return dc_result