import math
import operator
import sys

class Vector:
    def __init__(v):
        self.v == v

    def inner_product(self, v):
        return sum(map(operator.mul, self.v, v))

    def abs(self):
        return math.sqrt(sum(map(lambda x:x**2, self.v)))

    def cos(self, v):
        E = sys.float_info.epsilon
        return self.inner_product(v)/(E+self.abs())/(E+v.abs())