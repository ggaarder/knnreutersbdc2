import itertools
import random
import algo

def test_top_k(k, s):
    return set(algo.top_k(k, s[:])) == set(algo.naive_top_k(k, s[:]))

if __name__ == '__main__':
    for i in itertools.count(1):
        print('Testing top-k #{}'.format(i))

        sample = random.sample(range(10000000), random.randrange(10, 1000))
        k = random.randrange(1, len(sample))

        if not test_top_k(k, sample):
            print('Test failed in {} with k={}'.format(sample, k))
            break
        else:
            print('pass')