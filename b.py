import sys
import numpy as np
import argparse
from tqdm import tqdm

from a import go

import pickle
import uuid



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--n', type=int, default=1000000)
    parser.add_argument('--out_type', type=str, choices=['pickle', 'plot'], default='pickle')
    args = parser.parse_args()

    print(sys.argv)
    print(args)

    R = np.linspace(0., .5, int((.5 - 0.) / .025))
    Beta = np.linspace(.5, 1., int((1. - .5) / .025))
    size = [len(R), len(Beta)]


    for _ in tqdm(range(args.iters)):
        grid = np.zeros(size)
        for i in tqdm(range(len(Beta))):
            for j in tqdm(range(len(R))):
                mu = np.sqrt(2. * R[j] * np.log(float(args.n)))
                ep = np.power(float(args.n), -Beta[i])

                N0 = int(np.around((1. - ep) * float(args.n)))

                assert N0 >= 0
                assert N0 <= args.n

                grid[len(R) - 1 - j, i] += go(args.n, N0, mu, 1., 100)

        params = ',n=' + str(args.n) + ',iters=' + str(args.iters)
        filename = str(uuid.uuid4()) + '_microarray_prediction' + params + '.p'
        pickle.dump([args.iters, grid], open(filename, 'wb' ), protocol=2)


if __name__ == '__main__':
    main()

