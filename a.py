import sys
import numpy as np
import argparse

def dv(z, delta, pi0):
    pi1 = 1. - pi0

    f0 = np.exp(-np.power(z, 2.) / 2.) / (2. * np.pi)
    f1 = np.exp(-np.power(z - delta, 2.) / 2.) / (2. * np.pi)

    f0_p = -z * f0
    f1_p = (-z + delta) * f1

    f0_pp = (np.power(z, 2.) - 1.) * f0
    f1_pp = (np.power(z - delta, 2) - 1.) * f1
   
    a = pi0 * f0 + pi1 * f1
    b = pi0 * f0_p + pi1 * f1_p
    c = pi0 * f0_pp + pi1 * f1_pp

    d = z + b / a

    v = (a * c - np.power(b, 2.)) / np.power(a, 2.)

    return d, v

def go(N, N0, delta, c, n):
    assert N >= N0
    assert n % 2 == 0
    n2 = int(n / 2.)
    pi0 = float(N0) / float(N)

    #Sick
    null = np.random.normal(loc=0.0, scale=1.0, size=[n2, N0])
    nonnull = np.random.normal(loc=delta, scale=1.0, size=[n2, N - N0])
    sick = np.concatenate([null, nonnull], axis=-1)

    #Heathy
    null = np.random.normal(loc=0.0, scale=1.0, size=[n2, N0])
    nonnull = np.random.normal(loc=-delta, scale=1.0, size=[n2, N - N0])
    healthy = np.concatenate([null, nonnull], axis=-1)

    sick_bar = sick.mean(axis=0)
    healthy_bar = healthy.mean(axis=0)

    z = c * (sick_bar - healthy_bar)

    d, v = dv(z, delta, pi0)

    A = d / (2. * c)
    B = v / (4. * np.power(c, 2.)) + 1.

    w = A / B

    #Test phase
    nt = 50
    #Sick test set
    null = np.random.normal(loc=0.0, scale=1.0, size=[nt, N0])
    nonnull = np.random.normal(loc=delta, scale=1.0, size=[nt, N - N0])
    sick_test = np.concatenate([null, nonnull], axis=-1)

    #Heathy test set
    null = np.random.normal(loc=0.0, scale=1.0, size=[nt, N0])
    nonnull = np.random.normal(loc=-delta, scale=1.0, size=[nt, N - N0])
    healthy_test = np.concatenate([null, nonnull], axis=-1)

    num_correct = (sick_test @ w >= 0.).astype(np.float64).sum() + (healthy_test @ w < 0.).astype(np.float64).sum()
    percent_correct = num_correct / float(2 * nt)

    return percent_correct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--N0', type=int, default=500)
    parser.add_argument('--delta', type=float, default=1.0)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--n', type=int, default=100)
    args = parser.parse_args()

    print(sys.argv)
    print(args)

    go(args.N, args.N0, args.delta, args.c, args.n)

if __name__ == '__main__':
    main()
