# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:20:12 2022

@author: anton
"""
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys


def longueur(x, y, ordre):
    i = ordre[-1]
    x0, y0 = x[i], y[i]
    d = 0
    for o in ordre:
        x1, y1 = x[o], y[o]
        d += (x0 - x1) ** 2 + (y0 - y1) ** 2
        x0, y0 = x1, y1
    return d / 1000


def permutation(x, y, ordre):
    d = longueur(x, y, ordre)
    d0 = d + 1
    it = 1
    while d < d0:
        if ordre[0] != 0:
            sys.exit()
        it += 1
        print("iteration", it, "d=", d)
        d0 = d
        for i in tqdm(range(1, len(ordre) - 1)):
            for j in range(i + 2, len(ordre)):
                r = ordre[i:j].copy()
                r.reverse()
                ordre2 = ordre[:i] + r + ordre[j:]
                t = longueur(x, y, ordre2)
                if t < d:
                    d = t
                    ordre = ordre2

        plt.clf()
        xo = [x[o] for o in ordre + [ordre[0]]]
        yo = [y[o] for o in ordre + [ordre[0]]]
        plt.plot(xo, yo, "o-")
        plt.plot(xo[0], y[0], "r")
        plt.show()

    return ordre


def generate_order(graph):
    x = [graph.nodes[i]['pos'][0] for i in range(0, len(graph))]
    y = [graph.nodes[i]['pos'][1] for i in range(0, len(graph))]
    ordre = list(range(len(x)))
    ordre = permutation(x, y, ordre)
    print("longueur initiale", longueur(x, y, ordre))
    plt.plot(x, y, "o")
    print("longueur min", longueur(x, y, ordre))
    xo = [x[o] for o in ordre + [ordre[0]]]
    yo = [y[o] for o in ordre + [ordre[0]]]
    plt.plot(xo, yo, "o-")
    plt.text(xo[0], yo[0], "0", color="r", weight="bold", size="x-large")
    plt.text(xo[-2], yo[-2], "N-1", color="r", weight="bold", size="x-large")

    return ordre
