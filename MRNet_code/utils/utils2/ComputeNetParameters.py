

import numpy as np


def computerNetParameters(net):
    params = list(net.parameters())
    k = 0
    for index, i in enumerate(params):
        l = 1
        print(index+1, "layer structure:" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("layer paramenters: " +str(l))
        k += l
    print("network paramenters: " +str(k))


    return k
