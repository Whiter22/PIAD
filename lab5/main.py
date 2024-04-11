import pandas as pd
import numpy as np

#zad_1
def freq(col, prob):
    u_col = col.unique()
    # print(u_col)

    l = len(col)
    p = []
    n = {key: 0 for key in u_col}

    for u in col:
        n[u] += 1
    
    for key in n:
        p.append(n[key]/l)

    if prob:
        return [u_col, p]
    else:
        return [u_col, n]
    # return [x.unique(), ]


# data = pd.read_csv('C:/Users/rubin/OneDrive/Pulpit/PIAD/lab5/zoo.csv')
# col = data['legs']
# [xi, ni] = freq(col, False)
# print(xi, '\t', ni)


#zad_2
def freq2(x, y, prob):
    x_u = x.unique()
    y_u = y.unique()

    p = []
    n = {key: 0 for key in x_u}
    n1 = {key: 0 for key in y_u}

    for u in x:
        n[u] += 1    

    for u in y:
        n1[u] += 1

    n.update(n1) 

    l = len(n)
    for key in n:
        p.append(n[key]/l)

    if prob:
        return [x_u, y_u, p]
    else:
        return [x_u, y_u, n]


# col1 = data['animal']
# [xi, yi, ni] = freq2(col, col1, False)
# print(xi, '\t', yi, '\t', ni)
# print(sum(ni))


def entropy(prob):
    
    pass