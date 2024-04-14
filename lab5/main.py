import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

#zad_1
def freq(col, prob):
    u_col = col.unique()

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


data = pd.read_csv('lab5/zoo.csv')
col = data['eggs']
a = [xi, ni] = freq(col, True)
print('Zadanie_1:\n', xi, '\n', ni, '\n\n')
# print(sum(ni))

################################################################################
#zad_2
def freq2(x, y, prob=True):
    freq_dict = {}
    
    # Iteracja po parach wartości x i y
    for xi, yi in zip(x, y):
        # Aktualizacja liczności danej pary
        freq_dict[(xi, yi)] = freq_dict.get((xi, yi), 0) + 1
    
    # Jeśli prob=True, zamień liczności na częstości
    if prob:
        total_count = sum(freq_dict.values())
        freq_dict = {key: value / total_count for key, value in freq_dict.items()}
    
    # Rozdzielenie kluczy na xi i yi
    xi = [key[0] for key in freq_dict]
    yi = [key[1] for key in freq_dict]
    ni = list(freq_dict.values())
    
    return xi, yi, ni


x = data['type']
y = data['eggs']
xi, yi, ni = freq2(x, y, prob=True)
df = pd.crosstab(index=pd.Series(xi, name='Type'), columns=pd.Series(yi, name='Eggs'), values=ni, aggfunc='sum', margins=True, margins_name='Total')
df.fillna(0, inplace=True)
print('Zadanie_2:\n', df)
# print("xi:", xi)
# print("yi:", yi)
# print("ni:", ni)

################################################################################
#zad_3
def entropy(prob):
    n = len(prob)
    ent = 0

    for i in range(n):
        if prob[i] != 0:
            ent += prob[i] * np.log2(prob[i])

    return -ent

print('\nZadanie_3:')
print(f'ent: {entropy(a[1])}')

def infogain(x, y):
    # Obliczenie entropii dla kolumn x i y
    hx = entropy(freq(x, prob=True)[1])
    hy = entropy(freq(y, prob=True)[1])
    
    # Obliczenie entropii dla dwóch kolumn połączonych
    join_entropy = entropy(freq2(x, y, prob=True)[2])
    
    # Obliczenie informacji wzajemnej
    mutual_info = hx + hy - join_entropy
    
    return mutual_info

ig = infogain(x, y)
print(f'Informacja wzajemna: {ig}\n\n')

################################################################################
#zad_4
target_col = 'type'

info_gains = {}
for column in data.columns:
    if column != target_col:
        info_gains[column] = infogain(data[column], data[target_col])

sorted_info_gains = sorted(info_gains.items(), key=lambda x: x[1], reverse=True)

print(f'Zadanie_4:\nNajwiekszy przyrost info: {sorted_info_gains[0][0]}: {sorted_info_gains[0][1]}')
for column, gain in sorted_info_gains[1:]:
    print(f'{column}: {gain}')

################################################################################
#zad_5
def freq_sparse(col, prob=True):
    if hasattr(col, 'toarray'):  # Sprawdzenie czy kolumna jest rzadką macierzą
        col = col.toarray().flatten()

    u_col = np.unique(col)
    l = len(col)
    n = {key: 0 for key in u_col}

    for u in col:
        n[u] += 1
    
    if prob:
        p = [n[key]/l for key in n]
        return [u_col, p]
    else:
        return [u_col, list(n.values())]
    

def freq2_sparse(x, y, prob=True):
    freq_dict = {}
    
    # Konwersja rzadkich macierzy na tablice
    if hasattr(x, 'toarray'):
        x = x.toarray().flatten()
    if hasattr(y, 'toarray'):
        y = y.toarray().flatten()
    
    # Iteracja po parach wartości x i y
    for xi, yi in zip(x, y):
        freq_dict[(xi, yi)] = freq_dict.get((xi, yi), 0) + 1
    
    if prob:
        total_count = sum(freq_dict.values())
        freq_dict = {key: value / total_count for key, value in freq_dict.items()}
    
    # Rozdzielenie kluczy na xi i yi
    xi = [key[0] for key in freq_dict]
    yi = [key[1] for key in freq_dict]
    ni = list(freq_dict.values())
    
    return xi, yi, ni


data_sparse = csr_matrix(data['eggs'])
# print('\n\n', freq(data_sparse, True))
# print('\n', freq2(data_sparse, data_sparse, True))
# data_sparse = data['eggs']
print('\n\nZadanie_5:\n', freq_sparse(data_sparse))
print('\n', freq2_sparse(data_sparse, data_sparse))
