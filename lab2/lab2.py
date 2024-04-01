import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde

# df = pd.DataFrame({"x": [1, 2, 3, 4, 5], 'y': ['a', 'b', 'a',
# 'b', 'b']})


# def zad1():
#     mean_val = df.groupby('y')['x'].mean()
#     print(f"zad1 mean_val:\n{mean_val}")

# zad1()


# def zad2():
#     val_counts = df['y'].value_counts()
#     print(f"\nzad2 val_counts:\n{val_counts}")

# zad2()


def zad3():
    data_array = np.loadtxt("C:/Users/rubin/OneDrive/Pulpit/PIAD/autos.csv", delimiter=",", dtype=str)
    print(f"\nzad3\nautos by np\n{data_array}\n")
    data_df = pd.read_csv("C:/Users/rubin/OneDrive/Pulpit/PIAD/autos.csv")
    print(f"autos by pandas\n{data_df}")
    return data_df

df = zad3()


# def zad4_5():
#     make_fuel = df.groupby('make')['fuel-type'].value_counts()
#     mean_fuel = make_fuel.groupby('make').mean()
#     print(f"\nzad4 mean consumption:\n{mean_fuel}")
#     print(f"\nzad5 fuel-type:\n{make_fuel}")

# zad4_5()


def zad6_7_8():
    x = df['city-mpg']
    ls = np.linspace(10, 60, 100)

    #zad6
    coeff = np.polyfit(x, df['length'], 2)
    y = np.polyval(coeff, ls)

    coeff = np.polyfit(x, df['length'], 1)
    y1 = np.polyval(coeff, ls)

    #zad8
    plt.scatter(x, df['length'], color='green')
    plt.plot(ls, y)
    plt.plot(ls, y1)
    plt.show()

    #zad7
    correlation, p_value = pearsonr(x, df['length'])
    print(f"\nzad7\ndla 2 st correlation: {correlation}\np_value: {p_value}")

zad6_7_8()


def zad9_10():
    #probki_8
    
    #estymator
    kde = gaussian_kde(df['length'])(df['length'])
    plt.plot(df['length'], kde.T, 'o', color='red')
    plt.hist(df['length'], density=True, alpha=0.5, label='Próbki')
    plt.show()

    kde = gaussian_kde(df['width'])(df['width'])
    plt.plot(df['length'], kde.T, 'o', color='red')
    

zad9_10()

def zad11():
    X = df['length']
    Y = df['width']
    plt.plot(X,Y, '.')
    X, Y = np.mgrid[X.min():X.max():100j,Y.min():Y.max():100j]
    est = gaussian_kde(np.vstack([df['length'],df['width']]))
    Z = np.reshape(est(np.vstack([X.ravel(),Y.ravel()])).T,X.shape)
    plt.contour(X,Y,Z)
    plt.show()

zad11()
