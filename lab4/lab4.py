import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# from skimage import data
# from skimage import exposure

###############################################################################
# Dyskretyzacja

# def discrite(f, fs):
#     t = np.arange(0, 1, 1/fs)
#     s = np.sin(2*np.pi*f*t)
#     return t, s


# freq = 10 #Hz
# samples = [20, 21, 30, 45, 50, 100, 150, 200, 250, 1000]

# for sample in samples:
#     t, s = discrite(freq, sample)
#     plt.plot(t, s)
#     plt.show()

'''
4: twierdzenie Nyquista
5: aliasing
'''
###############################################################################
# Kwantyzacja

#axis = 1 - row, axis = 2 - col, axis = 2 depth
jasnosc = lambda img: (np.max(img, axis=2) + np.min(img, axis=2))/2    
usrednienie = lambda img: img.mean(axis=2)
luminacja = lambda img: np.dot(img, [0.21, 0.72, 0.07])

# 3.1
img = np.array(Image.open('img1.png'))
# 3.2
print(f'Shape: {img.shape}')
# 3.3
print(f'Depth: {img[0, 0].size}')

jas = jasnosc(img)
usr = usrednienie(img)
lum = luminacja(img)

met1, bins1 = np.histogram(jas.flatten(), bins=256, range=[0,255])
met2, bins2 = np.histogram(usr.flatten(), bins=256, range=[0,255])
met3, bins3 = np.histogram(lum.flatten(), bins=256, range=[0,255])
reduced, redu_bins = np.histogram(img.flatten(), bins=16, range=[0, 255])

print(f"stare zakresy:")
for i in range(len(redu_bins)-1):
    print(f"{redu_bins[i]} - {redu_bins[i+1]}")

#obliczanie srodkowej wartosci dla kazdego z przedzialow bin
midd_vals = (redu_bins[:-1] + redu_bins[1:])/2

print(f"nowe zakresy:")
for i in range(len(midd_vals)-1):
    print(f"{midd_vals[i]} - {midd_vals[i+1]}")

# -1 bo ostatni jest gorna granica ostatniego przedzialu
redu_img = np.digitize(img, redu_bins[:-1])

# zastepujemy kazdy piksel zredukowanego obrazu wartoscia wysrodkowana z danego przedzialu
for i in range(len(midd_vals)):
    redu_img[redu_img == i] = midd_vals[i]

plt.figure(figsize=(12, 6))

plt.subplot(2, 5, 1)
plt.imshow(img) 
plt.title('Original image')

plt.subplot(2, 5, 2)
plt.hist(jas.ravel(), color='red', edgecolor='black')
plt.title('Jasnosc')

plt.subplot(2, 5, 3)
plt.hist(usr.ravel(), color='green', edgecolor='black')
plt.title('Usrednienie')

plt.subplot(2, 5, 4)
plt.hist(lum.ravel(), color='blue', edgecolor='black')
plt.title('Luminacja')

plt.subplot(2, 5, 5)
plt.hist(redu_img.ravel(), color='orange', edgecolor='black')
plt.title('Reduced')

plt.subplot(2, 5, 7)
plt.imshow(jas)
plt.title('↓')

plt.subplot(2, 5, 8)
plt.imshow(usr)
plt.title('↓')

plt.subplot(2, 5, 9)
plt.imshow(lum)
plt.title('↓')

plt.subplot(2, 5, 10)
plt.imshow(redu_img)
plt.title('↓')

plt.tight_layout()
plt.show()

################################################################################
# 4 Binaryzacja

# 4.2
img = np.array(Image.open('gradient2.png'))
img_gray = jasnosc(img)

hist, bins = np.histogram(img_gray.ravel(), bins=256, range=[0,255])

# 4.3
def find_threshold(hist):
    #otsu method

    hist = hist.astype("float")
    # Obliczenie całkowitej liczby pikseli w obrazie
    total_pixels = np.sum(hist)
    # Obliczenie sumy intensywności pikseli
    sum_intensity = np.dot(np.arange(256), hist) # dla sredniej mF
    # print(sum_intensity)

    sumB = 0
    wB = 0 # waga Background'u
    maximum = 0
    threshold = 0

    # Iteracja przez możliwe wartości progu
    for t in range(256):
        # Dodanie wagi aktualnego piksela do sumy wag pikseli dla tła
        wB += hist[t] # liczba pikseli wzietych jako tlo
        # Kontynuacja pętli jeśli suma wag pikseli dla tła wynosi 0
        if wB == 0:
            continue
        # Obliczenie sumy wag pikseli dla pierwszego planu
        wF = total_pixels - wB # liczba pikseli ktore nie zostaly jeszcze zklasyfikowane jako tlo 
        # Przerwanie pętli jeśli suma wag pikseli dla pierwszego planu wynosi 0
        if wF == 0: # brak dalszych pikseli 
            break
        # Dodanie wartości intensywności piksela pomnożonej przez ilość pikseli o danej intensywności do sumy intensywności pikseli dla tła
        sumB += t * hist[t] # i = t hist[i] = pi
        # Obliczenie średniej intensywności piksela dla tła
        mB = sumB / wB #μ1
        # Obliczenie średniej intensywności piksela dla pierwszego planu
        mF = (sum_intensity - sumB) / wF #μ0
        # Obliczenie wariancji międzyklasowej dla aktualnego progu
        between_class_variance = wB * wF * (mB - mF) ** 2   # ω0*w1*(μ1 - μ0)^2
        # Aktualizacja maksimum i wartości progu jeśli aktualna wariancja międzyklasowa jest większa
        if between_class_variance > maximum:
            maximum = between_class_variance
            threshold = t

    return threshold

plt.figure(figsize=(10,5))

plt.subplot(1,4, 1)
plt.imshow(img)
plt.title('Gradient img')

plt.subplot(1, 4, 2)
plt.hist(img_gray.ravel(), color='orange', bins=256)
plt.title('Gradient to gray')

# 4.4
threshold = find_threshold(hist)
binary_image = np.where(img_gray > threshold, 1, 0)
 
# 4.5
plt.subplot(1, 4, 3)
plt.imshow(binary_image, cmap = 'gray')
plt.title('Binary Image')

plt.subplot(1, 4, 4)
plt.hist(binary_image.ravel(), range=[0,1])
plt.title('Binary Image')
plt.tight_layout()
plt.show()