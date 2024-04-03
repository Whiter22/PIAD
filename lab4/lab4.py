import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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


# img = np.array(Image.open('C:/Users/bolec/OneDrive/Pulpit/PIAD/lab4/img1.png'))
# print(f'Shape: {img.shape}')
# print(f'Depth: {img[0, 0].size}')

# jas = jasnosc(img)
# usr = usrednienie(img)
# lum = luminacja(img)

# met1, bins1 = np.histogram(jas.flatten(), bins=256, range=[0,255])
# met2, bins2 = np.histogram(usr.flatten(), bins=256, range=[0,255])
# met3, bins3 = np.histogram(lum.flatten(), bins=256, range=[0,255])
# reduced, redu_bins = np.histogram(lum.flatten(), bins=17, range=[0, 16])

# #obliczanie srodkowej wartosci dla kazdego z przedzialow bin
# midd_vals = (redu_bins[:-1] + redu_bins[1:])/2

# # -1 bo ostatni jest gorna granica ostatniego przedzialu
# redu_img = np.digitize(lum, redu_bins[:-1])

# # zastepujemy kazdy piksel zredukowanego obrazu wartoscia wysrodkowana z danego przedzialu
# for i in range(len(midd_vals)):
#     redu_img[redu_img == i] = midd_vals[i]


# plt.figure(figsize=(12, 6))

# plt.subplot(2, 5, 1)
# plt.imshow(img) 
# plt.title('Original image')

# plt.subplot(2, 5, 2)
# plt.hist(met1, color='red', edgecolor='black')
# plt.title('Jasnosc')

# plt.subplot(2, 5, 3)
# plt.hist(met2, color='green', edgecolor='black')
# plt.title('Usrednienie')

# plt.subplot(2, 5, 4)
# plt.hist(met3, color='blue', edgecolor='black')
# plt.title('Luminacja')

# plt.subplot(2, 5, 5)
# plt.hist(reduced, color='orange', edgecolor='black')
# plt.title('Reduced')

# plt.subplot(2, 5, 7)
# plt.imshow(jas, cmap='gray')
# plt.title('↓')

# plt.subplot(2, 5, 8)
# plt.imshow(usr, cmap='gray')
# plt.title('↓')

# plt.subplot(2, 5, 9)
# plt.imshow(lum, cmap='gray')
# plt.title('↓')

# plt.subplot(2, 5, 10)
# plt.imshow(redu_img, cmap='gray')
# plt.title('↓')

# plt.tight_layout()
# plt.show()

################################################################################
# Binaryzacja

def find_threshold(hist):
    hist = hist.astype("float")
    between_class_variances = []
    for t in range(256):
        # Background
        background_pixels = np.sum(hist[:t])
        background_weights = background_pixels / np.sum(hist[:])
        background_mean = np.sum(np.arange(0, t) * hist[:t]) / (background_pixels + 1e-6)

        # Foreground
        foreground_pixels = np.sum(hist[t:])
        foreground_weights = foreground_pixels / np.sum(hist[:])
        foreground_mean = np.sum(np.arange(t, 256) * hist[t:]) / (foreground_pixels + 1e-6)

        # Between-class variance
        between_class_variances.append(background_weights * foreground_weights * (background_mean - foreground_mean) ** 2)
        # print(between_class_variances)

    return np.argmax(between_class_variances)


img = np.array(Image.open('C:/Users/bolec/OneDrive/Pulpit/PIAD/lab4/gradient2.png'))
img_gray = jasnosc(img)

met1, bins1 = np.histogram(img_gray.flatten(), bins=256, range=[0,255])
# plt.subplot(1,4,1)
# plt.imshow(img)

# plt.subplot(1, 4, 2)
# plt.imshow(img_gray)

plt.subplot(1, 2, 1)
plt.hist(met1, color='magenta', edgecolor='black')
plt.title('Gradient to gray')

threshold = find_threshold(met1)
binary_image = np.where(img_gray > threshold, 1, 0)
plt.subplot(1, 2, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.show()