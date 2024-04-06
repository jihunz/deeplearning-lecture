import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage.color import rgb2gray
from skimage.transform import resize

fig = plt.figure(figsize=(2, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

############################################
# 원본이미지
############################################

im_org = skimage.data.coffee()
im_org = resize(im_org, (64, 64))

im = rgb2gray(im_org)

#plt.axis("off")
#plt.imshow(im, cmap="gray")
#plt.show()

ax = fig.add_subplot(3, 2, 1)
ax.title.set_text("Original")
ax.imshow(im_org)

ax = fig.add_subplot(3, 2, 2)
ax.title.set_text("Gray")
ax.imshow(im, cmap="gray")


############################################
# 수평 엣지 필터
############################################

filter1 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

new_image = np.zeros(im.shape)

im_pad = np.pad(im, 1, "constant")

for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        try:
            new_image[i, j] = (
                im_pad[i - 1, j - 1] * filter1[0, 0]
                + im_pad[i - 1, j] * filter1[0, 1]
                + im_pad[i - 1, j + 1] * filter1[0, 2]
                + im_pad[i, j - 1] * filter1[1, 0]
                + im_pad[i, j] * filter1[1, 1]
                + im_pad[i, j + 1] * filter1[1, 2]
                + im_pad[i + 1, j - 1] * filter1[2, 0]
                + im_pad[i + 1, j] * filter1[2, 1]
                + im_pad[i + 1, j + 1] * filter1[2, 2]
            )
        except:
            pass

#plt.axis("off")
#lt.imshow(im, cmap="gray")
#plt.show()

ax = fig.add_subplot(3, 2, 3)
ax.imshow(new_image, cmap="gray")


############################################
# 수직 엣지 필터
############################################

filter2 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

new_image = np.zeros(im.shape)

im_pad = np.pad(im, 1, "constant")

for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        try:
            new_image[i, j] = (
                im_pad[i - 1, j - 1] * filter2[0, 0]
                + im_pad[i - 1, j] * filter2[0, 1]
                + im_pad[i - 1, j + 1] * filter2[0, 2]
                + im_pad[i, j - 1] * filter2[1, 0]
                + im_pad[i, j] * filter2[1, 1]
                + im_pad[i, j + 1] * filter2[1, 2]
                + im_pad[i + 1, j - 1] * filter2[2, 0]
                + im_pad[i + 1, j] * filter2[2, 1]
                + im_pad[i + 1, j + 1] * filter2[2, 2]
            )
        except:
            pass

#plt.axis("off")
#lt.imshow(im, cmap="gray")
#plt.show()

ax = fig.add_subplot(3, 2, 4)
ax.imshow(new_image, cmap="gray")

############################################
# 수평 엣지 필터 - 스트라이드 2
############################################

filter1 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

stride = 2

new_image = np.zeros((int(im.shape[0] / stride), int(im.shape[1] / stride)))

im_pad = np.pad(im, 1, "constant")

for i in range(0, im.shape[0], stride):
    for j in range(0, im.shape[1], stride):
        try:
            new_image[int(i / stride), int(j / stride)] = (
                im_pad[i - 1, j - 1] * filter1[0, 0]
                + im_pad[i - 1, j] * filter1[0, 1]
                + im_pad[i - 1, j + 1] * filter1[0, 2]
                + im_pad[i, j - 1] * filter1[1, 0]
                + im_pad[i, j] * filter1[1, 1]
                + im_pad[i, j + 1] * filter1[1, 2]
                + im_pad[i + 1, j - 1] * filter1[2, 0]
                + im_pad[i + 1, j] * filter1[2, 1]
                + im_pad[i + 1, j + 1] * filter1[2, 2]
            )
        except:
            pass

#plt.axis("off")
#lt.imshow(im, cmap="gray")
#plt.show()

ax = fig.add_subplot(3, 2, 5)
ax.imshow(new_image, cmap="gray")


############################################
# 수직 엣지 필터 - 스트라이드 2
############################################
filter2 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

stride = 2

new_image = np.zeros((int(im.shape[0] / stride), int(im.shape[1] / stride)))

im_pad = np.pad(im, 1, "constant")

for i in range(0, im.shape[0], stride):
    for j in range(0, im.shape[1], stride):
        try:
            new_image[int(i / stride), int(j / stride)] = (
                im_pad[i - 1, j - 1] * filter2[0, 0]
                + im_pad[i - 1, j] * filter2[0, 1]
                + im_pad[i - 1, j + 1] * filter2[0, 2]
                + im_pad[i, j - 1] * filter2[1, 0]
                + im_pad[i, j] * filter2[1, 1]
                + im_pad[i, j + 1] * filter2[1, 2]
                + im_pad[i + 1, j - 1] * filter2[2, 0]
                + im_pad[i + 1, j] * filter2[2, 1]
                + im_pad[i + 1, j + 1] * filter2[2, 2]
            )
        except:
            pass

#plt.axis("off")
#lt.imshow(im, cmap="gray")
#plt.show()

ax = fig.add_subplot(3, 2, 6)
ax.imshow(new_image, cmap="gray")

plt.show()