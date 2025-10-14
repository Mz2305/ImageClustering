import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

name_image = "person"

image = cv2.imread(f'resources/{name_image}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to a 2D array of pixels
pixels = image.reshape((-1, 3)) # Each pixel is represented by its RGB values

# KMeans
number_of_colors = 32
kmeans = KMeans(n_clusters=number_of_colors)
kmeans.fit(pixels)
colors = np.array(kmeans.cluster_centers_, dtype='uint8')

labels = kmeans.labels_
segmented_image = colors[labels].reshape(image.shape)
fig, ax = plt.subplots(1, 2, figsize=(12, 6)) #fig = create a figure, ax = create subplots
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(segmented_image)
ax[1].set_title(f'Image with {number_of_colors} Dominant Colors')
ax[1].axis('off')
plt.show()

output_path = os.path.join("output", f"output_{name_image}_{number_of_colors}.jpg")
cv2.imwrite(output_path, segmented_image)