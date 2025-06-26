#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:



from google.colab import drive
drive.mount('/content/drive')


import os

DATASET_PATH = "/content/drive/MyDrive/ASL_Dataset"
if os.path.exists(DATASET_PATH):
    print("Dataset found at:", DATASET_PATH)
else:
    print("Error: Dataset path not found!")


# In[ ]:



IMG_HEIGHT, IMG_WIDTH = 224, 224


def load_images_from_folder(folder_path):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder_path)) 

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(folder_path, class_name)
        
        if os.path.isdir(class_folder): 
            for file in os.listdir(class_folder):
                img_path = os.path.join(class_folder, file)
                try:
                    img = cv2.imread(img_path) 
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    return np.array(images), np.array(labels), class_names

images, labels, class_names = load_images_from_folder(DATASET_PATH)
print(f"Loaded {len(images)} images from {len(class_names)} classes.")


# In[ ]:


def remove_no_sign_images(images, labels):
    filtered_images = []
    filtered_labels = []

    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
        edges = cv2.Canny(gray, 50, 150)

     
        if np.sum(edges) > 10000:  
            filtered_images.append(img)
            filtered_labels.append(labels[i])

    return np.array(filtered_images), np.array(filtered_labels)

images, labels = remove_no_sign_images(images, labels)
print(f"Filtered dataset size: {len(images)} images.")


# In[ ]:


def augment_images(images):
    augmented_images = []
    
    for img in images:
        flipped_h = cv2.flip(img, 1)
        flipped_v = cv2.flip(img, 0) 
        rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        augmented_images.extend([img, flipped_h, flipped_v, rotated_90])
    
    return np.array(augmented_images)

augmented_images = augment_images(images)
print(f"Total augmented images: {len(augmented_images)}")


# In[ ]:



fig, axes = plt.subplots(1, 4, figsize=(12, 4))
axes[0].imshow(images[0])
axes[0].set_title("Original")
axes[1].imshow(cv2.flip(images[0], 1))
axes[1].set_title("Flipped Horizontal")
axes[2].imshow(cv2.flip(images[0], 0))
axes[2].set_title("Flipped Vertical")
axes[3].imshow(cv2.rotate(images[0], cv2.ROTATE_90_CLOCKWISE))
axes[3].set_title("Rotated 90°")

for ax in axes:
    ax.axis('off')
plt.show()


# In[ ]:




