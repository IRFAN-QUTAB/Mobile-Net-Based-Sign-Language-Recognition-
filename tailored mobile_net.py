#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# In[ ]:



X = np.load("/content/drive/MyDrive/ASL_preprocessed_images.npy")
y = np.load("/content/drive/MyDrive/ASL_labels.npy")

X = X.astype("float32") / 255.0

num_classes = len(np.unique(y))
y = to_categorical(y, num_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")


# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model

def build_mobilenet_model(input_shape=(224, 224, 3), num_classes=26):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)

    for filters in [64, 128, 128, 256, 256, 512, 512]:
        x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, (1, 1), activation='relu')(x)
        x = BatchNormalization()(x)

    x = Conv2D(1024, (1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x) 

    model = Model(inputs, outputs)
    return model

model = build_mobilenet_model()
model.summary()


# In[ ]:


from sklearn.model_selection import ParameterGrid

param_grid = {
    "batch_size": [32, 64],
    "learning_rate": [0.001, 0.0001],
    "optimizer": ["adam", "rmsprop", "sgd", "adagrad"]
}

best_acc = 0
best_params = {}

for params in ParameterGrid(param_grid):
    print(f"Testing params: {params}")

    if params["optimizer"] == "adam":
        optimizer = Adam(learning_rate=params["learning_rate"])
    elif params["optimizer"] == "rmsprop":
        optimizer = RMSprop(learning_rate=params["learning_rate"])
    elif params["optimizer"] == "sgd":
        optimizer = SGD(learning_rate=params["learning_rate"])
    else:
        optimizer = Adagrad(learning_rate=params["learning_rate"])

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        batch_size=params["batch_size"], epochs=10, verbose=1)

    _, acc = model.evaluate(X_test, y_test, verbose=0)
 
    if acc > best_acc:
        best_acc = acc
        best_params = params

print(f"Best Hyperparameters: {best_params} with Accuracy: {best_acc:.4f}")


# In[ ]:



optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    batch_size=32, epochs=100, verbose=1)

model.save("/content/drive/MyDrive/ASL_MobileNet_100epochs.h5")
print("Model training completed and saved successfully!")


# In[ ]:


model = tf.keras.models.load_model("/content/drive/MyDrive/ASL_MobileNet_100epochs.h5")
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=-3).output)
num_samples = 100
random_indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
image_batch = X_test[random_indices]  
image_batch = image_batch.astype("float32") / 255.0

features = feature_extractor.predict(image_batch)

features = features.reshape(features.shape[0], -1)  


# In[ ]:


pca = PCA(n_components=2)
pca_features = pca.fit_transform(features)

tsne = TSNE(n_components=2, random_state=42, perplexity=10)
tsne_features = tsne.fit_transform(features)

# Create a combined figure with two subplots (PCA & t-SNE)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

# Plot PCA Scatter Plot
axes[0].scatter(pca_features[:, 0], pca_features[:, 1], c='blue', label="PCA Projection")
axes[0].set_xlabel("Feature Dimension 1")
axes[0].set_ylabel("Feature Dimension 2")
axes[0].legend()
axes[0].grid()
axes[0].set_title("PCA Feature Representation")

# Plot t-SNE Scatter Plot
axes[1].scatter(tsne_features[:, 0], tsne_features[:, 1], c='red', label="t-SNE Projection")
axes[1].set_xlabel("Embedded Feature 1")
axes[1].set_ylabel("Embedded Feature 2")
axes[1].legend()
axes[1].grid()
axes[1].set_title("t-SNE Feature Representation")

# Adjust layout and show the combined plot
plt.tight_layout()
plt.show()


# In[ ]:




