#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad
from sklearn.metrics import classification_report

dataset_path = "/content/drive/MyDrive/ASL_preprocessed_images.npy"
data = np.load(dataset_path, allow_pickle=True)

X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]

X_train = X_train / 255.0
X_test = X_test / 255.0

optimizers = {
    "Adam": Adam(learning_rate=0.0001),
    "RMSprop": RMSprop(learning_rate=0.0001),
    "SGD": SGD(learning_rate=0.0001),
    "Adagrad": Adagrad(learning_rate=0.0001)
}

def build_model():
    base_model = DenseNet169(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze pretrained layers

    inputs = base_model.input
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.4)(x)
    outputs = Dense(26, activation='softmax')(x)  # 26 ASL classes

    model = Model(inputs, outputs)
    return model

#DenseNet169
print("\nTesting DenseNet169...\n")
for opt_name, optimizer in optimizers.items():
    model = build_model()
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)
    acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nDenseNet169 with {opt_name} - Test Accuracy: {acc * 100:.2f}%")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(f"\nDenseNet169 with {opt_name} - Classification Report:\n")
    print(classification_report(y_true, y_pred))


# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]

X_train = X_train / 255.0
X_test = X_test / 255.0

optimizers = {
    "Adam": Adam(learning_rate=0.0001),
    "RMSprop": RMSprop(learning_rate=0.0001),
    "SGD": SGD(learning_rate=0.0001),
    "Adagrad": Adagrad(learning_rate=0.0001)
}

def build_model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze pretrained layers

    inputs = base_model.input
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.4)(x)
    outputs = Dense(26, activation='softmax')(x)  # 26 ASL classes

    model = Model(inputs, outputs)
    return model

#ResNet50
print("\nTesting ResNet50...\n")
for opt_name, optimizer in optimizers.items():
    model = build_model()
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)

    acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nResNet50 with {opt_name} - Test Accuracy: {acc * 100:.2f}%")

    # Get Predictions
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Print Classification Report
    print(f"\nResNet50 with {opt_name} - Classification Report:\n")
    print(classification_report(y_true, y_pred))


# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]

X_train = X_train / 255.0
X_test = X_test / 255.0

optimizers = {
    "Adam": Adam(learning_rate=0.0001),
    "RMSprop": RMSprop(learning_rate=0.0001),
    "SGD": SGD(learning_rate=0.0001),
    "Adagrad": Adagrad(learning_rate=0.0001)
}

def build_model():
    base_model = Xception(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze pretrained layers

    inputs = base_model.input
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.4)(x)
    outputs = Dense(26, activation='softmax')(x)  # 26 ASL classes

    model = Model(inputs, outputs)
    return model

#Xception
print("\nTesting Xception...\n")
for opt_name, optimizer in optimizers.items():
    model = build_model()
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)

    acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nXception with {opt_name} - Test Accuracy: {acc * 100:.2f}%")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(f"\nXception with {opt_name} - Classification Report:\n")
    print(classification_report(y_true, y_pred))


# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]
X_train = X_train / 255.0
X_test = X_test / 255.0

optimizers = {
    "Adam": Adam(learning_rate=0.0001),
    "RMSprop": RMSprop(learning_rate=0.0001),
    "SGD": SGD(learning_rate=0.0001),
    "Adagrad": Adagrad(learning_rate=0.0001)
}

def build_model():
    base_model = NASNetMobile(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze pretrained layers

    inputs = base_model.input
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.4)(x)
    outputs = Dense(26, activation='softmax')(x)  # 26 ASL classes

    model = Model(inputs, outputs)
    return model

#NASNetMobile
print("\nTesting NASNetMobile...\n")
for opt_name, optimizer in optimizers.items():
    model = build_model()
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)
    acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nNASNetMobile with {opt_name} - Test Accuracy: {acc * 100:.2f}%")
    
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(f"\nNASNetMobile with {opt_name} - Classification Report:\n")
    print(classification_report(y_true, y_pred))


# In[ ]:




