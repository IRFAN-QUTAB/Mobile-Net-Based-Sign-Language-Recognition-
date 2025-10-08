import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, LayerNormalization
from tensorflow.keras.models import Model, Sequential
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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

# 2. Self-Attention Block
def build_self_attention_block():
    """Build Self-Attention mechanism for feature enhancement"""
    
    class SelfAttention(keras.layers.Layer):
        def __init__(self, units):
            super(SelfAttention, self).__init__()
            self.units = units
            
        def build(self, input_shape):
            self.W_query = Dense(self.units)
            self.W_key = Dense(self.units)
            self.W_value = Dense(self.units)
            self.W_output = Dense(input_shape[-1])
            
        def call(self, x):
            # For CNN features, we need to reshape
            batch_size = tf.shape(x)[0]
            height = tf.shape(x)[1]
            width = tf.shape(x)[2]
            channels = x.shape[-1]
            
            # Reshape to (batch, height*width, channels)
            x_reshaped = tf.reshape(x, [batch_size, height * width, channels])
            
            # Compute Q, K, V
            Q = self.W_query(x_reshaped)
            K = self.W_key(x_reshaped)
            V = self.W_value(x_reshaped)
            
            # Compute attention scores
            scores = tf.matmul(Q, K, transpose_b=True)
            scores = scores / tf.math.sqrt(tf.cast(self.units, tf.float32))
            
            # Apply softmax
            attention_weights = tf.nn.softmax(scores, axis=-1)
            
            # Apply attention to values
            attention_output = tf.matmul(attention_weights, V)
            
            # Final linear transformation
            output = self.W_output(attention_output)
            
            # Reshape back to original shape
            output = tf.reshape(output, [batch_size, height, width, channels])
            
            # Add residual connection
            output = x + output
            
            return output
    
    # Build the self-attention model
    inputs = keras.Input(shape=(224, 224, 3))
    
    # Initial convolution layers to extract features
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    
    # Apply self-attention
    x = SelfAttention(units=128)(x)
    
    # Final pooling and dense layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(256, activation='relu')(x)
    
    model = Model(inputs, outputs, name="Self_Attention_Block")
    return model

# Build individual models
mobilenet_model = build_mobilenet_extractor()
self_attention_model = build_self_attention_block()

print(f"✓ MobileNet extractor built: Output shape = {mobilenet_model.output_shape}")
print(f"✓ Self-Attention block built: Output shape = {self_attention_model.output_shape}")

# ============================================
# CELL 6: BUILD HYBRID MODEL
# ============================================
print("Building hybrid MobileNet + Self-Attention model...")

# Combine both models
inputs = keras.Input(shape=(224, 224, 3))

# Get features from both models
mobilenet_features = mobilenet_model(inputs)
attention_features = self_attention_model(inputs)

# Concatenate features
combined = keras.layers.Concatenate()([mobilenet_features, attention_features])
combined = Dropout(0.3)(combined)
combined = Dense(512, activation='relu')(combined)
combined = LayerNormalization()(combined)  # Add layer normalization
combined = Dropout(0.2)(combined)
outputs = Dense(256, activation='relu')(combined)

# Create hybrid model
hybrid_model = Model(inputs, outputs, name="MobileNet_SelfAttention_Hybrid")

print(f"✓ Hybrid model built")
print(f"✓ Input shape: {hybrid_model.input_shape}")
print(f"✓ Output shape: {hybrid_model.output_shape}")
print(f"✓ Total parameters: {hybrid_model.count_params():,}")

# Compile the model for better initialization
hybrid_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
)
print("✓ Model compiled and ready for feature extraction")

# Create feature extractor (remove final softmax layer)
feature_extractor = Model(
    inputs=hybrid_model.input,
    outputs=hybrid_model.layers[-3].output  # the Dense(512, relu) layer
)

X_train_features = feature_extractor.predict(X_train, batch_size=32, verbose=1)
print(f"✓ Training features extracted: {X_train_features.shape}")

X_test_features = feature_extractor.predict(X_test, batch_size=32, verbose=1)
print(f"✓ Test features extracted: {X_test_features.shape}")

# Check model size
feature_extractor_size = os.path.getsize('hybrid_feature_extractor.h5') / (1024 * 1024)
print(f"\n✓ Feature Extractor Model Size: {feature_extractor_size:.2f} MB")
