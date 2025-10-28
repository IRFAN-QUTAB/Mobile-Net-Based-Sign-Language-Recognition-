import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

class ASLDataLoader:
    def __init__(self, data_path, img_size=(224, 224)):
        self.data_path = data_path
        self.img_size = img_size
        self.images = []
        self.labels = []
        
    def preprocess_image(self, img, margin=20, augment=True):
        """Preprocessing + Augmentation pipeline (WITHOUT rotation)"""
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Noise reduction (slight blur)
        img_rgb = cv2.GaussianBlur(img_rgb, (3, 3), 0)

        # Hand detection
        results = mp_hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            return None  # skip if no hands detected

        # Compute bounding box across all detected hands
        h, w = img_rgb.shape[:2]
        x_min, y_min, x_max, y_max = w, h, 0, 0
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

        # Add margin
        x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
        x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)

        # Crop
        cropped = img_rgb[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            return None

        # Resize
        resized = cv2.resize(cropped, self.img_size, interpolation=cv2.INTER_AREA)

        # Apply augmentations (brightness and background blur only, NO rotation)
        if augment:
            resized = self.apply_augmentations(resized)

        return resized

    def apply_augmentations(self, img):
        """Apply augmentations: brightness, background blur (NO rotation here)"""
        h, w = img.shape[:2]

        # 1. Random brightness adjustment (0.8x – 1.2x)
        if np.random.random() < 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)

        # 2. Random background blur
        if np.random.random() < 0.3:
            blurred = cv2.GaussianBlur(img, (15, 15), 0)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2, 255, -1)
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            img = cv2.bitwise_and(img, img, mask=mask) + cv2.bitwise_and(
                blurred, blurred, mask=255 - mask
            )

        return img
    
    def apply_rotation(self, img, angle):
        """Apply exact rotation to an image"""
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return rotated
        
    def load_data(self):
        """Load AUSL dataset from directory structure"""
        print("Loading AUSL dataset...")
        print(f"Dataset path: {self.data_path}")
        
        # Define the expected classes (0-9, A-Z)
        expected_classes = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z')+1)]
        
        # Count total images first
        total_images = 0
        class_counts = {}
        
        for class_name in expected_classes:
            class_path = os.path.join(self.data_path, class_name)
            
            if not os.path.exists(class_path):
                print(f"Warning: Class folder '{class_name}' not found")
                continue
                
            if os.path.isdir(class_path):
                img_files = [f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                class_counts[class_name] = len(img_files)
                total_images += len(img_files)
        
        print(f"Found {total_images} total images across {len(class_counts)} classes")
        print("\nImages per class:")
        for cls, count in sorted(class_counts.items()):
            print(f"  Class {cls}: {count} images")
        
        kept = 0
        # Load and preprocess images with progress bar
        with tqdm(total=total_images, desc="Preprocessing images") as pbar:
            for class_name in expected_classes:
                class_path = os.path.join(self.data_path, class_name)
                
                if not os.path.exists(class_path) or not os.path.isdir(class_path):
                    continue
                
                # Get all image files
                img_files = [f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                
                for img_name in img_files:
                    img_path = os.path.join(class_path, img_name)
                    try:
                        # Load image
                        img = cv2.imread(img_path)
                        if img is not None:
                            # Preprocess with MediaPipe + augmentation (NO rotation)
                            preprocessed = self.preprocess_image(img, augment=True)
                            if preprocessed is not None:
                                self.images.append(preprocessed)
                                self.labels.append(class_name)
                                kept += 1
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                    pbar.update(1)
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        print(f"\n✓ Successfully loaded {kept}/{total_images} usable images (with hand detection)")
        print(f"✓ Number of classes: {len(np.unique(self.labels))}")
        print(f"✓ Image shape: {self.images[0].shape if len(self.images) > 0 else 'No images'}")
        
        return self.images, self.labels

DATA_PATH = "path to dataset"
loader = ASLDataLoader(DATA_PATH)
images, labels = loader.load_data()

def display_samples(images, labels, n_samples=12):
    """Display sample images from the dataset"""
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    indices = np.random.choice(len(images), n_samples, replace=False)
    
    for idx, ax in enumerate(axes.flat):
        img_idx = indices[idx]
        ax.imshow(images[img_idx])
        ax.set_title(f"Class: {labels[img_idx]}")
        ax.axis('off')
    
    plt.suptitle("Sample Images from AUSL Dataset (After Preprocessing)")
    plt.tight_layout()
    plt.show()

display_samples(images, labels)

print("\nEncoding labels...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
print(f"✓ Labels encoded. Classes: {label_encoder.classes_}")
print(f"✓ Number of classes: {len(label_encoder.classes_)}")

print("\nSplitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    images, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

print("\nApplying rotation augmentation (+5° and -5°) to training set...")
X_train_augmented = []
y_train_augmented = []

for img, label in tqdm(zip(X_train, y_train), total=len(X_train), desc="Rotating training images"):
    # Original image
    X_train_augmented.append(img)
    y_train_augmented.append(label)
    
    # +5 degree rotation
    img_plus5 = loader.apply_rotation(img, 5)
    X_train_augmented.append(img_plus5)
    y_train_augmented.append(label)
    
    # -5 degree rotation
    img_minus5 = loader.apply_rotation(img, -5)
    X_train_augmented.append(img_minus5)
    y_train_augmented.append(label)

X_train = np.array(X_train_augmented)
y_train = np.array(y_train_augmented)

print(f"✓ Training set after rotation augmentation: {X_train.shape[0]} samples")
print(f"✓ Original training samples: {len(X_train)//3}")
print(f"✓ Augmentation factor: 3x (original + 5° + -5°)")

print("\nNormalizing images to [0, 1] range...")
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(f"✓ Training set normalized. Shape: {X_train.shape}, Range: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"✓ Test set normalized. Shape: {X_test.shape}, Range: [{X_test.min():.2f}, {X_test.max():.2f}]")

print("="*60)
