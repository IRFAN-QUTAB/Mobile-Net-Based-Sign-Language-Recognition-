#Tailored MobileNet Self-Attention KNN (TMSK) for Sign Language Recognition

This repository presents a hybrid deep learning and machine learning framework designed for efficient and accurate sign language recognition (SLR). The proposed Tailored MobileNet Self-Attention KNN (TMSK) model combines the feature extraction capability of MobileNet, the global spatial understanding of a Self-Attention module, and the robust classification power of K-Nearest Neighbors (KNN). The model is optimized for portability, low memory usage, and real-time performance, making it suitable for deployment on mobile and embedded systems.

#Overview
🏗️ Architecture

The TMSK architecture is composed of three major components:

##Tailored MobileNet Backbone:

Modified lightweight convolutional layers for efficient local feature extraction.

Depthwise separable convolutions reduce computation while preserving detail.

##Self-Attention Block:

Enhances the feature map by modeling long-range spatial dependencies.

Improves the network’s understanding of complex gestures and overlapping hand movements.

##KNN Classifier:

Replaces the traditional dense classification head with a lightweight and interpretable KNN model.

Achieves high accuracy with minimal memory footprint (~5 MB).

⚙️ Preprocessing & Data Augmentation

A robust preprocessing pipeline was implemented to ensure clean and consistent input across datasets:

Hand Detection and Cropping: Using MediaPipe Hands to detect hand landmarks and crop bounding boxes.

Fixed Margin Cropping: A 20-pixel margin added to preserve context around the hand region.

Noise Reduction: Gaussian blurring to smooth illumination variations.

Resizing: All images resized to 224×224 pixels for MobileNet compatibility.

🗂️ Datasets Used

The model was trained and tested on multiple benchmark datasets to assess performance and generalization:

Dataset	Type	Classes	Samples	Description
Australian Sign Language dataset with alphabets and digits
(MNIST)	American Sign Language digits and alphabets
ASL Finger Spelling	Alphabetic with Spcal Signs
Indian Sign Language alphabets (complex backgrounds)
LSA64	Video	64 videos	Argentinian Sign Language dynamic dataset (verbs and nouns)

📈 Visualization

Learning Curves: Training and validation accuracy across epochs.

Confusion Matrices: Class-wise prediction performance.

Feature Fusion Diagram: MobileNet + Self-Attention integration.

Prediction Visualization: Correct (green) and incorrect (red) results.
