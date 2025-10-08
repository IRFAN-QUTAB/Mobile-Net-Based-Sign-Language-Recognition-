import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
import re

# Function to crop the middle part of the frame (augmentation)
def crop_center(image, crop_size=(160, 160)):
    h, w, _ = image.shape
    crop_h, crop_w = crop_size
    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2
    return image[start_y:start_y+crop_h, start_x:start_x+crop_w]

class LSAVideoLoader:
    def __init__(self, video_dir, img_size=(224, 224), save_path=None):
        self.video_dir = video_dir
        self.img_size = img_size
        self.save_path = save_path

    def extract_class_from_filename(self, filename):
        """Extract class label from filename pattern"""
        match = re.match(r'(\d+)_(\d+)_(\d+)', filename)
        if match:
            return match.group(1)  # Return first group as class
        return None

    def extract_frames(self, video_path, n_frames=16, crop=False):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return None
            
        frames = []
        
        if total_frames >= n_frames:
            indices = np.linspace(0, total_frames-1, n_frames, dtype=int)
        else:
            indices = list(range(total_frames))
            while len(indices) < n_frames:
                indices.append(total_frames-1)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, self.img_size)
                if crop:
                    frame = crop_center(frame)  # Apply crop
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype('float32') / 255.0
                frames.append(frame)
        
        cap.release()
        
        while len(frames) < n_frames:
            frames.append(frames[-1] if frames else np.zeros((*self.img_size, 3)))
        
        return np.array(frames[:n_frames])

    def load_dataset(self, n_frames=16, max_videos=None, crop=False):
        """Load all videos from flat directory structure"""
        videos_data = []
        labels = []
        video_files_list = []
        
        video_files = [f for f in os.listdir(self.video_dir) if f.endswith('.mp4')]
        print(f"Found {len(video_files)} video files")
        
        classes = set()
        for video_file in video_files:
            class_label = self.extract_class_from_filename(video_file)
            if class_label:
                classes.add(class_label)
        
        print(f"Found {len(classes)} unique classes: {sorted(classes)}")
        
        if max_videos:
            video_files = video_files[:max_videos]
            print(f"Processing first {max_videos} videos")
        
        for video_file in tqdm(video_files, desc="Loading videos"):
            class_label = self.extract_class_from_filename(video_file)
            if class_label:
                video_path = os.path.join(self.video_dir, video_file)
                frames = self.extract_frames(video_path, n_frames, crop)
                
                if frames is not None:
                    videos_data.append(frames)
                    labels.append(class_label)
                    video_files_list.append(video_file)
                    
                    # Save frames if save_path is defined
                    if self.save_path:
                        video_dir = os.path.join(self.save_path, class_label)
                        os.makedirs(video_dir, exist_ok=True)
                        for i, frame in enumerate(frames):
                            frame_path = os.path.join(video_dir, f"{video_file}_frame_{i}.png")
                            cv2.imwrite(frame_path, (frame * 255).astype(np.uint8))
        
        return np.array(videos_data), np.array(labels), video_files_list

# Example usage

LSA_PATH = "Path to Dataset"
N_FRAMES = 16
CROP_SIZE = (160, 160)  # Define the crop size you want

loader = LSAVideoLoader(LSA_PATH, save_path='/dati/home/irfan.qutab/preprocessed_frames/')
videos, labels, video_files = loader.load_dataset(n_frames=N_FRAMES, max_videos=None, crop=True)

print(f"\n✓ Loaded {len(videos)} videos")
print(f"✓ Video shape: {videos[0].shape}")

