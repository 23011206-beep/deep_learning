import numpy as np
import pickle
import random
import os
import matplotlib.pyplot as plt
from PIL import Image
from test import forward_propagation  # Import functions from training script

def augment_image(img):
    """
    Tăng cường dữ liệu: xoay, lật và dịch chuyển màu sắc (Hue shift)
    """
    # 1. Random Hue Shift (Ánh xạ màu xanh sang các màu khác)
    if random.random() > 0.2: # 80% cơ hội đổi màu
        img_hsv = img.convert('HSV')
        h, s, v = img_hsv.split()
        h_array = np.array(h, dtype=np.int32)
        shift = random.randint(0, 255)
        h_array = (h_array + shift) % 256
        h = Image.fromarray(h_array.astype(np.uint8))
        img = Image.merge('HSV', (h, s, v)).convert('RGB')
    
    # 2. Random Rotation
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        img = img.rotate(angle)
        
    # 3. Random Flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
    return img

def load_and_preprocess_image(img_path, image_size, norm_stats, augment=False):
    """
    Load an image, resize it, flatten it and normalize it using training stats.
    """
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize((image_size, image_size))
    
    if augment:
        img_resized = augment_image(img_resized)
    
    # Flatten and normalize
    X = np.array(img_resized).flatten() / 255.0
    mean, std = norm_stats
    X = (X - mean) / std
    
    # Reshape to (input_dim, 1) for forward prop
    X = X.reshape(-1, 1)
    return img_resized, X

def main():
    # 1. Load model data
    model_file = 'model_data.pkl'
    if not os.path.exists(model_file):
        print(f"Error: {model_file} not found. Please run test.py first to train the model.")
        return

    with open(model_file, 'rb') as f:
        data = pickle.load(f)
    
    model = data['model']
    norm_stats = data['norm_stats']
    image_size = data['image_size']
    test_files = data['test_files']
    
    if not test_files:
        print("Error: No test files found in model_data.pkl.")
        return

    # 2. Select 10 random images from test set
    num_samples = min(10, len(test_files))
    selected_samples = random.sample(test_files, num_samples)
    print(f"Selected {num_samples} images for inference.")

    # 3. Preprocess, Predict, and Visualize
    rows = 2
    cols = (num_samples + 1) // rows
    plt.figure(figsize=(20, 10))

    for i, (img_path, true_label) in enumerate(selected_samples):
        # Preprocess - Apply augmentation if it's a PCB image (label 1)
        is_pcb = (true_label == 1)
        original_img, X = load_and_preprocess_image(img_path, image_size, norm_stats, augment=is_pcb)
        
        # Forward propagation
        output, _ = forward_propagation(model, X, training=False)
        prediction = np.argmax(output, axis=0)[0]
        confidence = output[prediction, 0] * 100

        pred_label = 'PCB' if prediction == 1 else 'NOT_PCB'
        true_label_str = 'PCB' if true_label == 1 else 'NOT_PCB'
        
        # Plotting
        plt.subplot(rows, cols, i + 1)
        plt.imshow(original_img)
        color = 'green' if prediction == true_label else 'red'
        plt.title(f"T: {true_label_str} | P: {pred_label}\n({confidence:.1f}%)", color=color, fontsize=10)
        plt.axis('off')
        
        print(f"[{i+1}/{num_samples}] {img_path} -> Pred: {pred_label} ({confidence:.2f}%)")

    plt.tight_layout()
    result_path = 'inference_results.png'
    plt.savefig(result_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"\nResults saved to {result_path}")
    print(f"Direct link to file: file:///{os.path.abspath(result_path)}")

if __name__ == "__main__":
    main()
