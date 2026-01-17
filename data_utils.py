import numpy as np
import os
import random
from PIL import Image

# ==================== Data Augmentation ====================

def augment_image(img):
    """
    Tăng cường dữ liệu: xoay, lật và dịch chuyển màu sắc (Hue shift)
    """
    # 1. Random Hue Shift (Ánh xạ màu xanh sang các màu khác)
    # Chuyển sang HSV để thay đổi sắc độ mà không làm mất chi tiết/độ sáng
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

# ==================== Dataset Loading ====================

def load_dataset(data_dir, split='train', max_images=None, batch_size=32, norm_stats=None, image_size=64):
    """
    Tải dataset từ cấu trúc thư mục data/pcb và data/not_pcb
    """
    pcb_dir = os.path.join(data_dir, "pcb")
    not_pcb_dir = os.path.join(data_dir, "not_pcb")
    
    if not os.path.exists(pcb_dir) or not os.path.exists(not_pcb_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục pcb hoặc not_pcb trong {data_dir}")
    
    pcb_files = [os.path.join(pcb_dir, f) for f in sorted(os.listdir(pcb_dir)) if f.endswith(('.jpg', '.png'))]
    not_pcb_files = [os.path.join(not_pcb_dir, f) for f in sorted(os.listdir(not_pcb_dir)) if f.endswith(('.jpg', '.png'))]
    
    # Tạo nhãn: pcb = 1, not_pcb = 0
    all_files = [(f, 1) for f in pcb_files] + [(f, 0) for f in not_pcb_files]
    
    # Shuffle ngẫu nhiên
    random.seed(42)
    random.shuffle(all_files)
    
    # Chia tập dữ liệu (80% train, 10% val, 10% test)
    n = len(all_files)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    if split == 'train':
        selected_files = all_files[:train_end]
    elif split == 'val':
        selected_files = all_files[train_end:val_end]
    else:
        selected_files = all_files[val_end:]
        
    if max_images is not None:
        selected_files = selected_files[:max_images]
        
    print(f"[{split}] Processing {len(selected_files)} images...")
    
    X_list = []
    y_list = []
    
    for img_path, label in selected_files:
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((image_size, image_size))
            
            # Chỉ áp dụng augmentation cho tập huấn luyện
            if split == 'train':
                img = augment_image(img)
                
            img_array = np.array(img).flatten() / 255.0
            X_list.append(img_array)
            y_list.append(label)
        except Exception as e:
            print(f"Bỏ qua ảnh lỗi {img_path}: {e}")
            continue
            
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Chuẩn hóa
    if norm_stats is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        norm_stats = (mean, std)
    else:
        mean, std = norm_stats
        
    X = (X - mean) / std
    
    # One-hot encoding
    num_classes = 2
    num_samples = len(y)
    y_one_hot = np.zeros((num_samples, num_classes))
    y_one_hot[np.arange(num_samples), y] = 1
    
    # Batches
    dataset = []
    for i in range(0, num_samples, batch_size):
        batch_X = X[i:i+batch_size].T
        batch_y = y_one_hot[i:i+batch_size].T
        dataset.append((batch_X, batch_y))
        
    return dataset, norm_stats, selected_files
