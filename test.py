import numpy as np
import os
import random
from PIL import Image

# ==================== Activation Functions ====================

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    """Derivative of sigmoid"""
    return a * (1 - a)

def relu(z):
    """ReLU activation function"""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU"""
    return (z > 0).astype(float)

def softmax(z):
    """Softmax activation function"""
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def dropout(A, keep_prob, training=True):
    """Dropout regularization"""
    if not training or keep_prob == 1.0:
        return A, None
    D = np.random.rand(*A.shape) < keep_prob
    A = A * D / keep_prob  # Inverted dropout
    return A, D

# ==================== Model Initialization ====================

def initialize_mlp_classification_model(layer_dims):
    """
    Khởi tạo model MLP với tham số ngẫu nhiên
    """
    np.random.seed(42)
    model = {}
    num_layers = len(layer_dims)
    
    for l in range(1, num_layers):
        # He initialization cho ReLU
        model[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        model[f'b{l}'] = np.zeros((layer_dims[l], 1))
    
    model['num_layers'] = num_layers - 1
    return model

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
    
    pcb_files = [os.path.join(pcb_dir, f) for f in os.listdir(pcb_dir) if f.endswith(('.jpg', '.png'))]
    not_pcb_files = [os.path.join(not_pcb_dir, f) for f in os.listdir(not_pcb_dir) if f.endswith(('.jpg', '.png'))]
    
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
        
    return dataset, norm_stats

# ==================== Forward Propagation ====================

def forward_propagation(model, X, keep_prob=1.0, training=True):
    cache = {'A0': X}
    A = X
    num_layers = model['num_layers']
    
    for l in range(1, num_layers):
        W = model[f'W{l}']
        b = model[f'b{l}']
        Z = np.dot(W, A) + b
        A = relu(Z)
        A, D = dropout(A, keep_prob, training)
        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A
        cache[f'D{l}'] = D
        
    W_last = model[f'W{num_layers}']
    b_last = model[f'b{num_layers}']
    Z = np.dot(W_last, A) + b_last
    A = softmax(Z)
    cache[f'Z{num_layers}'] = Z
    cache[f'A{num_layers}'] = A
    
    return A, cache

# ==================== Loss Function ====================

def cross_entropy_loss(output, label, model=None, lambda_reg=0.0):
    batch_size = label.shape[1]
    epsilon = 1e-15
    output_clipped = np.clip(output, epsilon, 1 - epsilon)
    ce_loss = -np.sum(label * np.log(output_clipped)) / batch_size
    
    l2_loss = 0
    if model is not None and lambda_reg > 0:
        for l in range(1, model['num_layers'] + 1):
            l2_loss += np.sum(model[f'W{l}'] ** 2)
        l2_loss = (lambda_reg / 2) * l2_loss
        
    return ce_loss + l2_loss

# ==================== Backward Propagation ====================

def backward_propagation(model, cache, X, label, keep_prob=1.0):
    gradients = {}
    num_layers = model['num_layers']
    batch_size = X.shape[1]
    
    dZ = cache[f'A{num_layers}'] - label
    
    for l in range(num_layers, 0, -1):
        A_prev = cache[f'A{l-1}']
        gradients[f'dW{l}'] = np.dot(dZ, A_prev.T) / batch_size
        gradients[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / batch_size
        
        if l > 1:
            dA_prev = np.dot(model[f'W{l}'].T, dZ)
            if cache.get(f'D{l-1}') is not None:
                dA_prev = dA_prev * cache[f'D{l-1}'] / keep_prob
            dZ = dA_prev * relu_derivative(cache[f'Z{l-1}'])
            
    return gradients

def calculate_gradient(model, cache, X, label, lambda_reg=0.0, keep_prob=1.0):
    gradients = backward_propagation(model, cache, X, label, keep_prob)
    if lambda_reg > 0:
        for l in range(1, model['num_layers'] + 1):
            gradients[f'dW{l}'] += lambda_reg * model[f'W{l}']
    return gradients

def update_parameters(model, gradients, learning_rate):
    for l in range(1, model['num_layers'] + 1):
        model[f'W{l}'] -= learning_rate * gradients[f'dW{l}']
        model[f'b{l}'] -= learning_rate * gradients[f'db{l}']
    return model

# ==================== Training ====================

def train(model, train_dataset, epochs, config, val_dataset=None, log_file='training_history.csv'):
    learning_rate = config['learning_rate']
    lambda_reg = config.get('lambda_reg', 0.01)
    patience = config.get('patience', 10)
    keep_prob = config.get('keep_prob', 0.7)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None
    
    if log_file:
        with open(log_file, 'w') as f:
            f.write('epoch,train_loss,val_loss,val_accuracy\n')
            
    for epoch in range(epochs):
        total_loss = 0
        for batch_images, batch_labels in train_dataset:
            output, cache = forward_propagation(model, batch_images, keep_prob, training=True)
            # Tính loss thuần (cross-entropy) để hiển thị, không cộng L2
            batch_loss = cross_entropy_loss(output, batch_labels)
            total_loss += batch_loss
            gradients = calculate_gradient(model, cache, batch_images, batch_labels, lambda_reg, keep_prob)
            model = update_parameters(model, gradients, learning_rate)
            
        avg_loss = total_loss / len(train_dataset)
        
        if val_dataset is not None:
            val_metrics = evaluation(model, val_dataset, print_metrics=False)
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
            
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"{epoch+1},{avg_loss:.6f},{val_loss:.6f},{val_acc:.4f}\n")
                    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in model.items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
    return best_model if best_model else model

# ==================== Evaluation ====================

def evaluation(model, dataset, print_metrics=True):
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_images, batch_labels in dataset:
        output, _ = forward_propagation(model, batch_images, keep_prob=1.0, training=False)
        total_loss += cross_entropy_loss(output, batch_labels)
        all_preds.extend(np.argmax(output, axis=0))
        all_labels.extend(np.argmax(batch_labels, axis=0))
        
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    TP = np.sum((all_preds == 1) & (all_labels == 1))
    TN = np.sum((all_preds == 0) & (all_labels == 0))
    FP = np.sum((all_preds == 1) & (all_labels == 0))
    FN = np.sum((all_preds == 0) & (all_labels == 1))
    
    acc = (TP + TN) / len(all_labels) * 100 if len(all_labels) > 0 else 0
    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {'loss': total_loss / len(dataset) if len(dataset) > 0 else 0, 'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}
    
    if print_metrics:
        print(f"\nEvaluation Results:")
        print(f"Accuracy: {acc:.2f}% | Precision: {precision:.2f}% | Recall: {recall:.2f}% | F1: {f1:.2f}%")
        print(f"Confusion Matrix: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
        
    return metrics

# ==================== Main ====================

def main():
    image_size = 64
    # Input size: 64x64x3 (RGB) = 12288
    layer_dims = [image_size * image_size * 3, 512, 256, 128, 64, 2]
    
    model = initialize_mlp_classification_model(layer_dims)
    print(f"Model initialized with architecture: {layer_dims}")
    
    data_dir = "./data"
    # Lấy ít ảnh hơn để demo tốc độ nếu cần, hoặc lấy toàn bộ
    train_set, norm = load_dataset(data_dir, split='train', image_size=image_size, max_images=5000)
    val_set, _ = load_dataset(data_dir, split='val', norm_stats=norm, image_size=image_size, max_images=1000)
    test_set, _ = load_dataset(data_dir, split='test', norm_stats=norm, image_size=image_size, max_images=1000)
    
    config = {
        "learning_rate": 0.001,
        "lambda_reg": 0.01,
        "keep_prob": 0.7,
        "patience": 10
    }
    
    print("\nStarting training...")
    model = train(model, train_set, epochs=50, config=config, val_dataset=val_set)
    
    print("\nFinal Test Evaluation:")
    evaluation(model, test_set)

if __name__ == "__main__":
    main()