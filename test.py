import numpy as np

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

# ==================== Model Initialization ====================

def initialize_mlp_classification_model(layer_dims):
    """
    Khởi tạo model MLP với tham số ngẫu nhiên
    
    Args:
        layer_dims: list chứa số neurons mỗi layer, ví dụ [784, 128, 64, 10]
                   - 784: input (28x28 pixels)
                   - 128, 64: hidden layers
                   - 10: output (10 classes)
    
    Returns:
        model: dictionary chứa weights và biases
    """
    np.random.seed(42)
    model = {}
    num_layers = len(layer_dims)
    
    for l in range(1, num_layers):
        # He initialization cho ReLU
        model[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        model[f'b{l}'] = np.zeros((layer_dims[l], 1))
    
    model['num_layers'] = num_layers - 1  # Số layer (không tính input)
    return model

# ==================== Dataset Loading ====================

def load_dataset(data_dir, split='train', max_images=None, batch_size=32, norm_stats=None):
    """
    Tải dataset từ thư mục data (được tạo bởi preprocess_deeppcb.py)
    Sử dụng file CSV để phân chia train/val/test
    
    Args:
        data_dir: đường dẫn tới thư mục chứa data
        split: 'train', 'val', hoặc 'test' - quyết định file CSV nào được sử dụng
        max_images: số ảnh tối đa để tải (chia đều cho 2 class). Nếu None thì tải tất cả.
        batch_size: kích thước mỗi batch
        norm_stats: tuple (mean, std) từ training set. Nếu None sẽ tự tính.
    
    Returns:
        dataset: list of tuples (batch_images, batch_labels)
        norm_stats: tuple (mean, std) để dùng cho test set
    """
    import os
    from PIL import Image
    
    # Đọc file CSV tương ứng
    csv_path = os.path.join(data_dir, f'{split}.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file {csv_path}")
    
    # Parse CSV file
    file_list = []  # [(filepath, label), ...]
    defect_files = []
    normal_files = []
    
    with open(csv_path, 'r') as f:
        lines = f.readlines()[1:]  # Bỏ header
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 2:
                    filepath = os.path.join(data_dir, parts[0])
                    label = int(parts[1])
                    if label == 1:
                        defect_files.append((filepath, label))
                    else:
                        normal_files.append((filepath, label))
    
    # Shuffle để lấy ngẫu nhiên
    np.random.seed(42)
    np.random.shuffle(defect_files)
    np.random.shuffle(normal_files)
    
    # Tính số ảnh mỗi class nếu có max_images
    if max_images is not None:
        images_per_class = max_images // 2
        # Đảm bảo không vượt quá số ảnh có sẵn
        num_defect = min(images_per_class, len(defect_files))
        num_normal = min(images_per_class, len(normal_files))
        # Cân bằng 2 class
        num_per_class = min(num_defect, num_normal)
        defect_files = defect_files[:num_per_class]
        normal_files = normal_files[:num_per_class]
    
    file_list = defect_files + normal_files
    print(f"[{split}] Loading {len(defect_files)} defect + {len(normal_files)} normal = {len(file_list)} images...")
    
    # Load và tiền xử lý ảnh
    X_list = []
    y_list = []
    
    for img_path, label in file_list:
        img = Image.open(img_path).convert('L')  # Chuyển sang grayscale
        img_array = np.array(img).flatten() / 255.0  # Normalize to [0, 1]
        X_list.append(img_array)
        y_list.append(label)
    
    # Chuyển sang numpy array
    X = np.array(X_list)  # Shape: (num_samples, image_size * image_size)
    y = np.array(y_list)  # Shape: (num_samples,)
    
    # Shuffle cả dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Chuẩn hóa dữ liệu (standardization)
    # Dùng chung mean/std từ training set để tránh data leakage
    if norm_stats is None:
        mean = X.mean()
        std = X.std() + 1e-8
        norm_stats = (mean, std)
    else:
        mean, std = norm_stats
    
    X = (X - mean) / std
    
    # Chuyển labels sang one-hot encoding (2 classes: normal=0, defect=1)
    num_classes = 2
    num_samples = len(y)
    y_one_hot = np.zeros((num_samples, num_classes))
    y_one_hot[np.arange(num_samples), y] = 1
    
    # Chia thành batches
    dataset = []
    for i in range(0, num_samples, batch_size):
        batch_X = X[i:i+batch_size].T  # Shape: (input_size, batch_size)
        batch_y = y_one_hot[i:i+batch_size].T  # Shape: (num_classes, batch_size)
        dataset.append((batch_X, batch_y))
    
    print(f"[{split}] Created {len(dataset)} batches with batch_size={batch_size}")
    return dataset, norm_stats

# ==================== Forward Propagation ====================

def forward_propagation(model, X):
    """
    Thực hiện forward propagation qua tất cả các layers
    
    Args:
        model: dictionary chứa parameters
        X: input data, shape (n_features, batch_size)
    
    Returns:
        output: predictions
        cache: dictionary chứa các giá trị trung gian cho backprop
    """
    cache = {'A0': X}
    A = X
    num_layers = model['num_layers']
    
    # Hidden layers với ReLU
    for l in range(1, num_layers):
        Z = np.dot(model[f'W{l}'], A) + model[f'b{l}']
        A = relu(Z)
        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A
    
    # Output layer với Softmax
    Z = np.dot(model[f'W{num_layers}'], A) + model[f'b{num_layers}']
    A = softmax(Z)
    cache[f'Z{num_layers}'] = Z
    cache[f'A{num_layers}'] = A
    
    return A, cache

# ==================== Loss Function ====================

def cross_entropy_loss(output, label, model=None, lambda_reg=0.0):
    """
    Tính toán cross-entropy loss với L2 regularization
    
    Args:
        output: predictions từ model, shape (num_classes, batch_size)
        label: one-hot encoded labels, shape (num_classes, batch_size)
        model: model dictionary (để tính L2 regularization)
        lambda_reg: hệ số regularization (0 = không regularization)
    
    Returns:
        loss: scalar cross-entropy loss + L2 regularization
    """
    batch_size = label.shape[1]
    # Thêm epsilon để tránh log(0)
    epsilon = 1e-15
    output_clipped = np.clip(output, epsilon, 1 - epsilon)
    ce_loss = -np.sum(label * np.log(output_clipped)) / batch_size
    
    # L2 Regularization
    l2_loss = 0
    if model is not None and lambda_reg > 0:
        num_layers = model['num_layers']
        for l in range(1, num_layers + 1):
            l2_loss += np.sum(model[f'W{l}'] ** 2)
        l2_loss = (lambda_reg / 2) * l2_loss
    
    return ce_loss + l2_loss

# ==================== Backward Propagation ====================

def backward_propagation(model, cache, X, label):
    """
    Tính toán gradients của model bằng backpropagation
    
    Args:
        model: dictionary chứa parameters
        cache: dictionary chứa các giá trị từ forward pass
        X: input data
        label: true labels
    
    Returns:
        gradients: dictionary chứa gradients cho tất cả parameters
    """
    gradients = {}
    num_layers = model['num_layers']
    batch_size = X.shape[1]
    
    # Output layer gradient (softmax + cross-entropy)
    dZ = cache[f'A{num_layers}'] - label  # Shape: (num_classes, batch_size)
    
    for l in range(num_layers, 0, -1):
        A_prev = cache[f'A{l-1}']
        
        gradients[f'dW{l}'] = np.dot(dZ, A_prev.T) / batch_size
        gradients[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / batch_size
        
        if l > 1:
            dA_prev = np.dot(model[f'W{l}'].T, dZ)
            dZ = dA_prev * relu_derivative(cache[f'Z{l-1}'])
    
    return gradients

def calculate_gradient(model, cache, X, label, lambda_reg=0.0):
    """
    Wrapper function để tính gradient với L2 regularization
    """
    gradients = backward_propagation(model, cache, X, label)
    
    # Thêm L2 regularization vào gradient của W
    if lambda_reg > 0:
        num_layers = model['num_layers']
        for l in range(1, num_layers + 1):
            gradients[f'dW{l}'] += lambda_reg * model[f'W{l}']
    
    return gradients

# ==================== Update Parameters ====================

def update_parameters(model, gradients, learning_rate):
    """
    Cập nhật tham số của model sử dụng gradient descent
    
    Args:
        model: dictionary chứa parameters
        gradients: dictionary chứa gradients
        learning_rate: tốc độ học
    
    Returns:
        model: model đã được cập nhật
    """
    num_layers = model['num_layers']
    
    for l in range(1, num_layers + 1):
        model[f'W{l}'] -= learning_rate * gradients[f'dW{l}']
        model[f'b{l}'] -= learning_rate * gradients[f'db{l}']
    
    return model

# ==================== Training ====================

def train(model, train_dataset, epochs, config, val_dataset=None, log_file='training_history.csv'):
    """
    Huấn luyện model với L2 regularization và early stopping
    
    Args:
        model: MLP model
        train_dataset: training data
        epochs: số epochs
        config: dictionary chứa hyperparameters
        val_dataset: validation data (cho early stopping)
        log_file: đường dẫn file CSV để lưu history (None = không lưu)
    
    Returns:
        model: model đã được huấn luyện
    """
    learning_rate = config['learning_rate']
    lambda_reg = config.get('lambda_reg', 0.01)  # L2 regularization
    patience = config.get('patience', 5)  # Early stopping patience
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None
    
    # Khởi tạo file CSV để lưu history
    history = []
    if log_file:
        with open(log_file, 'w') as f:
            f.write('epoch,train_loss,val_loss,val_accuracy\n')
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_images, batch_labels in train_dataset:
            # Forward propagation
            output, cache = forward_propagation(model, batch_images)
            
            # Calculate loss với L2 regularization
            batch_loss = cross_entropy_loss(output, batch_labels, model, lambda_reg)
            total_loss += batch_loss
            
            # Backward propagation với L2 regularization
            gradients = calculate_gradient(model, cache, batch_images, batch_labels, lambda_reg)
            
            # Update parameters
            model = update_parameters(model, gradients, learning_rate)
        
        avg_loss = total_loss / len(train_dataset)
        
        # Validation và Early stopping
        if val_dataset is not None:
            val_metrics = evaluation(model, val_dataset, print_metrics=False)
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
            
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Lưu vào history
            history.append({'epoch': epoch + 1, 'train_loss': avg_loss, 'val_loss': val_loss, 'val_accuracy': val_acc})
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f'{epoch + 1},{avg_loss:.6f},{val_loss:.6f},{val_acc:.4f}\n')
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Lưu best model (deep copy)
                best_model = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in model.items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}! Best val loss: {best_val_loss:.4f}")
                    print(f"Training history saved to: {log_file}")
                    return best_model if best_model else model
        else:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            # Lưu vào history (không có validation)
            history.append({'epoch': epoch + 1, 'train_loss': avg_loss, 'val_loss': None, 'val_accuracy': None})
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f'{epoch + 1},{avg_loss:.6f},,\n')
    
    if log_file:
        print(f"Training history saved to: {log_file}")
    
    return best_model if best_model else model

# ==================== Evaluation ====================

def evaluation(model, dataset, print_metrics=True):
    """
    Đánh giá model trên dataset với đầy đủ metrics
    
    Returns:
        metrics: dictionary chứa tất cả metrics
    """
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch_images, batch_labels in dataset:
        output, _ = forward_propagation(model, batch_images)
        batch_loss = cross_entropy_loss(output, batch_labels)
        total_loss += batch_loss
        
        # Thu thập predictions và labels
        predictions = np.argmax(output, axis=0)
        labels = np.argmax(batch_labels, axis=0)
        all_predictions.extend(predictions)
        all_labels.extend(labels)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Tính Confusion Matrix
    # Positive class = 1 (defect), Negative class = 0 (normal)
    TP = np.sum((all_predictions == 1) & (all_labels == 1))  # True Positive
    TN = np.sum((all_predictions == 0) & (all_labels == 0))  # True Negative
    FP = np.sum((all_predictions == 1) & (all_labels == 0))  # False Positive
    FN = np.sum((all_predictions == 0) & (all_labels == 1))  # False Negative
    
    # Tính các metrics
    total = len(all_labels)
    accuracy = (TP + TN) / total * 100 if total > 0 else 0
    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = TN / (TN + FP) * 100 if (TN + FP) > 0 else 0
    
    avg_loss = total_loss / len(dataset)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'confusion_matrix': {
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
        }
    }
    
    if print_metrics:
        print(f"\n{'-'*40}")
        print(f"  Loss:        {avg_loss:.4f}")
        print(f"  Accuracy:    {accuracy:.2f}%")
        print(f"  Precision:   {precision:.2f}%")
        print(f"  Recall:      {recall:.2f}%")
        print(f"  F1-Score:    {f1_score:.2f}%")
        print(f"  Specificity: {specificity:.2f}%")
        print(f"\n  Confusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Normal  Defect")
        print(f"  Actual Normal   {TN:5d}   {FP:5d}")
        print(f"  Actual Defect   {FN:5d}   {TP:5d}")
        print(f"{'-'*40}")
    
    return metrics

# ==================== Main ====================

def main():
    # Định nghĩa kiến trúc model: input -> hidden layers -> output
    # Input: 64x64 = 4096, Output: 2 classes (normal, defect)
    # Giảm kích thước model để tránh overfitting
    layer_dims = [4096, 128, 32, 2]  # 4096 input, 2 hidden layers, 2 output classes
    
    # Khởi tạo model
    model = initialize_mlp_classification_model(layer_dims)
    print(f"Model initialized with architecture: {layer_dims}")
    print(f"Total parameters: {sum(model[f'W{l}'].size + model[f'b{l}'].size for l in range(1, model['num_layers']+1))}")
    
    # Load datasets
    # Dùng chung norm_stats từ training set cho test set
    data_dir = "./data"
    train_dataset, norm_stats = load_dataset(data_dir=data_dir, split='train', max_images=10000, batch_size=32)
    val_dataset, _ = load_dataset(data_dir=data_dir, split='val', max_images=1000, batch_size=32, norm_stats=norm_stats)
    test_dataset, _ = load_dataset(data_dir=data_dir, split='test', max_images=None, batch_size=32, norm_stats=norm_stats)
    print(f"\nTraining samples: {len(train_dataset)} batches")
    print(f"Validation samples: {len(val_dataset)} batches")
    print(f"Test samples: {len(test_dataset)} batches")
    
    # Config với L2 regularization và early stopping
    config = {
        "learning_rate": 0.01,
        "lambda_reg": 0.001,  # L2 regularization strength
        "patience": 5  # Early stopping patience
    }
    
    # Training với validation set
    print(f"\n{'='*50}")
    print("Starting training with L2 regularization and early stopping...")
    print(f"{'='*50}\n")
    model = train(model, train_dataset, epochs=50, config=config, val_dataset=val_dataset)
    
    # Evaluation
    print(f"\n{'='*50}")
    print("Training Set Evaluation:")
    print(f"{'='*50}")
    train_metrics = evaluation(model, train_dataset)
    
    print(f"\n{'='*50}")
    print("Test Set Evaluation:")
    print(f"{'='*50}")
    test_metrics = evaluation(model, test_dataset)

if __name__ == "__main__":
    main()