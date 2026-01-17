# Deep Learning - PCB Classification

Dự án phân loại PCB (Printed Circuit Board) sử dụng Multi-Layer Perceptron (MLP) được xây dựng hoàn toàn bằng NumPy.

## 📋 Tổng quan

Chương trình này huấn luyện một mạng neural network để phân loại ảnh PCB thành 2 lớp:
- **PCB**: Ảnh chứa mạch in
- **Not PCB**: Ảnh không chứa mạch in

## 🏗️ Kiến trúc hệ thống

### Cấu trúc module (từ trừu tượng đến cụ thể)

```
+------------------------------------------------------------+
|                        train.py                            |
|                  (Orchestration Layer)                     |
+------------------------------------------------------------+
| - Điều phối toàn bộ quá trình training                     |
| - Quản lý vòng lặp training/validation                     |
| - Early stopping & model checkpoint                        |
| - Logging & evaluation                                     |
+------------------------------------------------------------+
                |                          |
                v                          v
+----------------------------+  +----------------------------+
|         model.py           |  |       data_utils.py        |
|       (Core Logic)         |  |       (Data Layer)         |
+----------------------------+  +----------------------------+
| - Activation functions     |  | - Data augmentation        |
| - Forward propagation      |  | - Đọc và load dataset      |
| - Backward propagation     |  | - Normalization            |
| - Loss calculation         |  | - Batching                 |
+----------------------------+  +----------------------------+
                |                          |
                v                          v
+------------------------------------------------------------+
|                      inference.py                          |
|                   (Application Layer)                      |
+------------------------------------------------------------+
| - Sử dụng model đã train để dự đoán                        |
| - Visualize kết quả                                        |
+------------------------------------------------------------+
```

---

## 📁 Chi tiết các file chính

### 1. **`model.py`** - Core Neural Network Logic
Định nghĩa toán học và logic của neural network

#### Chức năng chính:

##### 🔹 Activation Functions (Hàm kích hoạt)
```python
sigmoid(z)           # Sigmoid: 1/(1+e^-z)
relu(z)              # ReLU: max(0, z)
softmax(z)           # Softmax: e^zi / Σe^zj
dropout(A, keep_prob) # Dropout regularization
```

##### 🔹 Model Initialization
```python
initialize_mlp_classification_model(layer_dims)
# Khởi tạo weights và biases cho tất cả layers
# Sử dụng He initialization cho ReLU
```

##### 🔹 Forward Propagation
```python
forward_propagation(model, X, keep_prob, training)
# Tính toán output từ input qua tất cả layers
# Áp dụng dropout nếu đang training
# Cache các giá trị trung gian cho backward pass
```

##### 🔹 Loss Function
```python
cross_entropy_loss(output, label, model, lambda_reg)
# Cross-entropy loss cho classification
# Tùy chọn thêm L2 regularization
```

##### 🔹 Backward Propagation
```python
backward_propagation(model, cache, X, label, keep_prob)
# Tính gradient của loss theo từng parameter
# Backpropagate qua tất cả layers

calculate_gradient(model, cache, X, label, lambda_reg, keep_prob)
# Wrapper cho backward propagation
# Thêm L2 regularization gradient nếu cần

update_parameters(model, gradients, learning_rate)
# Cập nhật weights và biases theo gradient descent
```

**Input/Output**:
- Input: Raw data (numpy arrays), model parameters
- Output: Predictions, gradients, updated parameters

---

### 2. **`data_utils.py`** - Data Processing Layer
Xử lý và chuẩn bị dữ liệu cho training

#### Chức năng chính:

##### 🔹 Data Augmentation
```python
augment_image(img)
# 1. Random Hue Shift (80% probability)
#    - Chuyển RGB → HSV
#    - Dịch chuyển Hue channel
#    - Chuyển lại RGB
# 2. Random Rotation (0°, 90°, 180°, 270°)
# 3. Random Horizontal Flip (50% probability)
# 4. Random Vertical Flip (50% probability)
```

**Mục đích**: Tăng tính đa dạng của dữ liệu, giảm overfitting

##### 🔹 Dataset Loading
```python
load_dataset(data_dir, split, max_images, batch_size, norm_stats, image_size)
# 1. Đọc ảnh từ thư mục pcb/ và not_pcb/
# 2. Chia dataset: 80% train, 10% val, 10% test
# 3. Resize ảnh về image_size x image_size
# 4. Áp dụng augmentation (chỉ cho training set)
# 5. Normalize: (X - mean) / std
# 6. One-hot encoding cho labels
# 7. Tạo batches
```

**Input/Output**:
- Input: Đường dẫn thư mục, cấu hình
- Output: Batched dataset, normalization stats, file paths

---

### 3. **`train.py`** - Training Orchestration Layer
Điều phối toàn bộ quá trình training và evaluation

#### Chức năng chính:

##### 🔹 Training Loop
```python
train(model, train_dataset, epochs, config, val_dataset, log_file)
# 1. Khởi tạo early stopping parameters
# 2. Vòng lặp qua từng epoch:
#    a. Training phase:
#       - Forward pass với dropout
#       - Tính loss
#       - Backward pass
#       - Update parameters
#    b. Validation phase:
#       - Evaluate trên validation set
#       - Tính metrics (loss, accuracy)
#       - Early stopping check
#    c. Logging:
#       - Ghi metrics vào CSV
#       - In progress ra console
# 3. Trả về best model
```

**Hyperparameters**:
```python
config = {
    "learning_rate": 0.001,    # Tốc độ học
    "lambda_reg": 0.01,        # L2 regularization strength
    "keep_prob": 0.7,          # Dropout keep probability
    "patience": 10             # Early stopping patience
}
```

##### 🔹 Evaluation
```python
evaluation(model, dataset, print_metrics)
# 1. Forward pass trên toàn bộ dataset (no dropout)
# 2. Tính confusion matrix: TP, TN, FP, FN
# 3. Tính metrics:
#    - Accuracy: (TP + TN) / Total
#    - Precision: TP / (TP + FP)
#    - Recall: TP / (TP + FN)
#    - F1 Score: 2 * Precision * Recall / (Precision + Recall)
# 4. Trả về dictionary chứa tất cả metrics
```

##### 🔹 Main Function
```python
main()
# 1. Định nghĩa kiến trúc model
#    layer_dims = [12288, 512, 256, 128, 64, 2]
#    (Input: 64x64x3 RGB image flattened)
# 2. Load datasets (train, val, test)
# 3. Train model với early stopping
# 4. Lưu best model + normalization stats
# 5. Evaluate trên test set
```

**Input/Output**:
- Input: Configuration, datasets
- Output: Trained model (saved to `model_data.pkl`), training history CSV

---

### 4. **`inference.py`** - Application Layer
Sử dụng model đã train để dự đoán trên ảnh mới

#### Chức năng chính:

##### 🔹 Load Model
```python
# Load model_data.pkl chứa:
# - model: trained parameters
# - norm_stats: (mean, std) for normalization
# - image_size: kích thước ảnh input
# - test_files: danh sách file test
```

##### 🔹 Inference
```python
# 1. Chọn ngẫu nhiên N ảnh từ test set
# 2. Preprocess:
#    - Resize về image_size
#    - Flatten và normalize với norm_stats
# 3. Forward pass (no dropout)
# 4. Lấy prediction (argmax của output)
```

##### 🔹 Visualization
```python
# Hiển thị grid N ảnh với:
# - Ảnh gốc
# - Prediction (PCB / Not PCB)
# - Confidence score
# Lưu kết quả vào inference_results.png
```

**Input/Output**:
- Input: Trained model, test images
- Output: Predictions, visualization

---

## 🚀 Cách sử dụng

### 1. Chuẩn bị dữ liệu

Tạo cấu trúc thư mục:
```
data/
├── pcb/          # Ảnh PCB
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── not_pcb/      # Ảnh không phải PCB
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

### 2. Training

```bash
python train.py
```

**Output**:
- `model_data.pkl`: Model đã train
- `training_history.csv`: Lịch sử training

**Console output**:
```
Epoch 1/50 - Train Loss (dropout): 0.5724 - Train Loss (no dropout): 0.0650 - Train Acc: 98.50% - Val Loss: 0.0698 - Val Acc: 98.00%
Epoch 2/50 - Train Loss (dropout): 0.2647 - Train Loss (no dropout): 0.0620 - Train Acc: 98.70% - Val Loss: 0.0699 - Val Acc: 97.90%
...
```

### 3. Inference

```bash
python inference.py
```

**Output**:
- `inference_results.png`: Visualization của predictions

### 4. Visualize Training History (Optional)

```bash
python plot_history.py
```

**Output**:
- `training_plot.png`: Đồ thị loss và accuracy qua các epochs

---

## 🎯 Kiến trúc Model

```
Input Layer:    12288 neurons (64x64x3 RGB image flattened)
                  ↓
Hidden Layer 1:  512 neurons (ReLU + Dropout 0.7)
                  ↓
Hidden Layer 2:  256 neurons (ReLU + Dropout 0.7)
                  ↓
Hidden Layer 3:  128 neurons (ReLU + Dropout 0.7)
                  ↓
Hidden Layer 4:   64 neurons (ReLU + Dropout 0.7)
                  ↓
Output Layer:      2 neurons (Softmax)
                  ↓
              [PCB, Not PCB]
```

**Regularization techniques**:
- ✅ Dropout (keep_prob = 0.7)
- ✅ L2 Regularization (lambda = 0.01)
- ✅ Data Augmentation
- ✅ Early Stopping (patience = 10)

---

## 📊 Hiểu về Metrics

### Train Loss vs Val Loss

**Train Loss (dropout)**: Loss khi training với dropout
- Cao hơn vì 30% neurons bị tắt ngẫu nhiên
- Model phải học với ít thông tin hơn

**Train Loss (no dropout)**: Loss thực sự của model trên training set
- Tính toán với full model (không dropout)
- **So sánh metric này với Val Loss để phát hiện overfitting**

**Val Loss**: Loss trên validation set
- Nếu Val Loss >> Train Loss (no dropout) → Overfitting
- Nếu Val Loss ≈ Train Loss (no dropout) → Good generalization ✅

---

## 🔧 Dependencies

```
numpy
Pillow (PIL)
pickle (built-in)
```

---

## 📝 Notes

### Tại sao Train Loss cao hơn Val Loss?

Đây là hiện tượng **bình thường** khi sử dụng:
1. **Dropout**: Model yếu hơn khi training (30% neurons tắt)
2. **Data Augmentation**: Training data khó hơn (xoay, lật, đổi màu)

→ Validation sử dụng full model + ảnh gốc → Loss thấp hơn

### Best Practices

1. **Luôn theo dõi cả Train Loss (no dropout) và Val Loss**
2. **Early stopping** dựa trên Val Loss, không phải Train Loss
3. **Data augmentation** chỉ áp dụng cho training set
4. **Normalization stats** phải được tính từ training set và áp dụng cho tất cả splits

---

## 🎓 Learning Resources

File này được xây dựng để học deep learning từ scratch:
- **model.py**: Hiểu cách neural network hoạt động ở mức toán học
- **data_utils.py**: Hiểu data preprocessing và augmentation
- **train.py**: Hiểu training loop và optimization
- **inference.py**: Hiểu cách deploy model

---

## 📧 Contact

Nếu có câu hỏi hoặc gặp vấn đề, vui lòng tạo issue hoặc liên hệ.