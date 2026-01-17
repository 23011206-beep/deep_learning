import numpy as np
import os
import pickle
from model import (
    initialize_mlp_classification_model,
    forward_propagation,
    cross_entropy_loss,
    calculate_gradient,
    update_parameters
)
from data_utils import load_dataset

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
            # Tính train loss không dropout để so sánh công bằng
            train_metrics = evaluation(model, train_dataset, print_metrics=False)
            train_loss_no_dropout = train_metrics['loss']
            train_acc = train_metrics['accuracy']
            
            val_metrics = evaluation(model, val_dataset, print_metrics=False)
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
            print(f"Epoch {epoch+1}/{epochs} - Train Loss (dropout): {avg_loss:.4f} - Train Loss (no dropout): {train_loss_no_dropout:.4f} - Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
            
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
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    # Lấy ít ảnh hơn để demo tốc độ nếu cần, hoặc lấy toàn bộ
    train_set, norm, _ = load_dataset(data_dir, split='train', image_size=image_size, max_images=5000)
    val_set, _, _ = load_dataset(data_dir, split='val', norm_stats=norm, image_size=image_size, max_images=1000)
    test_set, _, test_files = load_dataset(data_dir, split='test', norm_stats=norm, image_size=image_size, max_images=1000)
    
    config = {
        "learning_rate": 0.001,
        "lambda_reg": 0.01,
        "keep_prob": 0.7,
        "patience": 10
    }
    
    print("\nStarting training...")
    best_model = train(model, train_set, epochs=50, config=config, val_dataset=val_set)
    
    # Lưu model tốt nhất
    model_data = {
        'model': best_model,
        'norm_stats': norm,
        'image_size': image_size,
        'test_files': test_files
    }
    
    with open('model_data.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\nSaved best model to model_data.pkl")
    
    print("\nFinal Test Evaluation (Best Model):")
    evaluation(best_model, test_set)

if __name__ == "__main__":
    main()
