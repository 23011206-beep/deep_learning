"""
Vẽ đồ thị training history từ file CSV
"""
import csv
import matplotlib.pyplot as plt

def plot_training_history(csv_file='training_history.csv', output_image='training_plot.png'):
    """
    Đọc file CSV và vẽ đồ thị loss, accuracy
    
    Args:
        csv_file: đường dẫn file CSV chứa training history
        output_image: đường dẫn file ảnh output
    """
    epochs = []
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Đọc dữ liệu từ CSV
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))
            val_accuracies.append(float(row['val_accuracy']))
    
    # Tạo figure với 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)
    
    # Plot 2: Validation Accuracy
    ax2.plot(epochs, val_accuracies, 'g-^', label='Validation Accuracy', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    
    # Thêm thông tin
    best_epoch = epochs[val_losses.index(min(val_losses))]
    best_val_loss = min(val_losses)
    best_val_acc = val_accuracies[val_losses.index(min(val_losses))]
    
    fig.suptitle(f'Training History (Best: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}%)', 
                 fontsize=12, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_image}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    plot_training_history()
