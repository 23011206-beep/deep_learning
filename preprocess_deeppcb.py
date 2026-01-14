"""
Tiền xử lý DeepPCB Dataset cho Binary Classification

Quy tắc:
- Patch 64x64, stride 32 (50% overlap)
- Nếu patch chứa TOÀN BỘ bounding box → defect (1)
- Nếu patch chứa MỘT PHẦN bounding box → cắt phần lỗi đi, resize lại 64x64 → normal (0)
- Nếu patch KHÔNG chứa bounding box nào → normal (0)

Output: Thư mục /defect/ và /normal/ để tiện training
"""

import numpy as np
import os
import sys
from pathlib import Path
import shutil

# Fix encoding cho Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Thử import PIL, nếu không có thì báo lỗi
try:
    from PIL import Image
except ImportError:
    print("Cần cài đặt Pillow: pip install Pillow")
    exit(1)

# ==================== CONFIGURATION ====================

# TODO: Điền đường dẫn dataset của bạn
INPUT_IMAGE_DIR = r"C:\Users\Admin\.cache\kagglehub\datasets\liuxiaolong1\pcb-defect-detection-dataset\versions\1\DeepPCB\train\images"      # Thư mục chứa ảnh gốc
INPUT_LABEL_DIR = r"C:\Users\Admin\.cache\kagglehub\datasets\liuxiaolong1\pcb-defect-detection-dataset\versions\1\DeepPCB\train\labels"      # Thư mục chứa file annotation
OUTPUT_DIR = r".\data"                    # Thư mục output

PATCH_SIZE = 160
STRIDE = 80

FINAL_IMAGE_SIZE = 64

# ==================== HELPER FUNCTIONS ====================

def parse_annotation_file(annotation_path):
    """
    Đọc file annotation và trả về list các bounding boxes
    Format: x_min, y_min, x_max, y_max, label
    
    Returns:
        list of tuples: [(x_min, y_min, x_max, y_max, label), ...]
    """
    bboxes = []
    with open(annotation_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    x_min, y_min, x_max, y_max, label = map(int, parts[:5])
                    bboxes.append((x_min, y_min, x_max, y_max, label))
    return bboxes


def check_bbox_in_patch(patch_x, patch_y, patch_size, bbox):
    """
    Kiểm tra quan hệ giữa patch và bounding box
    
    Args:
        patch_x, patch_y: tọa độ góc trên bên trái của patch
        patch_size: kích thước patch
        bbox: (x_min, y_min, x_max, y_max, label)
    
    Returns:
        'full': patch chứa toàn bộ bbox
        'partial': patch chứa một phần bbox
        'none': không giao nhau
    """
    px_min, py_min = patch_x, patch_y
    px_max, py_max = patch_x + patch_size, patch_y + patch_size
    
    bx_min, by_min, bx_max, by_max, _ = bbox
    
    # Kiểm tra có giao nhau không
    if px_max <= bx_min or px_min >= bx_max or py_max <= by_min or py_min >= by_max:
        return 'none'
    
    # Kiểm tra patch có chứa toàn bộ bbox không
    if px_min <= bx_min and py_min <= by_min and px_max >= bx_max and py_max >= by_max:
        return 'full'
    
    return 'partial'


def get_overlap_region(patch_x, patch_y, patch_size, bbox):
    """
    Tính vùng giao nhau giữa patch và bbox (trong tọa độ của patch)
    
    Returns:
        (x_min, y_min, x_max, y_max) trong tọa độ local của patch
    """
    px_min, py_min = patch_x, patch_y
    px_max, py_max = patch_x + patch_size, patch_y + patch_size
    
    bx_min, by_min, bx_max, by_max, _ = bbox
    
    # Tính vùng giao
    ox_min = max(px_min, bx_min) - px_min
    oy_min = max(py_min, by_min) - py_min
    ox_max = min(px_max, bx_max) - px_min
    oy_max = min(py_max, by_max) - py_min
    
    return (ox_min, oy_min, ox_max, oy_max)


def remove_defect_region(patch_img, overlap_regions):
    """
    Xóa vùng lỗi khỏi patch bằng cách thay bằng vùng lân cận
    Sử dụng kỹ thuật đơn giản: thay bằng màu trung bình của vùng xung quanh
    
    Args:
        patch_img: PIL Image của patch
        overlap_regions: list các vùng cần xóa [(x_min, y_min, x_max, y_max), ...]
    
    Returns:
        PIL Image đã xóa vùng lỗi
    """
    img_array = np.array(patch_img)
    
    for region in overlap_regions:
        x_min, y_min, x_max, y_max = region
        
        # Đảm bảo bounds hợp lệ
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(img_array.shape[1], int(x_max))
        y_max = min(img_array.shape[0], int(y_max))
        
        if x_max <= x_min or y_max <= y_min:
            continue
        
        # Tính màu trung bình của vùng xung quanh (border 5 pixels)
        border = 5
        surround_x_min = max(0, x_min - border)
        surround_y_min = max(0, y_min - border)
        surround_x_max = min(img_array.shape[1], x_max + border)
        surround_y_max = min(img_array.shape[0], y_max + border)
        
        # Lấy vùng xung quanh (không bao gồm vùng defect)
        mask = np.ones((surround_y_max - surround_y_min, surround_x_max - surround_x_min), dtype=bool)
        inner_y_min = y_min - surround_y_min
        inner_y_max = y_max - surround_y_min
        inner_x_min = x_min - surround_x_min
        inner_x_max = x_max - surround_x_min
        mask[inner_y_min:inner_y_max, inner_x_min:inner_x_max] = False
        
        surround_region = img_array[surround_y_min:surround_y_max, surround_x_min:surround_x_max]
        
        if len(img_array.shape) == 3:  # Color image
            mean_color = surround_region[mask].mean(axis=0).astype(np.uint8)
            img_array[y_min:y_max, x_min:x_max] = mean_color
        else:  # Grayscale
            mean_color = surround_region[mask].mean().astype(np.uint8)
            img_array[y_min:y_max, x_min:x_max] = mean_color
    
    return Image.fromarray(img_array)


def process_image(image_path, annotation_path, output_defect_dir, output_normal_dir, 
                  patch_size=64, stride=32, image_index=0):
    """
    Xử lý một ảnh và tạo các patches
    
    Returns:
        (num_defect, num_normal): số lượng patches mỗi loại
    """
    # Load ảnh
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Load annotations
    bboxes = parse_annotation_file(annotation_path)
    
    num_defect = 0
    num_normal = 0
    
    # Sliding window
    for y in range(0, img_height - patch_size + 1, stride):
        for x in range(0, img_width - patch_size + 1, stride):
            # Cắt patch
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            
            # Kiểm tra từng bbox
            has_full_bbox = False
            partial_regions = []
            
            for bbox in bboxes:
                status = check_bbox_in_patch(x, y, patch_size, bbox)
                
                if status == 'full':
                    has_full_bbox = True
                    break
                elif status == 'partial':
                    overlap = get_overlap_region(x, y, patch_size, bbox)
                    partial_regions.append(overlap)
            
            # Quyết định label và xử lý
            if has_full_bbox:
                # Patch chứa toàn bộ bbox → defect
                patch_name = f"img{image_index:04d}_y{y:04d}_x{x:04d}.png"
                patch_resized = patch.resize((FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE))
                patch_resized.save(os.path.join(output_defect_dir, patch_name))
                num_defect += 1
            else:
                # Không có full bbox
                if partial_regions:
                    # Có partial overlap → xóa vùng lỗi
                    patch = remove_defect_region(patch, partial_regions)
                
                # Resize về FINAL_IMAGE_SIZE và lưu như normal
                patch_name = f"img{image_index:04d}_y{y:04d}_x{x:04d}.png"
                patch_resized = patch.resize((FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE))
                patch_resized.save(os.path.join(output_normal_dir, patch_name))
                num_normal += 1
    
    return num_defect, num_normal


def create_dataset_split(output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Tạo file CSV chia train/val/test set
    """
    defect_dir = os.path.join(output_dir, 'defect')
    normal_dir = os.path.join(output_dir, 'normal')
    
    # Lấy danh sách files
    defect_files = [(f, 1) for f in os.listdir(defect_dir) if f.endswith('.png')]
    normal_files = [(f, 0) for f in os.listdir(normal_dir) if f.endswith('.png')]
    
    print(f"\nTổng số patches:")
    print(f"  - Defect: {len(defect_files)}")
    print(f"  - Normal: {len(normal_files)}")
    
    # Shuffle
    np.random.seed(42)
    np.random.shuffle(defect_files)
    np.random.shuffle(normal_files)
    
    # Split
    def split_list(lst, train_r, val_r):
        n = len(lst)
        train_end = int(n * train_r)
        val_end = int(n * (train_r + val_r))
        return lst[:train_end], lst[train_end:val_end], lst[val_end:]
    
    defect_train, defect_val, defect_test = split_list(defect_files, train_ratio, val_ratio)
    normal_train, normal_val, normal_test = split_list(normal_files, train_ratio, val_ratio)
    
    # Combine
    train_data = defect_train + normal_train
    val_data = defect_val + normal_val
    test_data = defect_test + normal_test
    
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)
    
    # Save CSV files
    def save_csv(data, filename):
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.write("filename,label\n")
            for fname, label in data:
                folder = 'defect' if label == 1 else 'normal'
                f.write(f"{folder}/{fname},{label}\n")
    
    save_csv(train_data, 'train.csv')
    save_csv(val_data, 'val.csv')
    save_csv(test_data, 'test.csv')
    
    print(f"\nĐã tạo files split:")
    print(f"  - train.csv: {len(train_data)} samples")
    print(f"  - val.csv: {len(val_data)} samples")
    print(f"  - test.csv: {len(test_data)} samples")


def main():
    """
    Main function để xử lý toàn bộ dataset
    """
    print("=" * 60)
    print("DeepPCB Dataset Preprocessing")
    print("=" * 60)
    
    # Kiểm tra paths
    if not os.path.exists(INPUT_IMAGE_DIR):
        print(f"\n❌ Không tìm thấy thư mục ảnh: {INPUT_IMAGE_DIR}")
        print("Vui lòng cập nhật biến INPUT_IMAGE_DIR ở đầu file.")
        return
    
    if not os.path.exists(INPUT_LABEL_DIR):
        print(f"\n❌ Không tìm thấy thư mục labels: {INPUT_LABEL_DIR}")
        print("Vui lòng cập nhật biến INPUT_LABEL_DIR ở đầu file.")
        return
    
    # Tạo thư mục output
    output_defect_dir = os.path.join(OUTPUT_DIR, 'defect')
    output_normal_dir = os.path.join(OUTPUT_DIR, 'normal')
    
    os.makedirs(output_defect_dir, exist_ok=True)
    os.makedirs(output_normal_dir, exist_ok=True)
    
    print(f"\nCấu hình:")
    print(f"  - Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  - Stride: {STRIDE}")
    print(f"  - Input images: {INPUT_IMAGE_DIR}")
    print(f"  - Input labels: {INPUT_LABEL_DIR}")
    print(f"  - Output: {OUTPUT_DIR}")
    
    # Lấy danh sách ảnh
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(INPUT_IMAGE_DIR).glob(f'*{ext}'))
        image_files.extend(Path(INPUT_IMAGE_DIR).glob(f'*{ext.upper()}'))
    
    image_files = sorted(set(image_files))
    print(f"\nTìm thấy {len(image_files)} ảnh")
    
    if len(image_files) == 0:
        print("❌ Không tìm thấy ảnh nào!")
        return
    
    # Xử lý từng ảnh
    total_defect = 0
    total_normal = 0
    
    print("\nĐang xử lý...")
    for idx, img_path in enumerate(image_files):
        # Tìm file annotation tương ứng
        img_name = img_path.stem
        
        # Thử các tên file annotation khác nhau
        possible_anno_names = [
            f"{img_name}.txt",
            f"{img_name}_label.txt",
            f"{img_name}.xml",  # Có thể cần parser riêng cho XML
        ]
        
        anno_path = None
        for anno_name in possible_anno_names:
            test_path = os.path.join(INPUT_LABEL_DIR, anno_name)
            if os.path.exists(test_path):
                anno_path = test_path
                break
        
        if anno_path is None:
            print(f"  ⚠ Không tìm thấy annotation cho: {img_name}")
            continue
        
        # Xử lý ảnh
        num_def, num_norm = process_image(
            str(img_path), anno_path,
            output_defect_dir, output_normal_dir,
            PATCH_SIZE, STRIDE, idx
        )
        
        total_defect += num_def
        total_normal += num_norm
        
        print(f"  [{idx+1}/{len(image_files)}] {img_name}: {num_def} defect, {num_norm} normal")
    
    print("\n" + "=" * 60)
    print("Kết quả:")
    print(f"  - Tổng patches defect: {total_defect}")
    print(f"  - Tổng patches normal: {total_normal}")
    print(f"  - Tỷ lệ defect: {total_defect/(total_defect+total_normal)*100:.2f}%")
    
    # Tạo split files
    print("\n" + "=" * 60)
    print("Tạo train/val/test split...")
    create_dataset_split(OUTPUT_DIR, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    print("\n" + "=" * 60)
    print("✅ Hoàn thành!")
    print(f"Output được lưu tại: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
