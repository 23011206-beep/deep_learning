import kagglehub

# Download latest version
path = kagglehub.dataset_download("liuxiaolong1/pcb-defect-detection-dataset")

print("Path to dataset files:", path)