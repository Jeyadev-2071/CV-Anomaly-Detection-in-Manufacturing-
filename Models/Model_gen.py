

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import roc_auc_score

###########################################
# 1. Test Dataset
###########################################

class MVTecTestDataset(Dataset):
    def __init__(self, test_dirs, transform=None):
        self.image_paths = []
        for test_dir in test_dirs:
            for fname in os.listdir(test_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(test_dir, fname))
        self.image_paths.sort()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

#############################################
# 2. Autoencoder Models
#############################################

class ImprovedAutoencoder(nn.Module):
    def __init__(self):
        super(ImprovedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class CarpetPatchAutoencoder(nn.Module):
    def __init__(self):
        super(CarpetPatchAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

#############################################
# 3. Utilities
#############################################

def compute_pixel_error_map(original, reconstructed):
    error = torch.abs(original - reconstructed)
    return error.mean(dim=1, keepdim=True).squeeze().cpu().numpy()

def get_mask_path(img_path):
    return img_path.replace("/test", "/ground_truth").replace(".png", "_mask.png")

def compute_iou(error_map, mask_path, threshold=0.5):
    if not os.path.exists(mask_path): return None
    mask = Image.open(mask_path).convert("L").resize(error_map.shape[::-1], Image.NEAREST)
    gt = (np.array(mask) > 128).astype(np.uint8)
    pred = (error_map > threshold).astype(np.uint8)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(intersection) / union if union > 0 else 1.0

def compute_roc_auc(error_map, mask_path):
    if not os.path.exists(mask_path): return None
    mask = Image.open(mask_path).convert("L").resize(error_map.shape[::-1], Image.NEAREST)
    y_true = (np.array(mask).flatten() > 128).astype(np.uint8)
    y_score = error_map.flatten()
    return roc_auc_score(y_true, y_score) if y_true.sum() > 0 else None

def generate_heatmap(error_map, original_tensor):
    error_norm = ((error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(error_norm, cv2.COLORMAP_JET)
    original_np = (original_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
    return overlay

def save_visualization(overlay, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

#############################################
# 4. Evaluation Function
#############################################

def evaluate(test_loader, model_bottle, model_carpet, device, input_base, output_base, threshold=0.3):
    model_bottle.eval()
    model_carpet.eval()
    results = []
    patch_size = 128
    stride = 64

    for image, img_path in test_loader:
        image = image.to(device)
        is_carpet = "carpet" in img_path[0].lower()
        mask_path = get_mask_path(img_path[0])

        if is_carpet:
            _, _, H, W = image.shape
            error_map = np.zeros((H, W), dtype=np.float32)
            count_map = np.zeros((H, W), dtype=np.uint8)
            with torch.no_grad():
                for i in range(0, H - patch_size + 1, stride):
                    for j in range(0, W - patch_size + 1, stride):
                        patch = image[:, :, i:i+patch_size, j:j+patch_size]
                        recon = model_carpet(patch)
                        patch_error = compute_pixel_error_map(patch, recon)
                        error_map[i:i+patch_size, j:j+patch_size] += patch_error
                        count_map[i:i+patch_size, j:j+patch_size] += 1
                error_map = error_map / np.maximum(count_map, 1)
        else:
            with torch.no_grad():
                recon = model_bottle(image)
            error_map = compute_pixel_error_map(image, recon)

        iou = compute_iou(error_map, mask_path, threshold)
        auc = compute_roc_auc(error_map, mask_path)
        error_percent = np.mean(error_map) * 100
        rel_path = os.path.relpath(img_path[0], input_base)
        out_path = os.path.join(output_base, rel_path)
        overlay = generate_heatmap(error_map, image.squeeze())
        save_visualization(overlay, out_path)
        results.append({
            "file": rel_path,
            "error_percent": error_percent,
            "iou": iou,
            "auc": auc
        })
    return pd.DataFrame(results)

#############################################
# 5. Main Execution
#############################################

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dirs = [
        "Input/bottle/test/broken_large",
        "Input/bottle/test/broken_small",
        "Input/bottle/test/contamination",
        "Input/carpet/test/color",
        "Input/carpet/test/cut",
        "Input/carpet/test/hole",
        "Input/carpet/test/thread"
    ]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    test_dataset = MVTecTestDataset(test_dirs, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model_bottle = ImprovedAutoencoder().to(device)
    model_carpet = CarpetPatchAutoencoder().to(device)
    # model_bottle.load_state_dict(torch.load("model_bottle.pth"))
    # model_carpet.load_state_dict(torch.load("model_carpet_patch.pth"))
    df = evaluate(
        test_loader,
        model_bottle,
        model_carpet,
        device,
        input_base="Input",
        output_base="Output/Heatmaps",
        threshold=0.3
    )
    print(df)
    df.to_csv("Output/anomaly_summary.csv", index=False)