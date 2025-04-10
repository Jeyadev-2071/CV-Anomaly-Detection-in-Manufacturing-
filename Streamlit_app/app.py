import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from model import ImprovedAutoencoder, CarpetPatchAutoencoder
from model import BottleCarpetClassifier



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_bottle = ImprovedAutoencoder().to(device)
model_carpet = CarpetPatchAutoencoder().to(device)
classifier = BottleCarpetClassifier().to(device)
classifier.load_state_dict(torch.load("Models_dump/classifier_bottle_carpet.pth"))
model_bottle.load_state_dict(torch.load("Models_dump/model_bottle.pth", map_location=device))
model_carpet.load_state_dict(torch.load("Models_dump/model_carpet_patch.pth", map_location=device))
model_bottle.eval()
model_carpet.eval()
classifier.eval()


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def predict_category(tensor, classifier):
    with torch.no_grad():
        tensor = tensor.unsqueeze(0).to(device)
        logits = classifier(tensor)
        pred = torch.argmax(logits, dim=1).item()
        return "bottle" if pred == 0 else "carpet"


def detect_carpet_anomaly(image_tensor, model, patch_size=128, stride=64):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    _, _, H, W = image_tensor.shape
    error_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.uint8)
    with torch.no_grad():
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                patch = image_tensor[:, :, i:i+patch_size, j:j+patch_size]
                recon = model(patch)
                patch_error = torch.abs(patch - recon).mean(dim=1).squeeze().cpu().numpy()
                error_map[i:i+patch_size, j:j+patch_size] += patch_error
                count_map[i:i+patch_size, j:j+patch_size] += 1
    return error_map / np.maximum(count_map, 1)


def detect_bottle_anomaly(image_tensor, model):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(image_tensor)
        error_map = torch.abs(image_tensor - recon).mean(dim=1).squeeze().cpu().numpy()
    return error_map


def overlay_heatmap(original, error_map):
    original = np.array(original.resize((256, 256)))
    error_norm = ((error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(error_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    return overlay


st.title("Multi-Category Anomaly Detector (Carpet & Bottle)")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

category = st.selectbox("Select category of image", options=["carpet", "bottle"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    tensor = transform(img)
    category_predicted = predict_category(tensor, classifier)
    st.info(f"Predicted Category: **{category_predicted.upper()}**")
    if category == category_predicted:
        st.success("✅ Category matches the uploaded image.")
    
        if category == "carpet":
            error_map = detect_carpet_anomaly(tensor, model_carpet)
        else:
            error_map = detect_bottle_anomaly(tensor, model_bottle)

        overlay = overlay_heatmap(img, error_map)
        overlay_resized = cv2.resize(overlay, (300, 300))
        img_resized = img.resize((300, 300))
        
        score = np.mean(error_map) * 100
        st.metric(label="Reconstruction Error (%)", value=f"{score:.2f}")
        if category == "carpet":
            if score > 40:
                st.error("⚠️ This is a Defective carpet (high anomaly detected).")
            else:
                st.success("✅ This is a GOOD carpet.")
        else:
            if score > 10:
                st.error("⚠️ This is a Defective bottle (high anomaly detected).")
            else:
                st.success("✅ This is a GOOD bottle.")

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_resized, caption="Original Image", use_column_width=False)
        with col2:
            st.image(overlay_resized, caption="Anomaly Heatmap", channels="BGR", use_column_width=False)
    else:
        st.error("⚠️ Category does not match the uploaded image. Please check the category selection.")

    
    
