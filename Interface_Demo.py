
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- U-Net Model Definitions ---
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        self.b = conv_block(512, 1024)
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        b = self.b(p4)
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        outputs = self.outputs(d4)
        return outputs

# --- Classification Model Definitions ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)  # 256x256 input → 64x64 feature map
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Load Models ---
seg_model = build_unet().to(device)
seg_model.load_state_dict(torch.load('/Users/jyotisingh/Desktop/niya/checkpoint.pth', map_location=device))
seg_model.eval()

cls_model = SimpleCNN().to(device)
cls_model.load_state_dict(torch.load('/Users/jyotisingh/Desktop/niya/model.pth', map_location=device))
cls_model.eval()

# --- Page Config ---
st.set_page_config(page_title="Fundus DR Detection", layout="wide")

# --- Sidebar Info ---
st.sidebar.title("🧬 Fundus DR Detection")
st.sidebar.markdown("Upload a fundus image to perform:")
st.sidebar.markdown("- 🔬 Segmentation using U-Net")
st.sidebar.markdown("- 🧠 Classification using CNN")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📤 Upload Fundus Image", type=["png", "jpg", "jpeg", "tif"])
st.sidebar.image("/Users/jyotisingh/Desktop/niya/assets/dr.jpg", caption="Sample Input", use_column_width=True)

# --- Title and Header ---
st.markdown("<h1 style='text-align: center;'>🧫 Fundus Image Segmentation & DR Classification</h1><hr>", unsafe_allow_html=True)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("⚠️ Could not read the image. Make sure it's a valid format.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔍 Original Image")
            st.image(image, channels="BGR", use_container_width=True)

        # --- Segmentation ---
        st.subheader("🔬 Step 1: U-Net Segmentation")
        img_resized = cv2.resize(image, (512, 512))
        input_seg = torch.tensor(img_resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            mask_pred = seg_model(input_seg)
            mask_pred = torch.sigmoid(mask_pred)
            mask_pred = mask_pred.squeeze().cpu().numpy()
            mask_binary = (mask_pred > 0.5).astype(np.uint8) * 255

        with col2:
            st.subheader("🧪 Segmented Mask")
            st.image(mask_binary, use_container_width=True, clamp=True)

        # --- Classification ---
        st.subheader("🧬 Step 2: DR Classification")
        input_cls = cv2.resize(mask_binary, (256, 256))
        input_tensor = torch.tensor(input_cls / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = cls_model(input_tensor)
            _, prediction = torch.max(output, 1)

        label_map = {0: "🟥 DR Detected", 1: "🟩 No DR Detected"}
        prediction_label = label_map[prediction.item()]

        st.success(f"🧠 **Prediction:** `{prediction_label}`")

        # --- Save Option ---
        with st.expander("💾 Save Result"):
            if st.button("📥 Save Segmented Output"):
                output_dir = "outputs"
                os.makedirs(output_dir, exist_ok=True)
                save_name = f"{prediction_label.split()[-1].lower()}_{uploaded_file.name}"
                output_path = os.path.join(output_dir, save_name)
                cv2.imwrite(output_path, mask_binary)
                st.success(f"✅ Segmented output saved to `{output_path}`")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>🧠 Built with PyTorch + Streamlit | 👩‍⚕️ Medical AI Prototype</p>", unsafe_allow_html=True)
