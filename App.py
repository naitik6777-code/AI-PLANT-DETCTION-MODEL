import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

# Load your trained model
@st.cache_resource
def load_model():
    model = torch.load("model/plant_model.pth", map_location="cpu")
    model.eval()
    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("🌱 Plant Detection AI")
st.write("Upload an image and let the AI detect the plant!")

uploaded_file = st.file_uploader("Upload a plant image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)
        st.success(f"Prediction: **{predicted.item()}**")
