import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Set wide layout
st.set_page_config(layout="wide")

# Light gradient background and styling
st.markdown("""
<style>
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(to right, #e0f7fa, #ffffff);
}

/* Soft shadow & rounded corners on containers */
.element-container, .stImage, .stFileUploader, .stButton {
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    border-radius: 12px;
    padding: 10px;
    background-color: #ffffff;
}

/* Button style and hover animation */
.stButton > button {
    background-color: #00b894;
    color: white;
    transition: background-color 0.3s ease;
    border-radius: 8px;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #019875;
}

/* Title style */
h1 {
    text-align: center;
    color: #00796b;
    font-family: Arial, sans-serif;
    text-shadow: 1px 1px 2px #aaa;
    margin-bottom: 5px;
}

/* Brief description style */
.description {
    text-align: center;
    font-family: Arial, sans-serif;
    color: #004d40;
    font-size: 18px;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>SolarGuard: Intelligent Solar Panel Defect Classifier</h1>", unsafe_allow_html=True)

# Brief description below title
st.markdown(
    "<p class='description'>A cutting-edge deep learning system to accurately detect and classify defects on solar panels, helping improve maintenance efficiency and energy output.</p>",
    unsafe_allow_html=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_labels = [
    'Bird-drop',
    'Clean',
    'Dusty',
    'Electrical-damage',
    'Physical-damage',
    'Snow-covered'
]

@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_labels))
    model.load_state_dict(torch.load(r"D:\SolarGuard Intelligent Defect Detection\solar_classifier.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Two column layout: big image left, uploader & results right
col1, col2 = st.columns([1,1])

with col1:
    img_path = r"D:\SolarGuard Intelligent Defect Detection\streamlit\image.png"
    img = Image.open(img_path)
    st.image(img, use_container_width=True)

with col2:
    uploaded_file = st.file_uploader("Upload a solar panel image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Classify Defect"):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_class = class_labels[predicted.item()]
                confidence_score = confidence.item()

            st.success(f"**Predicted Class:** {predicted_class}")
            st.info(f"**Confidence:** {confidence_score:.2%}")

            st.subheader("Class Probabilities")
            prob_dict = {label: float(probabilities[0][i]) for i, label in enumerate(class_labels)}
            st.bar_chart(prob_dict)
