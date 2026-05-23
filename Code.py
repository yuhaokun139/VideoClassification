import streamlit as st
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    CLIPProcessor,
    CLIPModel
)
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="Grocery Product Classifier", layout="wide")
st.title("🛒 Grocery Product Classifier & Attribute Tagger")
st.markdown("Upload a grocery product image...")

# ==========================
# Pipeline 1: Grocery category classification (fine-tuned ConvNeXt)
@st.cache_resource
def load_category_model():
    model_id = "facebook/convnext-tiny-224"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    return processor, model

# ==========================
# Pipeline 2: Fine-grained attribute prediction using Fashion-CLIP (zero-shot)
@st.cache_resource
def load_attribute_model():
    model_id = "patrickjohncyh/fashion-clip"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.eval()
    return processor, model

# Candidate attribute labels for different product categories
# You can expand this dictionary based on your training classes
attribute_candidates = {
    "apple": ["green apple", "red apple", "yellow apple", "gala apple", "fuji apple"],
    "banana": ["green banana", "ripe yellow banana", "overripe banana", "baby banana"],
    "orange": ["orange", "navel orange", "mandarin", "blood orange"],
    "lemon": ["lemon", "yellow lemon", "green lemon"],
    "milk": ["whole milk", "skim milk", "soy milk", "almond milk"],
    "bread": ["white bread", "whole wheat bread", "baguette", "croissant"],
    "tomato": ["red tomato", "green tomato", "cherry tomato", "roma tomato"],
    "potato": ["white potato", "sweet potato", "red potato"],
    "broccoli": ["fresh broccoli", "organic broccoli"],
    "carrot": ["orange carrot", "purple carrot", "baby carrot"],
}
DEFAULT_CANDIDATES = ["green apple", "red apple", "yellow apple", "banana", "orange", "whole milk"]

def predict_attributes(image, processor, model, candidate_labels):
    """
    Zero-shot classification using CLIP.
    Returns probabilities for each candidate label.
    """
    inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # outputs.logits_per_image: shape (1, num_candidates)
        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=0)
    return probs

# ==========================
# Main UI
uploaded_file = st.file_uploader("📁 Choose an image of a grocery product",
                                 type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Load models
    with st.spinner("Loading AI models..."):
        cat_processor, cat_model = load_category_model()
        attr_processor, attr_model = load_attribute_model()

    # ---- Pipeline 1: Category classification ----
    with st.spinner("Identifying product category..."):
        inputs = cat_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = cat_model(**inputs)
            logits = outputs.logits
            predicted_idx = logits.argmax(-1).item()
            probs = torch.nn.functional.softmax(logits, dim=-1)
            confidence = probs[0][predicted_idx].item()
        category = cat_model.config.id2label[predicted_idx].lower()
        st.success(f"Detected category: **{category.title()}** (confidence: {confidence:.2%})")

    # ---- Pipeline 2: Attribute prediction ----
    with st.spinner("Extracting fine-grained attributes..."):
        # Select candidate labels based on detected category
        candidates = attribute_candidates.get(category, DEFAULT_CANDIDATES)
        attr_probs = predict_attributes(image, attr_processor, attr_model, candidates)
        # Get top 3 most probable attributes
        top_probs, top_indices = torch.topk(attr_probs, k=min(3, len(candidates)))

    # Display results
    with col2:
        st.subheader("🔖 Fine-grained Attributes")
        for i in range(len(top_indices)):
            label = candidates[top_indices[i]]
            prob = top_probs[i].item()
            st.write(f"• **{label}**: {prob:.2%}")

        st.caption("Note: Attributes are predicted zero-shot using Fashion-CLIP.")
else:
    st.info("👆 Please upload an image to get started.")
