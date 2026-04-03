#!/usr/bin/env python3
"""
Integrated Potato Disease Prediction Dashboard
Weather + Image Analysis with ViT
"""

import streamlit as st
import torch
import numpy as np
import pickle
from pathlib import Path
from PIL import Image
from torchvision import transforms
import timm
import json
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Potato Disease Predictor",
    page_icon="🥔",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2d6a4f;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #52b788;
        margin-top: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .risk-high {
        background-color: #ffe5e5;
        border-left: 5px solid #ff4444;
    }
    .risk-medium {
        background-color: #fff8e5;
        border-left: 5px solid #ffaa00;
    }
    .risk-low {
        background-color: #e5ffe5;
        border-left: 5px solid: #44ff44;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_weather_model():
    """Load weather prediction model"""
    try:
        model_path = Path("data/models/best_weather_model.pkl")
        if not model_path.exists():
            model_path = Path("data/models/xgboost_weather_model.pkl")
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        with open("data/processed/label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        
        with open("data/processed/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        return model, le, scaler
    except Exception as e:
        st.error(f"❌ Weather model not loaded: {e}")
        return None, None, None

@st.cache_resource
def load_vit_model():
    """Load ViT image classification model"""
    try:
        # Setup device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        # Load metadata
        with open("data/processed/metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Load model
        model_path = Path("data/models/vit_best.pth")
        checkpoint = torch.load(model_path, map_location=device)
        model_name = checkpoint.get('model_name', 'vit_small_patch16_224')
        
        model = timm.create_model(model_name, pretrained=False, num_classes=3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return model, transform, metadata, device
    except Exception as e:
        st.error(f"❌ ViT model not loaded: {e}")
        return None, None, None, None

def predict_weather(weather_data, model, scaler, le):
    """Predict disease risk from weather"""
    # Prepare features
    features = np.array([[
        weather_data['temperature'],
        weather_data['humidity'],
        weather_data['rainfall'],
        weather_data['wind_speed'],
        weather_data['dew_point']
    ]])
    
    # Pad to match training features
    if scaler.mean_.shape[0] > features.shape[1]:
        features = np.pad(features, ((0, 0), (0, scaler.mean_.shape[0] - features.shape[1])),
                         mode='constant', constant_values=0)
    
    # Scale and predict
    features_scaled = scaler.transform(features[:, :scaler.mean_.shape[0]])
    prediction = model.predict(features_scaled)
    proba = model.predict_proba(features_scaled) if hasattr(model, 'predict_proba') else None
    
    return le.classes_[prediction[0]], proba

def predict_image(image, model, transform, metadata, device):
    """Predict disease from leaf image"""
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = probs.argmax(1).item()
        confidence = probs[0][pred_idx].item()
    
    idx_to_class = {v: k for k, v in metadata['classes'].items()}
    predicted_class = idx_to_class[pred_idx]
    
    # Get all class probabilities
    class_probs = {}
    for idx, class_name in idx_to_class.items():
        class_probs[class_name] = probs[0][idx].item()
    
    return predicted_class, confidence, class_probs

def main():
    # Header
    st.markdown('<h1 class="main-header">🥔 Potato Disease Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("### Dual-Model AI: Weather Analysis + Vision Transformer")
    
    # Load models
    with st.spinner("Loading AI models..."):
        weather_model, weather_le, weather_scaler = load_weather_model()
        vit_model, vit_transform, vit_metadata, device = load_vit_model()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🌤️ Weather Prediction", "🖼️ Image Analysis", "🔬 Combined Analysis"])
    
    # Tab 1: Weather Prediction
    with tab1:
        st.markdown('<p class="sub-header">Weather-Based Disease Risk Prediction</p>', unsafe_allow_html=True)
        
        if weather_model is None:
            st.warning("⚠️ Weather model not available. Please train the model first.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Enter Weather Conditions")
                temperature = st.slider("Temperature (°C)", 10.0, 35.0, 22.0, 0.5)
                humidity = st.slider("Humidity (%)", 40.0, 100.0, 85.0, 1.0)
                rainfall = st.slider("Rainfall (mm)", 0.0, 30.0, 5.0, 0.5)
            
            with col2:
                st.write("")  # Spacing
                st.write("")
                wind_speed = st.slider("Wind Speed (km/h)", 0.0, 25.0, 8.0, 0.5)
                dew_point = st.slider("Dew Point (°C)", 5.0, 25.0, 18.0, 0.5)
            
            if st.button("🔍 Predict Disease Risk", type="primary"):
                weather_data = {
                    'temperature': temperature,
                    'humidity': humidity,
                    'rainfall': rainfall,
                    'wind_speed': wind_speed,
                    'dew_point': dew_point
                }
                
                prediction, proba = predict_weather(weather_data, weather_model, weather_scaler, weather_le)
                
                # Display result
                risk_class = "risk-high" if "High" in prediction else "risk-medium" if "Medium" in prediction else "risk-low"
                
                st.markdown(f"""
                <div class="result-box {risk_class}">
                    <h2>Prediction: {prediction}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Show probabilities if available
                if proba is not None:
                    st.subheader("📊 Risk Probabilities")
                    prob_df = {weather_le.classes_[i]: proba[0][i] for i in range(len(weather_le.classes_))}
                    
                    fig = go.Figure(data=[
                        go.Bar(x=list(prob_df.keys()), y=list(prob_df.values()),
                              marker_color=['#ff4444' if 'High' in k else '#ffaa00' if 'Medium' in k else '#44ff44' 
                                          for k in prob_df.keys()])
                    ])
                    fig.update_layout(title="Risk Distribution", yaxis_title="Probability")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 Recommendations")
                if "High" in prediction:
                    st.error("""
                    ⚠️ **High Risk Conditions Detected!**
                    - Apply fungicide preventatively
                    - Improve field drainage
                    - Increase plant spacing for airflow
                    - Monitor crops daily
                    """)
                elif "Medium" in prediction:
                    st.warning("""
                    ⚠️ **Moderate Risk**
                    - Monitor weather closely
                    - Check plants for early symptoms
                    - Have fungicide ready
                    """)
                else:
                    st.success("""
                    ✅ **Low Risk**
                    - Continue regular monitoring
                    - Maintain good cultural practices
                    """)
    
    # Tab 2: Image Analysis
    with tab2:
        st.markdown('<p class="sub-header">Leaf Image Disease Detection (ViT)</p>', unsafe_allow_html=True)
        
        if vit_model is None:
            st.warning("⚠️ ViT model not available. Please train the model first.")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📤 Upload Leaf Image")
                uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
                
                if uploaded_file is not None:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                if uploaded_file is not None:
                    with st.spinner("🔬 Analyzing image with Vision Transformer..."):
                        predicted_class, confidence, class_probs = predict_image(
                            image, vit_model, vit_transform, vit_metadata, device
                        )
                    
                    st.subheader("🎯 Prediction Results")
                    
                    # Format class name
                    disease_name = predicted_class.replace('Potato___', '').replace('_', ' ')
                    
                    st.markdown(f"""
                    <div class="result-box risk-high" style="border-left: 5px solid #2d6a4f;">
                        <h2>{disease_name}</h2>
                        <h3>Confidence: {confidence*100:.1f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show all probabilities
                    st.subheader("📊 Class Probabilities")
                    prob_data = {k.replace('Potato___', '').replace('_', ' '): v*100 
                                for k, v in class_probs.items()}
                    
                    fig = px.bar(x=list(prob_data.keys()), y=list(prob_data.values()),
                                labels={'x': 'Class', 'y': 'Confidence (%)'},
                                color=list(prob_data.values()),
                                color_continuous_scale='RdYlGn_r')
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Disease info
                    st.subheader("📚 Disease Information")
                    if "Early_Blight" in predicted_class:
                        st.info("""
                        **Early Blight** (Alternaria solani)
                        - Symptoms: Dark brown spots with concentric rings
                        - Favored by: Warm temperatures (24-29°C), high humidity
                        - Treatment: Fungicides, crop rotation, resistant varieties
                        """)
                    elif "Late_Blight" in predicted_class:
                        st.info("""
                        **Late Blight** (Phytophthora infestans)
                        - Symptoms: Water-soaked lesions, white fungal growth
                        - Favored by: Cool temperatures (15-20°C), wet conditions
                        - Treatment: Systemic fungicides, destroy infected plants
                        """)
                    else:
                        st.success("""
                        **Healthy Plant** ✅
                        - No disease detected
                        - Continue regular monitoring and care
                        """)
    
    # Tab 3: Combined Analysis
    with tab3:
        st.markdown('<p class="sub-header">🔬 Integrated Weather + Image Analysis</p>', unsafe_allow_html=True)
        st.info("💡 Upload an image and enter weather conditions for comprehensive analysis!")
        
        # This tab can show both predictions side by side
        st.markdown("**Coming soon: Real-time field monitoring integration**")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔬 Powered by Vision Transformer (ViT) + XGBoost | 🚀 Mac MPS Optimized</p>
        <p>Built for precision agriculture and crop health monitoring</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
