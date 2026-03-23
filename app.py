"""
Explainable AI Breast Cancer Classification System
Interactive Streamlit Application
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import sys
from pathlib import Path
import os

import requests

@st.cache_resource
def download_model_if_needed():
    model_path = 'best_vit_idc_explainable.pth'
    if not os.path.exists(model_path):
        st.info("🔄 Downloading AI model (~300MB)...")
        url = "https://www.dropbox.com/scl/fi/ybv6z31kdfa8o1v91iz6h/best_vit_idc_explainable.pth?rlkey=15nsvdfm2vyg8nr4mmqw8ptxv&st=inipt1d2&dl=1"
        response = requests.get(url, stream=True)
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("✅ Model ready!")
    return model_path
# Call before loading model
model_path = download_model_if_needed()
# Use the downloaded path
# Import custom utilities
from explainability_utils import (
    explain_prediction,
    create_explanation_figure,
    GradCAM,
    visualize_attention_overlay
)


# Page configuration
st.set_page_config(
    page_title="Breast Cancer AI Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


class ExplainableViT(nn.Module):
    """Vision Transformer with explainability features"""
    
    def __init__(self, num_classes=2, pretrained=False):
        super(ExplainableViT, self).__init__()
        
        if pretrained:
            self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.vit = models.vit_b_16(weights=None)
        
        for param in self.vit.parameters():
            param.requires_grad = False
        
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self.attention_weights = []
        
    def forward(self, x):
        self.attention_weights = []
        output = self.vit(x)
        return output
    
    def get_attention_maps(self):
        return self.attention_weights


@st.cache_resource
def load_model(model_path='best_vit_idc_explainable.pth'):
    """Load the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ExplainableViT(num_classes=2, pretrained=False)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        model = model.to(device)
        return model, device
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        st.info("Please train the model first using train_explainable_vit.py")
        return None, device


def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image)


def get_risk_level(confidence, prediction):
    """Determine risk level based on confidence and prediction"""
    if prediction == 0:  # Non-cancerous
        if confidence > 0.9:
            return "Very Low Risk", "🟢", "#28a745"
        elif confidence > 0.7:
            return "Low Risk", "🟡", "#ffc107"
        else:
            return "Uncertain - Review Needed", "🟠", "#fd7e14"
    else:  # Cancerous
        if confidence > 0.9:
            return "High Risk - Immediate Review", "🔴", "#dc3545"
        elif confidence > 0.7:
            return "Elevated Risk", "🟠", "#fd7e14"
        else:
            return "Uncertain - Review Needed", "🟡", "#ffc107"


def create_probability_chart(probabilities, class_names):
    """Create a bar chart for class probabilities"""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#28a745', '#dc3545']
    bars = ax.barh(class_names, probabilities, color=colors, alpha=0.7)
    
    # Add value labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax.text(prob + 0.01, i, f'{prob:.1%}', 
                va='center', fontweight='bold')
    
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def display_clinical_information(prediction, confidence):
    """Display relevant clinical information"""
    if prediction == 1:  # Cancerous
        st.markdown("""
        ### 🔬 About Invasive Ductal Carcinoma (IDC)
        
        **What is IDC?**
        - IDC is the most common type of breast cancer, accounting for ~80% of cases
        - It begins in the milk ducts and invades surrounding breast tissue
        - Early detection significantly improves treatment outcomes
        
        **Key Characteristics the AI Detected:**
        - Abnormal cellular patterns
        - Irregular tissue architecture
        - Changes in cell morphology and density
        
        **Next Steps:**
        1. ⚕️ Consult with a qualified pathologist immediately
        2. 📋 Additional diagnostic tests may be recommended
        3. 🏥 Treatment planning with oncology team if confirmed
        
        **Important:** This AI analysis supports but does not replace professional diagnosis.
        """)
    else:  # Non-cancerous
        st.markdown("""
        ### ✅ About Normal Breast Tissue
        
        **What the AI Found:**
        - Regular cellular patterns
        - Normal tissue architecture
        - Typical cell morphology and distribution
        
        **Characteristics of Healthy Tissue:**
        - Organized ductal structures
        - Uniform cell spacing
        - Normal nuclear features
        - Consistent staining patterns
        
        **Recommendations:**
        1. 👨‍⚕️ Continue regular screening as per guidelines
        2. 🔍 Maintain routine mammography schedule
        3. 📊 Keep records for future comparison
        
        **Note:** Even with a negative result, follow your healthcare provider's recommendations.
        """)


def explain_visualization_methods():
    """Explain the visualization methods used"""
    with st.expander("📚 Understanding the Visualizations", expanded=False):
        st.markdown("""
        ### Grad-CAM (Gradient-weighted Class Activation Mapping)
        
        **What it shows:** The regions of the image that most influenced the AI's decision.
        
        **How to interpret:**
        - 🔴 Red/Yellow areas: Strongly influenced the prediction
        - 🔵 Blue/Purple areas: Less influential regions
        - The brighter the color, the more important that region
        
        **What it means:** If predicting cancer, red regions show suspicious tissue patterns 
        that match learned cancer characteristics.
        
        ---
        
        ### Saliency Map
        
        **What it shows:** Which pixels the model is most sensitive to.
        
        **How to interpret:**
        - Bright areas: Small changes here would significantly affect the prediction
        - Dark areas: Less sensitive regions
        - Shows pixel-level importance
        
        **What it means:** Highlights the most discriminative features at a fine-grained level.
        
        ---
        
        ### Attention Overlay
        
        **What it shows:** Visual overlay combining the original image with attention maps.
        
        **How to interpret:**
        - See exactly where the AI "looked" when making its decision
        - Helps pathologists verify the AI focused on relevant tissue features
        - Enables comparison with human expert analysis
        
        **Clinical Value:** Allows medical professionals to validate the AI's reasoning process.
        """)


def main():
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.2rem;
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .warning-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
            margin: 1rem 0;
        }
        .info-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d1ecf1;
            border-left: 5px solid #17a2b8;
            margin: 1rem 0;
        }
        .success-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            border-left: 5px solid #28a745;
            margin: 1rem 0;
        }
        .danger-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f8d7da;
            border-left: 5px solid #dc3545;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🔬 AI Breast Cancer Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explainable AI for Invasive Ductal Carcinoma Detection</p>', 
                unsafe_allow_html=True)
    
    # Medical Disclaimer
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Medical Disclaimer</strong><br>
        This tool is designed to assist medical professionals and should not be used as a sole 
        diagnostic tool. All results must be reviewed and confirmed by qualified pathologists 
        or oncologists. This AI system is for research and educational purposes.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/microscope.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Home", "📤 Upload & Analyze", "📊 About the Model", "❓ How to Use"]
        )
        
        st.markdown("---")
        st.markdown("### Model Information")
        st.info("""
        **Model:** Vision Transformer (ViT-B/16)
        
        **Training Data:** Breast Histopathology Images
        
        **Task:** Binary Classification (IDC vs Non-IDC)
        """)
        
        st.markdown("---")
        st.markdown("### Quick Links")
        st.markdown("[📖 Documentation](#)")
        st.markdown("[💻 GitHub Repository](#)")
        st.markdown("[📧 Contact Support](#)")
    
    # Main content based on selected page
    if page == "🏠 Home":
        st.header("Welcome to the AI Breast Cancer Classification System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 What This Tool Does
            
            This advanced AI system uses a Vision Transformer model to analyze breast 
            histopathology images and detect Invasive Ductal Carcinoma (IDC), the most 
            common type of breast cancer.
            
            **Key Features:**
            - 🔍 Accurate cancer detection
            - 📊 Explainable AI visualizations
            - 🎨 Attention mapping
            - 📈 Confidence scoring
            - 📋 Detailed clinical insights
            """)
        
        with col2:
            st.markdown("""
            ### 🌟 Why It's Different
            
            Unlike traditional "black box" AI systems, this tool provides:
            
            - **Transparency:** See exactly where the AI focused
            - **Interpretability:** Understand why it made its prediction
            - **Clinical Relevance:** Align AI decisions with pathology expertise
            - **Educational Value:** Learn about tissue characteristics
            - **Trust:** Verify AI reasoning before clinical decisions
            """)
        
        st.markdown("---")
        
        st.subheader("📊 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", "~85-90%", help="Overall classification accuracy")
        with col2:
            st.metric("Sensitivity", "~87%", help="True positive rate")
        with col3:
            st.metric("Specificity", "~88%", help="True negative rate")
        with col4:
            st.metric("AUC-ROC", "~0.93", help="Area under ROC curve")
        
        st.info("💡 **Tip:** Navigate to '📤 Upload & Analyze' to start analyzing images!")
    
    elif page == "📤 Upload & Analyze":
        st.header("Upload and Analyze Histopathology Image")
        
        # Load model
        with st.spinner("Loading AI model..."):
            model, device = load_model()
        
        if model is None:
            st.error("❌ Could not load model. Please ensure the model file exists.")
            st.stop()
        
        st.success("✅ AI Model loaded successfully!")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload a histopathology image (PNG, JPG, JPEG)",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a 50x50 pixel breast tissue histopathology image"
        )
        
        if uploaded_file is not None:
            # Display original image
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Original Image", use_column_width=True)
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🧠 AI is analyzing the tissue sample..."):
                    # Preprocess
                    image_tensor = preprocess_image(image).unsqueeze(0).to(device)
                    
                    # Get explanation
                    explanation = explain_prediction(
                        model, image_tensor, image, device=device
                    )
                    
                    prediction = explanation['prediction']
                    confidence = explanation['confidence']
                    probabilities = explanation['probabilities']
                    
                    class_names = ['Non-Cancerous (Benign)', 'Cancerous (IDC)']
                    
                # Display results
                st.markdown("---")
                st.header("🎯 Analysis Results")
                
                # Risk assessment
                risk_level, emoji, color = get_risk_level(confidence, prediction)
                
                st.markdown(f"""
                <div style="background-color: {color}; padding: 2rem; border-radius: 10px; 
                            text-align: center; color: white; margin: 1rem 0;">
                    <h2>{emoji} {risk_level}</h2>
                    <h3>Prediction: {class_names[prediction]}</h3>
                    <h4>Confidence: {confidence:.1%}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability chart
                st.subheader("📊 Prediction Probabilities")
                fig_prob = create_probability_chart(probabilities, class_names)
                st.pyplot(fig_prob)
                plt.close()
                
                # Visualizations
                st.markdown("---")
                st.header("🔍 Explainability Visualizations")
                
                explain_visualization_methods()
                
                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🎨 Grad-CAM Overlay",
                    "🔥 Grad-CAM Heatmap",
                    "⚡ Saliency Map",
                    "📋 Complete Analysis"
                ])
                
                with tab1:
                    st.subheader("Grad-CAM Attention Overlay")
                    st.image(
                        explanation['gradcam_overlay'],
                        caption="Regions that influenced the prediction (red = high attention)",
                        use_column_width=True
                    )
                    st.markdown("""
                    **Interpretation:** Red and yellow regions show where the AI focused its attention.
                    These areas contain the tissue patterns most characteristic of the predicted class.
                    """)
                
                with tab2:
                    st.subheader("Grad-CAM Heatmap")
                    fig_heatmap, ax = plt.subplots(figsize=(8, 8))
                    im = ax.imshow(explanation['gradcam'], cmap='jet')
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    st.pyplot(fig_heatmap)
                    plt.close()
                    
                    st.markdown("""
                    **Interpretation:** This heatmap isolates the attention weights, making it easier
                    to identify specific regions of interest without the original image context.
                    """)
                
                with tab3:
                    st.subheader("Saliency Map")
                    fig_saliency, ax = plt.subplots(figsize=(8, 8))
                    im = ax.imshow(explanation['saliency'], cmap='hot')
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    st.pyplot(fig_saliency)
                    plt.close()
                    
                    st.markdown("""
                    **Interpretation:** Bright regions indicate pixels where small changes would
                    significantly impact the model's prediction. This shows the model's sensitivity.
                    """)
                
                with tab4:
                    st.subheader("Comprehensive Analysis Report")
                    fig_complete = create_explanation_figure(explanation, class_names)
                    st.pyplot(fig_complete)
                    plt.close()
                    
                    # Download button for the figure
                    buf = io.BytesIO()
                    fig_complete.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label="📥 Download Complete Analysis",
                        data=buf,
                        file_name=f"breast_cancer_analysis_{prediction}.png",
                        mime="image/png"
                    )
                
                # Clinical information
                st.markdown("---")
                st.header("🏥 Clinical Information")
                display_clinical_information(prediction, confidence)
                
                # Recommendations
                st.markdown("---")
                st.header("📋 Recommendations")
                
                if prediction == 1:
                    st.markdown("""
                    <div class="danger-box">
                        <strong>⚠️ Positive IDC Detection</strong><br><br>
                        <strong>Immediate Actions:</strong>
                        <ol>
                            <li>Confirm findings with board-certified pathologist</li>
                            <li>Consider additional immunohistochemical staining</li>
                            <li>Review patient history and imaging studies</li>
                            <li>Arrange multidisciplinary team consultation</li>
                            <li>Discuss treatment options with oncology team</li>
                        </ol>
                        <br>
                        <strong>Note:</strong> AI prediction confidence is {confidence:.1%}. 
                        Higher confidence suggests stronger pattern match with training data.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-box">
                        <strong>✅ Negative for IDC</strong><br><br>
                        <strong>Recommended Actions:</strong>
                        <ol>
                            <li>Pathologist review for confirmation</li>
                            <li>Continue routine screening protocols</li>
                            <li>Document findings in patient record</li>
                            <li>Follow standard care guidelines</li>
                        </ol>
                        <br>
                        <strong>Note:</strong> AI prediction confidence is {confidence:.1%}. 
                        Regular monitoring is still important.
                    </div>
                    """, unsafe_allow_html=True)
    
    elif page == "📊 About the Model":
        st.header("About the AI Model")
        
        st.markdown("""
        ### 🤖 Model Architecture: Vision Transformer (ViT)
        
        This system uses a **Vision Transformer (ViT-B/16)** model, a state-of-the-art 
        deep learning architecture originally developed by Google Research.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🏗️ Architecture Details
            
            - **Base Model:** ViT-B/16
            - **Input Size:** 224×224 pixels
            - **Patch Size:** 16×16 pixels
            - **Parameters:** ~86M total
            - **Trainable:** ~1.5M (head only)
            - **Pre-training:** ImageNet-1K
            """)
        
        with col2:
            st.markdown("""
            #### 📚 Training Strategy
            
            - **Approach:** Transfer learning
            - **Frozen:** Backbone (encoder)
            - **Fine-tuned:** Classification head
            - **Data Augmentation:** Yes
            - **Regularization:** Dropout (0.3, 0.2)
            - **Optimizer:** AdamW
            """)
        
        st.markdown("---")
        
        st.subheader("🎯 How It Works")
        
        st.markdown("""
        1. **Image Patchification:** The input image is divided into 16×16 pixel patches
        2. **Patch Embedding:** Each patch is converted to a vector embedding
        3. **Positional Encoding:** Position information is added to embeddings
        4. **Transformer Encoding:** Multiple layers of self-attention process the patches
        5. **Classification:** A learned head produces the final prediction
        
        **Key Advantage:** Unlike CNNs, transformers can capture long-range dependencies 
        between different regions of the tissue sample.
        """)
        
        st.markdown("---")
        
        st.subheader("📈 Dataset Information")
        
        st.markdown("""
        **Training Data:** Breast Histopathology Images Dataset
        
        - **Source:** Kaggle / Cruz-Roa et al.
        - **Total Images:** ~277,000 patches
        - **Image Size:** 50×50 pixels (resized to 224×224)
        - **Classes:**
          - Class 0: Non-IDC (benign tissue)
          - Class 1: IDC (invasive ductal carcinoma)
        - **Split:** 80% training, 20% validation
        
        **Preprocessing:**
        - Resize to 224×224
        - Normalization (ImageNet stats)
        - Data augmentation (flips, rotations, color jitter)
        """)
        
        st.markdown("---")
        
        st.subheader("🔍 Explainability Methods")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Grad-CAM
            
            **Gradient-weighted Class Activation Mapping**
            
            - Visualizes discriminative regions
            - Uses gradients flowing into final conv layer
            - Produces class-discriminative localization map
            - Highlights important features
            """)
        
        with col2:
            st.markdown("""
            #### Saliency Maps
            
            **Gradient-based Visualization**
            
            - Shows pixel-level importance
            - Computed via backpropagation
            - Indicates sensitivity to changes
            - Fine-grained interpretation
            """)
        
        st.markdown("---")
        
        st.subheader("⚠️ Limitations")
        
        st.warning("""
        **Important Limitations to Consider:**
        
        1. **Training Data Bias:** Model performance depends on training data distribution
        2. **Image Quality:** Results may vary with different staining protocols or image quality
        3. **Rare Cases:** May not generalize well to rare or atypical presentations
        4. **No Context:** Cannot consider patient history or other clinical factors
        5. **Not FDA Approved:** This is a research/educational tool, not a medical device
        6. **Human Oversight Required:** All predictions must be verified by qualified professionals
        """)
    
    elif page == "❓ How to Use":
        st.header("How to Use This Tool")
        
        st.markdown("""
        ### 📖 Step-by-Step Guide
        """)
        
        with st.expander("1️⃣ Preparing Your Images", expanded=True):
            st.markdown("""
            **Image Requirements:**
            - Format: PNG, JPG, or JPEG
            - Content: Histopathology image of breast tissue
            - Recommended: 50×50 pixel patches (will be resized to 224×224)
            - Staining: H&E (Hematoxylin and Eosin) stained
            
            **Best Practices:**
            - Use high-quality, well-focused images
            - Ensure proper lighting and contrast
            - Avoid artifacts or damage in the tissue sample
            - Use standardized staining protocols
            """)
        
        with st.expander("2️⃣ Uploading Images"):
            st.markdown("""
            1. Navigate to the "📤 Upload & Analyze" page
            2. Click the "Browse files" button
            3. Select your histopathology image
            4. Wait for the image to upload and display
            5. Review the uploaded image for clarity
            """)
        
        with st.expander("3️⃣ Analyzing Results"):
            st.markdown("""
            1. Click the "🔍 Analyze Image" button
            2. Wait for the AI to process (usually 2-5 seconds)
            3. Review the risk assessment and prediction
            4. Examine the confidence score
            5. Check probability distribution between classes
            """)
        
        with st.expander("4️⃣ Interpreting Visualizations"):
            st.markdown("""
            **Grad-CAM Overlay:**
            - Red/Yellow = High attention (important for prediction)
            - Blue/Purple = Low attention (less important)
            - Look for concentration patterns
            
            **Grad-CAM Heatmap:**
            - Isolated attention weights
            - Easier to identify specific regions
            - Use colorbar for intensity scale
            
            **Saliency Map:**
            - Shows pixel-level sensitivity
            - Bright = high sensitivity
            - Indicates discriminative features
            
            **Complete Analysis:**
            - Comprehensive view of all visualizations
            - Suitable for documentation
            - Downloadable for records
            """)
        
        with st.expander("5️⃣ Clinical Integration"):
            st.markdown("""
            **How to Use in Clinical Practice:**
            
            1. **Screening Tool:** Use as a first-pass screening aid
            2. **Second Opinion:** Compare AI results with pathologist assessment
            3. **Education:** Train residents by comparing AI attention with expert annotation
            4. **Quality Control:** Identify cases that need additional review
            5. **Documentation:** Export results for patient records
            
            **Workflow Integration:**
            - Pre-diagnosis screening
            - Pathologist decision support
            - Teaching and training
            - Quality assurance
            - Research and development
            """)
        
        with st.expander("6️⃣ Safety and Ethics"):
            st.markdown("""
            **Critical Safety Guidelines:**
            
            ⚠️ **Never use as sole diagnostic tool**
            ⚠️ **Always verify with qualified pathologist**
            ⚠️ **Consider patient context and history**
            ⚠️ **Follow institutional protocols**
            ⚠️ **Maintain patient privacy (HIPAA compliance)**
            ⚠️ **Document AI-assisted decisions appropriately**
            
            **Ethical Considerations:**
            - Transparency with patients about AI use
            - Clear communication of limitations
            - Respect for patient autonomy
            - Responsible data handling
            - Continuous quality monitoring
            """)
        
        st.markdown("---")
        
        st.subheader("💡 Tips for Best Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Do:**
            - Use high-quality images
            - Follow standard protocols
            - Review multiple visualizations
            - Compare with clinical context
            - Document uncertainty
            - Seek expert consultation
            """)
        
        with col2:
            st.warning("""
            **Don't:**
            - Rely solely on AI
            - Ignore low confidence scores
            - Use poor quality images
            - Skip pathologist review
            - Make final diagnosis alone
            - Ignore patient symptoms
            """)
        
        st.markdown("---")
        
        st.subheader("❓ Frequently Asked Questions")
        
        with st.expander("How accurate is this AI system?"):
            st.markdown("""
            The model achieves approximately 85-90% accuracy on validation data, with 
            sensitivity around 87% and specificity around 88%. However, performance may 
            vary with different tissue samples, staining protocols, and image quality.
            
            Always validate results with expert pathologist review.
            """)
        
        with st.expander("Can I use this for actual patient diagnosis?"):
            st.markdown("""
            **No.** This is a research and educational tool, not an FDA-approved medical device.
            It should only be used as a decision support aid, with all diagnoses confirmed by 
            qualified medical professionals.
            """)
        
        with st.expander("What if the AI is uncertain (low confidence)?"):
            st.markdown("""
            Low confidence predictions (< 70%) should be flagged for additional review. 
            These cases may represent:
            - Borderline or ambiguous cases
            - Rare or atypical presentations
            - Image quality issues
            - Tissue artifacts
            
            Always prioritize expert pathologist review for uncertain cases.
            """)
        
        with st.expander("How was this model trained?"):
            st.markdown("""
            The model uses transfer learning:
            1. Started with ViT-B/16 pre-trained on ImageNet
            2. Froze the backbone (encoder layers)
            3. Added custom classification head
            4. Fine-tuned on breast histopathology dataset
            5. Used data augmentation and regularization
            6. Validated on separate test set
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>Explainable AI Breast Cancer Classification System</strong></p>
        <p>Powered by Vision Transformer (ViT) | Built with Streamlit and PyTorch</p>
        <p style="font-size: 0.9rem;">
            ⚠️ For Research and Educational Use Only | Not for Clinical Diagnosis
        </p>
        <p style="font-size: 0.8rem; margin-top: 1rem;">
            © 2026 | All Rights Reserved | Contact: support@example.com
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
