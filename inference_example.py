"""
Example Inference Script
Demonstrates how to use the model for prediction on a single image
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from explainability_utils import (
    explain_prediction,
    create_explanation_figure,
    visualize_attention_overlay
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


def load_model(model_path='best_vit_idc_explainable.pth', device='cuda'):
    """
    Load trained model
    
    Args:
        model_path: Path to saved model checkpoint
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded model
        device: Device being used
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = ExplainableViT(num_classes=2, pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model = model.to(device)
    
    return model, device


def preprocess_image(image_path):
    """
    Preprocess image for model input
    
    Args:
        image_path: Path to image file
    
    Returns:
        image_tensor: Preprocessed image tensor
        original_image: Original PIL image
    """
    # Load image
    original_image = Image.open(image_path).convert('RGB')
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transform
    image_tensor = transform(original_image)
    
    return image_tensor, original_image


def predict_single_image(model, image_path, device='cuda', save_explanation=True):
    """
    Make prediction on a single image with explainability
    
    Args:
        model: Trained model
        image_path: Path to image
        device: Device to run on
        save_explanation: Whether to save explanation figure
    
    Returns:
        results: Dictionary with prediction and explanation
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {image_path}")
    print(f"{'='*70}\n")
    
    # Preprocess
    image_tensor, original_image = preprocess_image(image_path)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Get prediction with explanation
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # Class names
    class_names = ['Non-Cancerous (Benign)', 'Cancerous (IDC)']
    
    # Print results
    print("🎯 PREDICTION RESULTS")
    print(f"   Class: {class_names[pred_class]}")
    print(f"   Confidence: {confidence:.2%}\n")
    
    print("📊 CLASS PROBABILITIES")
    for i, class_name in enumerate(class_names):
        prob = probs[0, i].item()
        bar = '█' * int(prob * 50)
        print(f"   {class_name:30s} {prob:6.2%} {bar}")
    
    # Get explanation
    print("\n🔍 Generating explainability visualizations...")
    explanation = explain_prediction(model, image_tensor, original_image, device)
    
    # Create and save explanation figure
    if save_explanation:
        fig = create_explanation_figure(explanation, class_names)
        
        # Create output filename
        output_path = Path(image_path).stem + '_explanation.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Explanation saved: {output_path}")
    
    # Risk assessment
    if pred_class == 0:  # Non-cancerous
        if confidence > 0.9:
            risk = "Very Low Risk 🟢"
        elif confidence > 0.7:
            risk = "Low Risk 🟡"
        else:
            risk = "Uncertain - Review Needed 🟠"
    else:  # Cancerous
        if confidence > 0.9:
            risk = "High Risk - Immediate Review 🔴"
        elif confidence > 0.7:
            risk = "Elevated Risk 🟠"
        else:
            risk = "Uncertain - Review Needed 🟡"
    
    print(f"\n⚠️  RISK ASSESSMENT: {risk}")
    
    print("\n" + "="*70)
    print("📋 CLINICAL RECOMMENDATION")
    print("="*70)
    
    if pred_class == 1:
        print("""
⚕️  Positive for IDC detected
   
   Recommended Actions:
   1. Immediate pathologist review
   2. Consider additional immunohistochemical staining
   3. Review patient history and imaging
   4. Consult multidisciplinary team
   5. Discuss treatment options with oncology
   
   ⚠️  This is AI-assisted analysis. Always confirm with qualified pathologist.
        """)
    else:
        print("""
✅ Negative for IDC
   
   Recommended Actions:
   1. Pathologist confirmation
   2. Continue routine screening
   3. Document in patient record
   4. Follow standard care guidelines
   
   ℹ️  Regular monitoring remains important even with negative result.
        """)
    
    return {
        'prediction': pred_class,
        'confidence': confidence,
        'probabilities': probs[0].cpu().numpy(),
        'risk_level': risk,
        'explanation': explanation
    }


def batch_predict(model, image_dir, device='cuda'):
    """
    Predict on multiple images in a directory
    
    Args:
        model: Trained model
        image_dir: Directory containing images
        device: Device to run on
    
    Returns:
        results: List of prediction results
    """
    image_dir = Path(image_dir)
    
    # Find all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(image_dir.glob(ext))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return []
    
    print(f"\nFound {len(image_files)} images")
    print("Starting batch prediction...\n")
    
    results = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
        
        try:
            result = predict_single_image(
                model, str(image_path), device,
                save_explanation=False  # Don't save individual explanations for batch
            )
            results.append({
                'image': image_path.name,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'risk_level': result['risk_level']
            })
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
            results.append({
                'image': image_path.name,
                'prediction': -1,
                'confidence': 0.0,
                'risk_level': 'Error'
            })
    
    # Print summary
    print("\n" + "="*70)
    print("📊 BATCH PREDICTION SUMMARY")
    print("="*70 + "\n")
    
    class_names = ['Non-Cancerous', 'Cancerous (IDC)']
    
    for result in results:
        pred = result['prediction']
        if pred >= 0:
            pred_name = class_names[pred]
            print(f"{result['image']:40s} {pred_name:20s} {result['confidence']:6.2%}  {result['risk_level']}")
        else:
            print(f"{result['image']:40s} {'Error':20s}")
    
    # Statistics
    valid_results = [r for r in results if r['prediction'] >= 0]
    if valid_results:
        cancerous = sum(1 for r in valid_results if r['prediction'] == 1)
        print(f"\n📈 Statistics:")
        print(f"   Total processed: {len(valid_results)}")
        print(f"   Cancerous: {cancerous} ({cancerous/len(valid_results)*100:.1f}%)")
        print(f"   Non-cancerous: {len(valid_results)-cancerous} ({(len(valid_results)-cancerous)/len(valid_results)*100:.1f}%)")
    
    return results


def main():
    """
    Main function - example usage
    """
    print("\n" + "="*70)
    print("🔬 BREAST CANCER AI CLASSIFIER - INFERENCE EXAMPLE")
    print("="*70)
    
    # Configuration
    MODEL_PATH = 'best_vit_idc_explainable.pth'
    IMAGE_PATH = 'path/to/your/image.png'  # Update this
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"\n❌ Error: Model not found at {MODEL_PATH}")
        print("   Please train the model first using train_explainable_vit.py")
        return
    
    # Load model
    print("\n📦 Loading model...")
    model, device = load_model(MODEL_PATH, DEVICE)
    print("✅ Model loaded successfully!\n")
    
    # Example 1: Single image prediction
    if Path(IMAGE_PATH).exists():
        print("\n" + "="*70)
        print("EXAMPLE 1: SINGLE IMAGE PREDICTION")
        print("="*70)
        
        result = predict_single_image(model, IMAGE_PATH, device)
    else:
        print(f"\n⚠️  Example image not found: {IMAGE_PATH}")
        print("   Update IMAGE_PATH in the script to point to your image")
    
    # Example 2: Batch prediction (commented out by default)
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: BATCH PREDICTION")
    print("="*70)
    
    IMAGE_DIR = 'path/to/image/directory'
    if Path(IMAGE_DIR).exists():
        results = batch_predict(model, IMAGE_DIR, device)
        
        # Save results to CSV
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv('batch_predictions.csv', index=False)
        print(f"\n✅ Results saved to batch_predictions.csv")
    """
    
    print("\n" + "="*70)
    print("✅ INFERENCE COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
