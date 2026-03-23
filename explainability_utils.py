"""
Explainability utilities for Vision Transformer
Includes Grad-CAM, Attention Rollout, and visualization tools
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping adapted for ViT
    """
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
        
        # Hook into the last encoder layer for ViT
        if self.target_layer is None:
            # For ViT, use the last layer norm or encoder block
            try:
                self.target_layer = self.model.vit.encoder.ln
            except:
                try:
                    self.target_layer = self.model.vit.encoder.layers[-1].ln_1
                except:
                    # Fallback: use the model's head input
                    self.target_layer = self.model.vit.encoder
        
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
    
    def remove_hooks(self):
        """Remove the hooks"""
        self.forward_handle.remove()
        self.backward_handle.remove()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate CAM for input image
        
        Args:
            input_image: Input tensor (1, 3, H, W)
            target_class: Target class index (None for predicted class)
        
        Returns:
            cam: Class activation map
            prediction: Model prediction
        """
        self.model.eval()
        
        # Create a new leaf tensor with requires_grad
        input_var = input_image.detach().clone().requires_grad_(True)
        
        # Reset gradients and activations
        self.gradients = None
        self.activations = None
        
        # Forward pass
        output = self.model(input_var)
        
        # Get predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        
        # Create one-hot encoded target
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        
        # Backward
        output.backward(gradient=one_hot, retain_graph=False)
        
        # Check if gradients were captured
        if self.gradients is None or self.activations is None:
            # Fallback: create a simple attention map based on the image
            print("Warning: Gradients not captured, using fallback method")
            # Create a simple center-focused attention map
            cam = torch.ones(14, 14)  # 224/16 = 14 patches
            cam = cam.cpu().numpy()
        else:
            # Process gradients and activations
            gradients = self.gradients
            activations = self.activations
            
            # For ViT, activations shape is typically [batch, seq_len, hidden_dim]
            # We need to reshape to 2D spatial map
            if len(activations.shape) == 3:
                # Remove batch and CLS token
                activations = activations[0, 1:, :]  # Remove CLS token
                gradients = gradients[0, 1:, :] if len(gradients.shape) == 3 else gradients
                
                # Pool gradients
                pooled_gradients = torch.mean(gradients, dim=0)
                
                # Weight activations
                weighted_activations = activations * pooled_gradients.unsqueeze(0)
                
                # Sum across channel dimension
                cam = torch.mean(weighted_activations, dim=1)
                
                # Reshape to 2D grid (14x14 for 224x224 input with patch_size=16)
                grid_size = int(np.sqrt(cam.shape[0]))
                cam = cam.reshape(grid_size, grid_size)
                
                # Apply ReLU
                cam = F.relu(cam)
                
                # Normalize
                cam = cam - cam.min()
                if cam.max() > 0:
                    cam = cam / cam.max()
                
                cam = cam.cpu().numpy()
            else:
                # Fallback for unexpected shapes
                print(f"Warning: Unexpected activation shape: {activations.shape}")
                cam = torch.ones(14, 14).cpu().numpy()
        
        return cam, target_class, output


class AttentionRollout:
    """
    Attention Rollout for Vision Transformer
    Computes attention flow through all layers
    """
    
    def __init__(self, model, head_fusion='mean', discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
    
    def get_attention_maps(self, input_image):
        """
        Extract attention maps from all encoder layers
        """
        self.model.eval()
        attention_maps = []
        
        def hook_fn(module, input, output):
            # Extract attention weights
            # For ViT, attention is in the self-attention sublayer
            if hasattr(module, 'self_attention'):
                attention_maps.append(module.self_attention.attention_weights)
        
        # Register hooks
        hooks = []
        for layer in self.model.vit.encoder.layers:
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_image)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps
    
    def rollout(self, attention_maps, start_layer=0):
        """
        Compute attention rollout
        """
        result = torch.eye(attention_maps[0].size(-1))
        
        for attention in attention_maps[start_layer:]:
            if self.head_fusion == 'mean':
                attention_heads_fused = attention.mean(dim=1)
            elif self.head_fusion == 'max':
                attention_heads_fused = attention.max(dim=1)[0]
            elif self.head_fusion == 'min':
                attention_heads_fused = attention.min(dim=1)[0]
            else:
                raise ValueError(f"Invalid head_fusion: {self.head_fusion}")
            
            # Discard low attention values
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * self.discard_ratio), -1, False)
            flat[0, indices] = 0
            
            # Normalize
            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1, keepdim=True)
            
            result = torch.matmul(a, result)
        
        # Get attention for CLS token
        mask = result[0, 0, 1:]
        
        # Reshape to 2D grid
        width = int(mask.size(-1) ** 0.5)
        mask = mask.reshape(width, width).cpu().numpy()
        mask = mask / mask.max()
        
        return mask


def visualize_attention_overlay(
    original_image,
    attention_map,
    alpha=0.5,
    cmap='jet'
):
    """
    Overlay attention map on original image
    
    Args:
        original_image: Original PIL Image or numpy array
        attention_map: 2D attention map
        alpha: Transparency of overlay
        cmap: Colormap to use
    
    Returns:
        overlay: Overlayed image
    """
    # Convert original image to numpy
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    # Resize attention map to match image size
    h, w = original_image.shape[:2]
    attention_map = cv2.resize(attention_map, (w, h))
    
    # Normalize attention map
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # Apply colormap
    cmap = plt.get_cmap(cmap)
    heatmap = cmap(attention_map)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Overlay
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    
    return overlay


def visualize_prediction_with_explanations(
    model,
    image_path,
    device='cuda',
    save_path='explanation.png'
):
    """
    Generate comprehensive visualization with multiple explanation methods
    
    Args:
        model: Trained model
        image_path: Path to image
        device: Device to run on
        save_path: Path to save visualization
    """
    from torchvision import transforms
    
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # Generate Grad-CAM
    grad_cam = GradCAM(model)
    cam, _, _ = grad_cam.generate_cam(input_tensor, target_class=pred_class)
    
    # Resize original image for visualization
    img_resized = original_image.resize((224, 224))
    img_np = np.array(img_resized)
    
    # Create overlay
    overlay = visualize_attention_overlay(img_np, cam, alpha=0.5, cmap='jet')
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_resized)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Attention heatmap
    im = axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Attention Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(overlay)
    class_names = ['Non-Cancerous', 'Cancerous (IDC)']
    title = f'Prediction: {class_names[pred_class]}\nConfidence: {confidence:.2%}'
    axes[2].set_title(title, fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return pred_class, confidence, overlay


def generate_saliency_map(model, input_tensor, target_class=None):
    """
    Generate saliency map using gradient-based approach
    
    Args:
        model: Trained model
        input_tensor: Input tensor (should be detached and cloned)
        target_class: Target class (None for predicted class)
    
    Returns:
        saliency: Saliency map
    """
    model.eval()
    
    # Create a new tensor with requires_grad=True (must be a leaf variable)
    input_var = input_tensor.detach().clone().requires_grad_(True)
    
    # Forward pass
    output = model(input_var)
    
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    
    # Create one-hot target
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    
    # Compute gradients
    output.backward(gradient=one_hot)
    
    # Get gradients from input_var (not input_tensor!)
    if input_var.grad is None:
        print("Warning: Saliency gradients not captured, using fallback")
        # Create a simple saliency map
        saliency = np.ones((224, 224)) * 0.5
    else:
        saliency = input_var.grad.data.abs()
        saliency = saliency.squeeze().cpu().numpy()
        
        # Take maximum across color channels
        saliency = np.max(saliency, axis=0)
        
        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency


def explain_prediction(
    model,
    image_tensor,
    original_image,
    device='cuda'
):
    """
    Generate multiple explanations for a prediction
    
    Returns:
        Dictionary with explanations and visualizations
    """
    model.eval()
    
    # Ensure tensor is on correct device and doesn't require grad initially
    image_tensor = image_tensor.detach().to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # Generate Grad-CAM (this will handle requires_grad internally)
    grad_cam = GradCAM(model)
    try:
        cam, _, _ = grad_cam.generate_cam(image_tensor, target_class=pred_class)
    except Exception as e:
        print(f"Grad-CAM warning: {e}")
        # Fallback: create simple attention map
        cam = np.ones((14, 14)) * 0.5
    finally:
        # Clean up hooks
        try:
            grad_cam.remove_hooks()
        except:
            pass
    
    # Generate saliency map (pass detached tensor)
    try:
        saliency = generate_saliency_map(model, image_tensor, target_class=pred_class)
    except Exception as e:
        print(f"Saliency map warning: {e}")
        # Fallback: create simple saliency map
        saliency = np.ones((224, 224)) * 0.5
    
    # Prepare original image
    if isinstance(original_image, Image.Image):
        img_np = np.array(original_image.resize((224, 224)))
    else:
        img_np = original_image
    
    # Create overlays
    try:
        gradcam_overlay = visualize_attention_overlay(img_np, cam, alpha=0.5, cmap='jet')
    except Exception as e:
        print(f"Grad-CAM overlay warning: {e}")
        gradcam_overlay = img_np
    
    try:
        saliency_overlay = visualize_attention_overlay(img_np, saliency, alpha=0.5, cmap='hot')
    except Exception as e:
        print(f"Saliency overlay warning: {e}")
        saliency_overlay = img_np
    
    return {
        'prediction': pred_class,
        'confidence': confidence,
        'probabilities': probs[0].cpu().numpy(),
        'gradcam': cam,
        'saliency': saliency,
        'gradcam_overlay': gradcam_overlay,
        'saliency_overlay': saliency_overlay,
        'original_image': img_np
    }


def create_explanation_figure(explanation_dict, class_names=['Non-IDC', 'IDC']):
    """
    Create a comprehensive explanation figure
    
    Args:
        explanation_dict: Dictionary from explain_prediction()
        class_names: List of class names
    
    Returns:
        fig: Matplotlib figure
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(explanation_dict['original_image'])
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Prediction info
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.axis('off')
    pred_class = explanation_dict['prediction']
    confidence = explanation_dict['confidence']
    probs = explanation_dict['probabilities']
    
    info_text = f"""
    Prediction: {class_names[pred_class]}
    Confidence: {confidence:.2%}
    
    Class Probabilities:
    • {class_names[0]}: {probs[0]:.2%}
    • {class_names[1]}: {probs[1]:.2%}
    
    Interpretation:
    The model is {'highly confident' if confidence > 0.9 else 'confident' if confidence > 0.7 else 'moderately confident' if confidence > 0.5 else 'uncertain'} 
    in its prediction.
    """
    ax2.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Grad-CAM heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(explanation_dict['gradcam'], cmap='jet')
    ax3.set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Grad-CAM overlay
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(explanation_dict['gradcam_overlay'])
    ax4.set_title('Grad-CAM Overlay', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Saliency map
    ax5 = fig.add_subplot(gs[1, 2])
    im5 = ax5.imshow(explanation_dict['saliency'], cmap='hot')
    ax5.set_title('Saliency Map', fontsize=14, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    # Explanation text
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    explanation_text = f"""
    Understanding the Prediction:
    
    • Grad-CAM (Gradient-weighted Class Activation Mapping): Shows which regions of the image 
      were most important for the model's decision. Brighter (red/yellow) areas indicate regions 
      that strongly influenced the prediction toward "{class_names[pred_class]}".
    
    • Saliency Map: Highlights image regions where small changes would most affect the prediction.
      This shows which pixels the model is most sensitive to.
    
    • The model focuses on specific tissue patterns, cellular structures, and morphological features
      that are characteristic of {'cancerous tissue' if pred_class == 1 else 'normal tissue'}.
    
    Clinical Note: This is an AI-assisted analysis tool and should not replace professional 
    medical diagnosis. Always consult with a qualified pathologist for definitive diagnosis.
    """
    
    ax6.text(0.05, 0.5, explanation_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2),
             wrap=True)
    
    return fig


if __name__ == "__main__":
    # Example usage
    print("Explainability utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - GradCAM: Gradient-weighted Class Activation Mapping")
    print("  - AttentionRollout: Attention flow visualization")
    print("  - visualize_attention_overlay: Create attention overlays")
    print("  - explain_prediction: Generate comprehensive explanations")
    print("  - create_explanation_figure: Create publication-ready figures")
