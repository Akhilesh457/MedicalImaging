"""
Quick Start Guide - Breast Cancer AI Classifier
This script helps you get started with the system
"""

import os
import sys

def print_header():
    """Print welcome header"""
    print("\n" + "="*70)
    print("🔬 EXPLAINABLE AI BREAST CANCER CLASSIFICATION SYSTEM")
    print("="*70 + "\n")

def check_dependencies():
    """Check if all required packages are installed"""
    print("📦 Checking dependencies...\n")
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'streamlit': 'Streamlit',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'matplotlib': 'Matplotlib',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn'
    }
    
    missing = []
    installed = []
    
    for module, name in required.items():
        try:
            __import__(module)
            installed.append(f"✅ {name}")
        except ImportError:
            missing.append(f"❌ {name}")
    
    for pkg in installed:
        print(pkg)
    
    if missing:
        print("\n⚠️  Missing packages:")
        for pkg in missing:
            print(pkg)
        print("\n📥 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed!\n")
        return True

def check_model():
    """Check if trained model exists"""
    print("🤖 Checking for trained model...\n")
    
    model_path = 'best_vit_idc_explainable.pth'
    
    if os.path.exists(model_path):
        print(f"✅ Model found: {model_path}")
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"   Size: {size_mb:.2f} MB\n")
        return True
    else:
        print(f"❌ Model not found: {model_path}")
        print("\n📝 To train the model:")
        print("   1. Download the dataset")
        print("   2. Update DATASET_PATH in train_explainable_vit.py")
        print("   3. Run: python train_explainable_vit.py")
        print("\n⏰ Training takes 1-2 hours with GPU, 8-12 hours with CPU\n")
        return False

def check_cuda():
    """Check CUDA availability"""
    print("🖥️  Checking compute capability...\n")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
        else:
            print("⚠️  CUDA not available - will use CPU")
            print("   Training and inference will be slower\n")
    except:
        print("❌ Could not check CUDA status\n")

def show_next_steps(has_model, has_deps):
    """Show next steps based on current state"""
    print("="*70)
    print("📋 NEXT STEPS")
    print("="*70 + "\n")
    
    if not has_deps:
        print("1️⃣  Install dependencies:")
        print("   pip install -r requirements.txt\n")
    
    if not has_model:
        print("2️⃣  Train the model:")
        print("   python train_explainable_vit.py\n")
        print("   OR download pre-trained model if available\n")
    
    if has_deps and has_model:
        print("🚀 You're ready to go!")
        print("\n▶️  Launch the Streamlit app:")
        print("   streamlit run app.py")
        print("\n🌐 The app will open at: http://localhost:8501")
        print("\n📖 Usage:")
        print("   1. Navigate to 'Upload & Analyze'")
        print("   2. Upload a histopathology image")
        print("   3. Click 'Analyze Image'")
        print("   4. Review results and visualizations")
    
    print("\n" + "="*70)
    print("📚 Documentation: README.md")
    print("❓ Issues: https://github.com/yourusername/project/issues")
    print("="*70 + "\n")

def main():
    """Main quick start function"""
    print_header()
    
    # Check dependencies
    has_deps = check_dependencies()
    
    # Check CUDA
    check_cuda()
    
    # Check model
    has_model = check_model()
    
    # Show next steps
    show_next_steps(has_model, has_deps)

if __name__ == "__main__":
    main()
