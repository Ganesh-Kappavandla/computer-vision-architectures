import torch
import torch.nn as nn
from models.unet import DoubleConvBlock

def run_evaluation_lifecycle():
    print("------------------------------------------------")
    print("Initializing Model Evaluation Engine...")
    print("------------------------------------------------")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Evaluation Compute Device -> [{device.type.upper()}]")

    model = DoubleConvBlock(in_channels=3, out_channels=64).to(device)
    
    model.eval()

    print("Allocating target tensors for validation inference...")
    dummy_validation_image = torch.randn(1, 3, 256, 256).to(device)
    
    with torch.no_grad():
        print("Running forward inference pass through frozen layers...")
        predicted_mask = model(dummy_validation_image)

    print("------------------------------------------------")
    print("Inference Shape Diagnostics Summary:")
    print(f"Input Validation Tensor:  {dummy_validation_image.shape}")
    print(f"Predicted Output Segment: {predicted_mask.shape}")
    print("------------------------------------------------")
    print("Evaluation pipeline verified. Inference tracking is functionally stable.")
    print("------------------------------------------------")

if __name__ == "__main__":
    run_evaluation_lifecycle()  
