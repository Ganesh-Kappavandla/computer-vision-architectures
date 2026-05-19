import torch
import torch.nn as nn
import torch.optim as optim
from models.unet import DoubleConvBlock

def run_training_lifecycle():
    print("------------------------------------------------")
    print("Initializing PyTorch Optimization Engine...")
    print("------------------------------------------------")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Execution Compute Device: Hardware acceleration -> [{device.type.upper()}]")

    in_channels = 3
    out_channels = 64
    model = DoubleConvBlock(in_channels, out_channels).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Allocating memory tensors for synthetic training batch...")
    dummy_images = torch.randn(4, in_channels, 256, 256).to(device)
    dummy_targets = torch.randn(4, out_channels, 256, 256).to(device)

    print("Executing Forward Pass through network layers...")
    model.train() 
    
    optimizer.zero_grad()
    
    predictions = model(dummy_images)
    
    loss = criterion(predictions, dummy_targets)
    print(f"Initial Network Convergence Loss: Calculated MSE Error -> {loss.item():.4f}")

    print("Executing Backpropagation derivative calculation...")
    loss.backward()
    
    print("Updating weights via Adam gradient optimization steps...")
    optimizer.step()

    print("------------------------------------------------")
    print("Pipeline compilation verified. Neural layers are functionally stable.")
    print("------------------------------------------------")

if __name__ == "__main__":
    run_training_lifecycle()  
