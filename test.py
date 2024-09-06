import torch

# Load the model
model = torch.load('model/one.pkl')
model.eval()  # Set the model to evaluation mode

# Check if the model is loaded
if model:
    print("Model is loaded successfully")
    # Optionally, print model architecture
    print(model)
else:
    print("Model loading failed")
