import torch
from models import TransferNet  # Import the model architecture from your models.py

# Initialize the model
model = TransferNet(num_class=6)

# Load the state dictionary into the model
model_path = '/home/amirkh/Python/Main/adapted-model/final_model_6.pt'
model.load_state_dict(torch.load(model_path))

# Set model to evaluation mode (optional, depending on your use case)
model.eval()

# Print the architecture
print(model)
