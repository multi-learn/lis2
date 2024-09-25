import torch
import matplotlib

from PNRIA.configs.config import load_yaml
from PNRIA.torch_c.models.custom_model import BaseModel
from PNRIA.torch_c.optim import BaseOptimizer
from PNRIA.torch_c.scheduler import BaseScheduler

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# Create a random input tensor
batch_size = 1
height = 32  # Height of each slice
width = 32

# Generate random input data
random_input = torch.randn(2, 2, height, width)

# Generate random positional data (for the encoder)
# random_position = torch.randn(batch_size, 2 ,height, width)  # 2 here corresponds to the number of variables

config = load_yaml("configs/config_model_unet.yml")

# Load the model from the configuration
model = BaseModel.from_config(config['model'])
optim = BaseOptimizer.from_config(config['optimizer'], model.parameters())
scheduler = BaseScheduler.from_config(config['scheduler'], optim, steps_per_epoch=1)

# model =  UNet2()
model.eval()
# print(model)
# Move to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
random_input = random_input.to(device)
# random_position = random_position.to(device)

# Perform inference
with torch.no_grad():
    # Preprocess the input with positional encoding if applicable
    output = model(random_input)

# Print shapes to verify
print(f"Input Shape: {random_input.shape}")
print(f"Output Shape: {output.shape}")

print( "optim : ", optim)
print( "Scheduled : ", scheduler)
