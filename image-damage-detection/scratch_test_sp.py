import torch
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
from PIL import Image
import numpy as np

print("Loading processor and model...")
processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
model.eval()

# Create a dummy image
img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
pil_image = Image.fromarray(img)

inputs = processor(images=[pil_image], return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

print("Outputs keys:", outputs.keys())
for key, value in outputs.items():
    if torch.is_tensor(value):
        print(f"  {key}: tensor of shape {value.shape}")
    elif isinstance(value, tuple):
        print(f"  {key}: tuple of length {len(value)}")
        for i, item in enumerate(value):
            if torch.is_tensor(item):
                print(f"    item {i}: tensor of shape {item.shape}")
    else:
        print(f"  {key}: {type(value)}")
