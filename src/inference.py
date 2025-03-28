# inference.py
import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = models.resnet18(pretrained=False)  # pretrained=False for inference
        self.features.fc = nn.Linear(self.features.fc.in_features, num_classes)

    def forward(self, x):
        return self.features(x)

# Step 2: Define model_fn for SageMaker
def model_fn(model_dir):
    model = SimpleCNN(num_classes=2)
    model_path = os.path.join(model_dir, "model.pth")
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def input_fn(request_body, content_type):
    if content_type == 'application/x-npy':
        try:
            np_array = np.frombuffer(request_body, dtype=np.float32).reshape(1, 3, 128, 128)
            return torch.tensor(np_array)
        except Exception as e:
            raise ValueError(f"❌ Invalid input format: {e}")
    else:
        raise ValueError(f"❌ Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    with torch.no_grad():
        output = model(input_data)
        prob = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(prob).item()
        confidence = prob[pred].item()
        labels = {0: "Real", 1: "Fake"}
        return {"prediction": labels[pred], "confidence": confidence}
