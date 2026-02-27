import torch
import torch.nn as nn

class RecoveredBaselineModel(nn.Module):
    def __init__(self, input_dim=320, hidden_dim=1024, output_dim=1, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        p = self.proj(x)
        h = h + p
        h = self.relu(self.fc2(h))
        h = self.drop(h)
        return self.out(h)

# Load the original full object
model = torch.load("baseline_public_v1.pth", map_location="cpu", weights_only=False)

# Save ONLY the weights
torch.save(model.state_dict(), "baseline_state_dict.pth")

print("Successfully extracted state_dict -> baseline_state_dict.pth")
