import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------
# Prunable Linear Layer
# -----------------------------------------------------------
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores
        self.gate_scores = nn.Parameter(
            torch.randn(out_features, in_features)
        )

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)


# -----------------------------------------------------------
# Network
# -----------------------------------------------------------
class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# -----------------------------------------------------------
# ✅ FIXED Sparsity Loss (MEAN, not SUM)
# -----------------------------------------------------------
def sparsity_loss(model):
    total_gates = 0
    total_elements = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total_gates += gates.sum()
            total_elements += gates.numel()

    return total_gates / total_elements   # 🔥 KEY FIX


# -----------------------------------------------------------
# Sparsity Calculation
# -----------------------------------------------------------
def calculate_sparsity(model, threshold=0.05):
    total = 0
    zero = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.numel()
            zero += (gates < threshold).sum().item()

    return 100 * zero / total


# -----------------------------------------------------------
# Evaluation
# -----------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


# -----------------------------------------------------------
# Training
# -----------------------------------------------------------
def train_model(lambda_val, train_loader, test_loader, epochs=10):
    model = PrunableNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            ce_loss = criterion(outputs, labels)
            sp_loss = sparsity_loss(model)

            loss = ce_loss + lambda_val * sp_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}")

    accuracy = evaluate(model, test_loader)
    sparsity = calculate_sparsity(model)

    return model, accuracy, sparsity


# -----------------------------------------------------------
# Gate Distribution Plot
# -----------------------------------------------------------
def plot_gate_distribution(model):
    all_gates = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            all_gates.extend(gates.detach().cpu().numpy().flatten())

    plt.hist(all_gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate values")
    plt.ylabel("Frequency")
    plt.show()


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))  # ✅ correct for CIFAR-10
    ])

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    lambdas = [1e-5, 1e-4, 1e-3]

    results = []

    for lam in lambdas:
        print(f"\n==============================")
        print(f"Training with lambda = {lam}")
        print(f"==============================")

        model, acc, sparsity = train_model(lam, train_loader, test_loader)

        print(f"\nLambda: {lam}")
        print(f"Test Accuracy: {acc:.2f}%")
        print(f"Sparsity: {sparsity:.2f}%")

        results.append((lam, acc, sparsity))

    plot_gate_distribution(model)

    print("\nFinal Results:")
    for r in results:
        print(f"Lambda: {r[0]}, Accuracy: {r[1]:.2f}%, Sparsity: {r[2]:.2f}%")


if __name__ == "__main__":
    main()