import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# changing the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  FaceMaskDataset class for training, validation, and test data
class FaceMaskDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label


# FaceMaskDetector class
class FaceMaskDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path).to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_model(self, model_path):
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def detect(self, frame):
        frame = self.transform(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(frame)
        probabilities = torch.softmax(output, dim=1)[0]
        mask_probability = probabilities[1].item()
        no_mask_probability = probabilities[0].item()
        label = "Mask" if mask_probability > no_mask_probability else "No Mask"
        return label


# The training parameters
train_data_dir = "D:/another_try_mask/archive/Face Mask Dataset/Train"
val_data_dir = "D:/another_try_mask/archive/Face Mask Dataset/Validation"
test_data_dir = "D:/another_try_mask/archive/Face Mask Dataset/Test"
model_save_path = "model3.pt"
batch_size = 20
num_epochs = 20
# Load the training data
train_image_paths = []
train_labels = []

for folder_name in os.listdir(train_data_dir):
    folder_path = os.path.join(train_data_dir, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".png") or file_name.endswith(".jpg"):
                file_path = os.path.join(folder_path, file_name)
                train_image_paths.append(file_path)
                train_labels.append(1 if folder_name == "1" else 0)
# Load the validation data
val_image_paths = []
val_labels = []

for folder_name in os.listdir(val_data_dir):
    folder_path = os.path.join(val_data_dir, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".png") or file_name.endswith(".jpg"):
                file_path = os.path.join(folder_path, file_name)
                val_image_paths.append(file_path)
                val_labels.append(1 if folder_name == "1" else 0)
# Load the test data
test_image_paths = []
test_labels = []

for folder_name in os.listdir(test_data_dir):
    folder_path = os.path.join(test_data_dir, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".png") or file_name.endswith(".jpg"):
                file_path = os.path.join(folder_path, file_name)
                test_image_paths.append(file_path)
                test_labels.append(1 if folder_name == "1" else 0)
# FaceMaskDataset class for training, validation, and test data
train_dataset = FaceMaskDataset(
    train_image_paths, train_labels, transform=transforms.ToTensor()
)
val_dataset = FaceMaskDataset(
    val_image_paths, val_labels, transform=transforms.ToTensor()
)
test_dataset = FaceMaskDataset(
    test_image_paths, test_labels, transform=transforms.ToTensor()
)
#  training, validation, and test data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# Function to show a batch of images during training
import matplotlib.pyplot as plt


def show_batch(images, labels):
    fig, axes = plt.subplots(figsize=(12, 6), ncols=4)
    for i, ax in enumerate(axes):
        ax.imshow(images[i].permute(1, 2, 0))
        ax.set_title("With Mask" if labels[i] == 1 else "Without Mask")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    # FaceMaskModel class


class FaceMaskModel(nn.Module):
    def __init__(self):
        super(FaceMaskModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.model(x)


# FaceMaskModel class
model = FaceMaskModel()
# the loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# Move the model to the device
model = model.to(device)

# list to save the train loss and accuracy for plotting
train_loss_values = []
train_accuracy_values = []


# Train the model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    train_accuracy = correct / total
    train_loss = running_loss / len(train_loader)

    train_loss_values.append(train_loss)  # Append train loss value to the list
    train_accuracy_values.append(
        train_accuracy
    )  # Append train accuracy value to the list

    # Model evaluation on the validation data
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_loss += loss.item()

    val_accuracy = val_correct / val_total
    val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy*100:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy*100:.2f}%")
    print("-------------------------------")


# Plotting train loss and accuracy

# list of epoch numbers
epochs = list(range(1, num_epochs + 1))

plt.figure(figsize=(10, 4))


# Plot train loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_values, label="Train Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
# plt.show()

# Plot train accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy_values, label="Train Accuracy", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()
# plt.show()


plt.tight_layout()
plt.show()

# Save the trained model
torch.save(model.state_dict(), model_save_path)

# Model evaluation on test data
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        test_loss += loss.item()

test_accuracy = test_correct / test_total
test_loss = test_loss / len(test_loader)

print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy*100:.2f}%")


# save the model one more time after checking good accuracy
torch.save(model.state_dict(), model_save_path)
