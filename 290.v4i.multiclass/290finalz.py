import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

# define the Dataset class
class MultiLabelDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Extract filenames and labels
        self.image_filenames = self.data['filename'].values  
        self.labels = self.data.iloc[:, 1:].values.astype('float32')  # all other columns are labels

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert('RGB')  

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(self.labels[idx])

        return image, labels

# define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_csv = '/Users/kabeermakkar/Desktop/290.v4i.multiclass/train/_classes.csv'
train_dir = '/Users/kabeermakkar/Desktop/290.v4i.multiclass/train'

val_csv = '/Users/kabeermakkar/Desktop/290.v4i.multiclass/valid/_classes.csv'
val_dir = '/Users/kabeermakkar/Desktop/290.v4i.multiclass/valid'

# Initialize datasets and dataloaders
train_dataset = MultiLabelDataset(csv_file=train_csv, img_dir=train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = MultiLabelDataset(csv_file=val_csv, img_dir=val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# define the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, train_dataset.labels.shape[1])  # output size = number of labels
model = model.to(device)

# define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

# save the trained model
torch.save(model.state_dict(), '/Users/kabeermakkar/Desktop/multi_label_model.pth')

# validation loop
model.eval()
with torch.no_grad():
    val_loss = 0.0
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

# prediction function
def predict(image_path, model, transform, class_labels):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        probabilities = torch.sigmoid(logits)  
        predicted_classes = (probabilities > 0.5).int().squeeze()  # threshold at 0.5

    predictions = [class_labels[i] for i, val in enumerate(predicted_classes) if val == 1]
    return predictions

# define class labels from the CSV file columns
class_labels = list(train_dataset.data.columns[1:])  # Exclude "filename"

# Test the prediction function
# test_image_path = '/Users/kabeermakkar/Desktop/290.v4i.multiclass/test_images/avocado.jpeg'
# predictions = predict(test_image_path, model, transform, class_labels)
# print(f"Predictions for {test_image_path}: {predictions}")
