import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import backbones
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class TransferModel(nn.Module):
    def __init__(self, num_class, base_net='resnet152', use_bottleneck=True, bottleneck_width=256):
        super(TransferModel, self).__init__()
        self.num_class = num_class
        self.base_network = backbones.get_backbone(base_net)
        self.use_bottleneck = use_bottleneck

        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()

        self.classifier_layer = nn.Linear(feature_dim, num_class)

    def forward(self, x):
        features = self.base_network(x)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        outputs = self.classifier_layer(features)
        return outputs

class Target(Dataset):
    def __init__(self, csv_file, root_dir, split, split_column, transform=None, dataset_type='F1'):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.split = split
        self.split_column = split_column
        self.transform = transform
        self.dataset_type = dataset_type
        self.data_frame = self.data_frame[self.data_frame[self.split_column] == self.split]
        self.class_names = pd.Categorical(self.data_frame['taxon']).categories.tolist()
        self.num_classes = len(self.class_names)
        self.data_frame['label'] = pd.Categorical(self.data_frame['taxon']).codes

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if self.dataset_type == 'F1':
            taxon = self.data_frame.iloc[idx, 0]
            image_name = self.data_frame.iloc[idx, 2]
            img_path = os.path.join(self.root_dir, taxon, image_name)
            label = self.data_frame.iloc[idx, -1]
        else:  # F2
            individual = self.data_frame.iloc[idx, 0]
            taxon = self.data_frame.iloc[idx, 1]
            image_name = self.data_frame.iloc[idx, 6]
            img_path = os.path.join(self.root_dir, individual, image_name)
            label = self.data_frame['label'].iloc[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    def get_num_classes(self):
        return self.num_classes

def test(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy, cm, np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(cm, class_names, save_path, accuracy):
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Main execution
if __name__ == "__main__":
    # Define paths and parameters
    data_dir = '/home/amirkh/Python/data/Detect dataset/Cropped images'
    csv_file = '/home/amirkh/Python/Main/CSV/Fin1-3(6).csv'
    da_model_path = '/home/amirkh/Python/Main/adapted-model/final_model_6.pt'
    cm_save_path = '/home/amirkh/Python/Main/Testing_matrix.png'
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up data transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Set up test dataset and dataloader
    split_column = '0'
    test_dataset = Target(csv_file=csv_file, root_dir=data_dir, split='test', split_column=split_column, transform=tf, dataset_type='F1')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=32)
    
    # Set up the model
    model = TransferModel(num_class=test_dataset.num_classes, base_net='resnet152', use_bottleneck=True, bottleneck_width=256)
    model = model.to(device)
    
    # Load pre-trained weights
    transfernet_state_dict = torch.load(da_model_path, map_location=device)
    model.load_state_dict(transfernet_state_dict, strict=False)
    print("Model loaded successfully.")
    
    # Test the model
    accuracy, cm, true_labels, pred_labels = test(model, test_loader, device)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(cm, test_dataset.class_names, cm_save_path, accuracy)
    print(f"Confusion matrix saved to {cm_save_path}")

    # Print class-wise statistics
    print("\nClass-wise Statistics:")
    for i, class_name in enumerate(test_dataset.class_names):
        class_total = np.sum(true_labels == i)
        class_correct = np.sum((true_labels == i) & (pred_labels == i))
        
        print(f"\nClass: {class_name}")
        print(f"  Total images: {class_total}")
        print(f"  Correctly classified: {class_correct}")
        print(f"  Accuracy: {100 * class_correct / class_total:.2f}%")