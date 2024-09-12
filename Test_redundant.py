import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import numpy as np
from torchvision import transforms, models
from models import TransferNet
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl


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
        self.num_classes = len(pd.Categorical(self.data_frame['taxon']).categories)


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

class ResNet152Lightning(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet152(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        return self(x)

def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model.predict(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, cm, np.array(all_labels), np.array(all_preds)

def plot_confusion_matrices(cm1, cm2, class_names, save_path, acc1, acc2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
    
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title(f'Domain Adaptation Model\nAccuracy: {acc1:.2f}%')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title(f'Conventional ResNet152\nAccuracy: {acc2:.2f}%')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    data_dir = '/home/amirkh/Python/data/Detect dataset/Cropped images'
    csv_file = '/home/amirkh/Python/Fin1_5splits_taxon.csv'
    data_dir2 = '/home/amirkh/Python/data/IDA/Images'
    csv_file2 = '/home/amirkh/Python/Main/CSV/Fin2(6).csv'
    da_model_path = '/home/amirkh/Python/Main/final_model_6.pt'
    conv_model_path = '/home/amirkh/Python/Main/Res152_6_E1_0.0001.pth'
    cm_save_path = '/home/amirkh/Python/Main/comparison_confusion_F2(6)_matrices.png'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    split_column = '0'

    test_dataset = Target(csv_file=csv_file2, root_dir=data_dir2, split='test', split_column=split_column, transform=tf_test, dataset_type='F2')
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=32)

    # Domain Adaptation Model
    da_model = TransferNet(num_class=test_dataset.num_classes, base_net='resnet152')
    da_model.load_state_dict(torch.load(da_model_path, map_location=device))
    da_model = da_model.to(device)

    # Conventional ResNet152
    conv_model = ResNet152Lightning(num_classes=test_dataset.num_classes)
    conv_model.model.load_state_dict(torch.load(conv_model_path, map_location=device))
    conv_model = conv_model.to(device)

    # Test both models
    da_accuracy, da_cm, da_true_labels, da_pred_labels = test_model(da_model, test_dataloader, device)
    conv_accuracy, conv_cm, conv_true_labels, conv_pred_labels = test_model(conv_model, test_dataloader, device)

    print(f"Domain Adaptation Model Accuracy: {da_accuracy:.2f}%")
    print(f"Conventional ResNet152 Accuracy: {conv_accuracy:.2f}%")

    # Plot and save confusion matrices
    plot_confusion_matrices(da_cm, conv_cm, test_dataset.class_names, cm_save_path, da_accuracy, conv_accuracy)
    print(f"Comparison confusion matrices saved to {cm_save_path}")

    # Print class-wise statistics for both models
    print("\nClass-wise Statistics:")
    for i, class_name in enumerate(test_dataset.class_names):
        da_total = np.sum(da_true_labels == i)
        da_correct = np.sum((da_true_labels == i) & (da_pred_labels == i))
        conv_total = np.sum(conv_true_labels == i)
        conv_correct = np.sum((conv_true_labels == i) & (conv_pred_labels == i))
        
        print(f"\nClass: {class_name}")
        print(f"Domain Adaptation Model:")
        print(f"  Total images: {da_total}")
        print(f"  Correctly classified: {da_correct}")
        print(f"  Accuracy: {100 * da_correct / da_total:.2f}%")
        print(f"Conventional ResNet152:")
        print(f"  Total images: {conv_total}")
        print(f"  Correctly classified: {conv_correct}")
        print(f"  Accuracy: {100 * conv_correct / conv_total:.2f}%")

if __name__ == "__main__":
    main()