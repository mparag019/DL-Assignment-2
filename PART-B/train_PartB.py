import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split, ConcatDataset
from torchvision.datasets import ImageFolder
import wandb
from wandb.sdk.wandb_run import Run
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse

wandb.login()

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy



# Training loop
def training_model(epochs, optimizer, criterion, model, train_loader, val_loader):
    for epoch in range(epochs):
        model.train()
        training_loss = 0.0
        train_accuracy = 0.0
        pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in train_loader:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device) 
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            train_accuracy += calculate_accuracy(outputs, labels)
            pbar.set_postfix({'Train Loss': training_loss / (pbar.n + 1), 'Train Acc': train_accuracy / (pbar.n + 1)})
            pbar.update(1)

        pbar.close()


        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device) 
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                val_accuracy += calculate_accuracy(outputs, labels)

        train_accuracy /= len(train_loader)
        training_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        print(f'Epoch {epoch+1}/{epochs}, Train_Loss: {training_loss:.4f},  Train_Acc: {train_accuracy:.4f},  Val_Loss: {val_loss:.4f},  Val_Accuracy: {val_accuracy:.4f}')
        wandb.log({"epoch": epoch+1, "train_loss": training_loss, "val_loss": val_loss, "val_accuracy": val_accuracy, "train_accuracy": train_accuracy})
    return model

def apply_additional_transforms(loader, additional_transforms, batch_size):
    transformed_dataset = []
    original_dataset = []
    pbar = tqdm(total=len(loader))
    for images, labels in loader:
        images1 = additional_transforms(images)
        for i in range(batch_size):
            original_dataset.append((images[i], labels[i]))
            transformed_dataset.append((images1[i], labels[i]))
        pbar.set_postfix()
        pbar.update(1)

    pbar.close()
    return original_dataset, transformed_dataset

def augment_data(data_augmentation, train_loader, batch_size):
    if data_augmentation:
        additional_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        ])

        # Apply additional transformations to the new DataLoader
        original_dataset, transformed_dataset = apply_additional_transforms(train_loader, additional_transforms, batch_size)
        combined_dataset = ConcatDataset([original_dataset, transformed_dataset])

        # Create a new DataLoader using the combined dataset
        combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    else:
        combined_loader = train_loader
    return combined_loader


def train_CNN(num_filters, activation, filter_organization, data_augmentation, batch_norm, dropout, batch_size, epochs, train_sampler, val_sampler,  strategy):
    
    # Load pre-trained model (ResNet50)
    model = torchvision.models.resnet50(pretrained=True)
    model.to(device)
    if strategy == 1:
    # Strategy 1: Freeze all layers except the last layer
        for param in model.parameters():
            param.requires_grad = False
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 101)  # 101 classes in iNaturalist
    elif strategy == 2:
    # Strategy 2: Freeze layers up to a certain depth
    # Freeze layers up to layer 4
        for name, param in model.named_parameters():
            if 'layer4' not in name:  # Freeze layers up to layer 4
                param.requires_grad = False
    else:
    # Strategy 3: Layer-wise fine-tuning
        for param in model.layer4.parameters():
            param.requires_grad = True
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create DataLoader instances for train and validation sets using the samplers
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(train_data, batch_size=batch_size, sampler=val_sampler)

    combined_loader = augment_data(data_augmentation, train_loader, batch_size)
    
    model = training_model(epochs, optimizer, criterion, model, combined_loader, val_loader)
    return model


def evaluate(test_data, batch_size, model):
    # Test the model
    test_loader = DataLoader(test_data, batch_size=batch_size)
    model.eval()

    test_accuracy = 0.0
    pbar = tqdm(total=len(test_loader))
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device) 
            outputs = model(images)
            images.to("cpu")
            labels.to("cpu")
            for i in range(len(images)):
                image = images[i]
                label = labels[i]
                output = outputs[i].argmax(dim = 0)
                if (label == output):
                    test_accuracy += 1
            pbar.set_postfix()
            pbar.update(1)

    pbar.close()
    test_accuracy /= len(test_data)
    wandb.log({"test_accuracy": test_accuracy})


def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser for Sweep Configuration")

    # Model Parameters
    parser.add_argument('--num_filters', type=int, choices=[32, 64, 128], required=False, default=128)
    parser.add_argument('--activation', type=str, choices=['ReLU', 'GELU', 'SiLU', 'Mish'], required=False, default ='GELU')
    parser.add_argument('--filter_organization', type=float, choices=[1, 2, 0.5], required=False, default = 1)
    parser.add_argument('--data_augmentation', type=str, choices=[True, False], required=False, default = False)
    parser.add_argument('--batch_normalization', type=str, choices=[True, False], required=False, default = True)
    parser.add_argument('--dropout', type=float, choices=[0.2, 0.3], required=False, default = 0.3)
    parser.add_argument('--epoch', type=int, choices=[5, 10], required=False, default = 10)
    parser.add_argument('--batch_size', type=float, choices=[32, 64, 128], required=False, default = 32)
    parser.add_argument('--strategy', type=float, choices=[1,2,3], required=False, default = 1)

    args = parser.parse_args()
    return args

def train(args):
    # Initialize wandb
    wandb.init(project="DL-Assignment-2" ,entity="cs23m047")
    epochs = args.epoch
    batch_size = args.batch_size
    num_filters = args.num_filters
    activation = args.activation
    filter_organization = args.filter_organization
    data_augmentation = args.data_augmentation
    batch_norm = args.batch_normalization
    dropout = args.dropout
    strategy = args.strategy

    wandb.run.name = f'num_filters_{num_filters}activation{activation}filter_organization{filter_organization}data_augmentation{data_augmentation}batch_normalization{batch_norm}dropout{dropout}epoch{epochs}'


    classes = ["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]
   # Check if CUDA (GPU) is available, and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Set up data transformations
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Load the dataset
    train_data = ImageFolder('C:/Parag/parag/IITM/Sem2/DL/Assignment2/nature_12K/inaturalist_12K/train', transform=train_transforms)
    test_data = ImageFolder('C:/Parag/parag/IITM/Sem2/DL/Assignment2/nature_12K/inaturalist_12K/val', transform=test_transforms)
    # Count the number of samples in each class
    class_counts = {}
    pbar = tqdm(total=len(train_data))
    for _, label in train_data:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
        pbar.set_postfix()
        pbar.update(1)

    pbar.close()

    # Calculate the number of samples per class for validation set
    val_size_per_class = {label: int(count * 0.2) for label, count in class_counts.items()}

    # Initialize lists to hold indices for train and validation sets
    train_indices = []
    val_indices = []

    # Iterate through the dataset and assign samples to train or validation set
    pbar = tqdm(total=len(train_data))
    for idx, (_, label) in enumerate(train_data):
        if val_size_per_class[label] > 0:
            val_indices.append(idx)
            val_size_per_class[label] -= 1
        else:
            train_indices.append(idx)
        pbar.set_postfix()
        pbar.update(1)

    pbar.close()

    # Create SubsetRandomSampler for train and validation sets
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    model = train_CNN(num_filters, activation, filter_organization, data_augmentation, batch_norm, dropout, batch_size, epochs, train_sampler, val_sampler, strategy)
    evaluate(test_data, model, batch_size)

args = parse_args()
train(args)
