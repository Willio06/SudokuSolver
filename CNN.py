import openml
import os
import json
import random
import math
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim

import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
import torchvision.transforms.v2 as transforms
import openml_pytorch as opt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

skip_training = False

class ModelLogger(pl.Callback):
    """ Helper class to log metrics and save models in a systematic way.
        Don't modify this class unless you have a sound reason to do so.
        
    Attributes:
        output_dir (str) : Directory to save model and metrics (base_dir by default)
        model_name (str) : Name of the model. Should be model_1 for your final model in Question 1.
        use_half (bool)  : Whether to save model in half precision. This reduces file size but is 
                           less accurate and not ideal for continued training.
        device (str)     : Device to save the model to (cpu, cuda, mps).
    """
    def __init__(self, model_name="model", test_loader=None, output_dir="./", use_half=False, device="cpu"):
        super().__init__()
        self.output_dir = output_dir
        self.model_name = model_name
        self.test_loader = test_loader
        self.use_half = use_half
        self.device = device
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
        self.current_val_loss = 0.0
        self.current_val_acc = 0.0
        os.makedirs(output_dir, exist_ok=True)

    # Logs the loss and accuracy for each epoch
    def log_epoch(self, train_loss, val_loss, train_acc=None, val_acc=None):
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(train_acc if train_acc is not None else 0.0)
        self.history["val_acc"].append(val_acc if val_acc is not None else 0.0)

    # Saves the model weights to a file
    def save_model(self, model):
        path = os.path.join(self.output_dir, f"{self.model_name}.pt")
        model_cpu = model.to("cpu")
        state_dict = model_cpu.state_dict()

        if self.use_half:
            state_dict = {k: v.half() if v.dtype == torch.float32 else v for k, v in state_dict.items()}
        torch.save(state_dict, path)
        model.to(self.device)

    # Loads the stored weights into the given model
    def load_model(self, model):
        path = os.path.join(self.output_dir, f"{self.model_name}.pt")
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model.to(self.device)

    # Saves the metrics (learning curve) to a file
    def save_metrics(self):
        path = os.path.join(self.output_dir, f"{self.model_name}_metrics.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    # Loads the metrics (learning curve) from a file
    def load_metrics(self):
        path = os.path.join(self.output_dir, f"{self.model_name}_metrics.json")
        with open(path, "r") as f:
            self.history = json.load(f)
        if "test_acc" in self.history:
            print(f"Test Accuracy: {self.history['test_acc']:.4f}")

    # Plots the learning curves based on the stored metrics
    def plot_learning_curves(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)
        plt.figure(figsize=(8, 4))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["train_loss"], label='Train Loss')
        plt.plot(epochs, self.history["val_loss"], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history["train_acc"], label='Train Accuracy')
        plt.plot(epochs, self.history["val_acc"], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()

        plt.tight_layout()
        filename = os.path.join(self.output_dir, f"{self.model_name}_learning_curves.png")
        plt.savefig(filename)
        plt.show()

    # Returns the best (smoothed) validation accuracy and the epoch at which it was recorded
    def get_best_validation_accuracy(logger, window_size=3):
        val_accuracies = logger.history["val_acc"]
        
        if len(val_accuracies) < window_size:
            # Not enough data points, return the max
            return max(val_accuracies), val_accuracies.index(max(val_accuracies))
        
        # Calculate moving average
        moving_averages = []
        for i in range(len(val_accuracies) - window_size + 1):
            window = val_accuracies[i:i + window_size]
            moving_averages.append(sum(window) / window_size)
        
        # Find best moving average
        best_avg = max(moving_averages)
        best_window_end = moving_averages.index(best_avg) + window_size - 1
        
        print(f"Best validation accuracy: {best_avg:.4f} at epoch {best_window_end} (smoothed over {window_size} epochs)")

    @torch.no_grad()
    def evaluate_on_testset(self, model):
        try:
            model = model.to(self.device)
            model.eval()

            total_correct, total_samples = 0, 0
            for batch in tqdm(self.test_loader, desc="Evaluating on test data", leave=False):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                logits = model(x)
                preds = logits.detach().cpu().argmax(dim=1)
                y_cpu = y.detach().cpu()
                total_correct += (preds == y_cpu).sum().item()
                total_samples += y_cpu.size(0)

            accuracy = total_correct / total_samples
            self.history["test_acc"] = accuracy
            print(f"Test Accuracy: {accuracy:.4f}")
        except RuntimeError as e:
            print(f"Could not evaluate on test set. Is the model architecture correct?\n \033[91m Error: {e} \033[0m ")

    # At the end of a training run, this stores the model weights and metrics and returns a metric plot
    def finalize(self, model, testset=True):
        if testset:
            self.evaluate_on_testset(model)
        self.save_metrics()
        self.save_model(model)
        self.plot_learning_curves()
        self.get_best_validation_accuracy()

    # Reports on the metric of a trained model
    def report(self):
        self.load_metrics()
        self.plot_learning_curves()
        self.get_best_validation_accuracy()

    # --- PyTorch Lightning integration ---

    # Callback function to store the training loss and accuracy at the end of each epoch
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return # Don't log anything during a sanity check
        
        # Store the train metrics at the end of training epoch
        train_loss = float(trainer.callback_metrics.get("train_loss", 0.0))
        train_acc = float(trainer.callback_metrics.get("train_acc", 0.0))
        # Save these for later use when validation completes
        self.current_train_loss = train_loss
        self.current_train_acc = train_acc

        # Use the stored training metrics
        # on_validation_epoch_end runs first, so we can use the stored values here
        self.log_epoch(train_loss, self.current_val_loss, train_acc, self.current_val_acc)
    
    # Callback function to store the validation loss and accuracy at the end of each epoch
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return # Don't log anything during a sanity check
        
        # Get validation metrics from trainer's callback_metrics
        val_loss = float(trainer.callback_metrics.get("val_loss", 0.0))
        val_acc = float(trainer.callback_metrics.get("val_acc", 0.0))

        self.current_val_loss = val_loss
        self.current_val_acc = val_acc
        
    # Callback to store the model and metrics at the end of training
    def on_train_end(self, trainer, pl_module):
        self.finalize(pl_module)

class Loader():
    def __init__(self, X, y, transform=None):
        self.transform = transform
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.X = torch.tensor(self.X.values).reshape(-1, 1, 28, 28)
        self.y = torch.from_numpy(self.y.values.astype(np.int64)).reshape(-1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x_item = self.X[idx]
        y_item = self.y[idx]
        if self.transform:
            x_item = self.transform(x_item)
        assert y_item.numel() > 0, f"Empty label at index {idx}"
        return x_item, y_item


torch.cuda.is_available()
MNIST = openml.datasets.get_dataset(554)
X_all, y_all, categorical_indicator, attribute_names = MNIST.get_data(target=MNIST.default_target_attribute)
X_all = X_all.astype(float)/255.0  # Normalize the dataset to [0, 1]

noise = np.random.normal(loc=0.0, scale=0.1, size=(4000, 28*28))
noise = np.clip(noise, 0, 1)  # Keep values in valid pixel range
X_new = pd.DataFrame(noise)
X_new.columns = [f"pixel{i+1}" for i in range(28*28)]
Y_new = pd.Series([10] * 4000).astype(object)

X_all = pd.concat([X_all, X_new], ignore_index=True)
y_all = pd.concat([y_all, Y_new], ignore_index=True)

y_all = y_all.astype(int)
print("The dataset has {} images of numbers and {} classes".format(X_all.shape[0], len(np.unique(y_all))))
label_to_idx = {label: idx for idx, label in enumerate(sorted(y_all.unique()))}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
y_all = y_all.map(label_to_idx)

# Train-test split and data loaders
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

transformss = transforms.Compose([
    transforms.Lambda(lambda x: x.type(torch.float32)),  # Convert to float32
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random affine transformation
    transforms.GaussianNoise(0.1)  # Add Gaussian noise
])
train_loader = Loader(X_train, y_train, transform=transformss)
val_loader = Loader(X_val, y_val)
test_loader = Loader(X_test, y_test)

def show_sample_images(dataset, num_samples=10):
    """
    Show sample images from a PandasDataset.

    Args:
        dataset (PandasDataset): Dataset with 784-dim flattened images in X.
        num_samples (int): Number of images to display.
    """
    indices = random.sample(range(len(dataset)), num_samples)
    images, labels = zip(*[dataset[i] for i in indices])
    # Stack and reshape images (assumes transform_X returns torch tensors)
    images = [img.numpy().reshape((28,28)) for img in images]
    
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap='plasma')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

show_sample_images(train_loader)
torch.set_float32_matmul_precision('medium')


class MNIST_ResNet(pl.LightningModule):
    def __init__(self, finetune_mode="full", classes=10):
        super().__init__()
        self.save_hyperparameters()

        # Load full pre-trained ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_dim = self.resnet.fc.in_features
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Replace FC head with your own
        self.resnet.fc = nn.Identity()  # We'll do classification separately
        self.classifier = nn.Sequential(nn.BatchNorm1d(self.feature_dim),  # Add batch normalization
                                        nn.Dropout(0.5,inplace=True),
                                        nn.Linear(self.feature_dim, 50),
                                        nn.BatchNorm1d(50),  # Add batch normalization
                                        nn.Dropout(0.2,inplace=True),
                                        nn.Linear(50, classes))
        # Apply fine-tuning strategy
        self.finetune_mode = finetune_mode
        self.freeze_layers()

    def freeze_layers(self):
        # Freeze all
        for param in self.resnet.parameters():
            param.requires_grad = False

        if self.finetune_mode == "last_block":
            # Unfreeze last block (layer4)
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True
        elif self.finetune_mode == "full":
            for param in self.resnet.parameters():
                param.requires_grad = True
        # else 'head': leave everything except classifier frozen

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)  # not usually necessary, but valid

        features = self.resnet(x)
        logits = self.classifier(features)
        return logits.squeeze(1)

# Create your model
# Define how you train your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
def train_model_1(model,logger, criter, optimizer, scheduler, epochs=30):
    for epoch in range(epochs):
        trainLoss=[]
        trainAcc=[]
        valLoss=[]
        valAcc=[]
        model.train()
        train_loader_tqdm = tqdm(DataLoader(train_loader, batch_size=32, shuffle=True), desc="Training")  # Wrap the loader

        for batch_index, (data, targets) in enumerate(train_loader_tqdm):
            # Move data and targets to the device (GPU/CPU)
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            if data.dim() == 3:
                data = data.unsqueeze(1)  # shape: (B, H, W) → (B, 1, H, W)
            elif data.shape[1] != 1:
                data = data.unsqueeze(1)  # safer fallback

            # Forward pass: compute the model output
            out = model(data)     
            loss = criter(out, targets)
            
            preds = out.argmax(dim=1)
            correct = (preds == targets).sum().item()
            accuracy = correct / targets.size(0) * 100
            trainLoss.append(loss.item())
            #print(loss.item())
            trainAcc.append(accuracy)
            # Backward pass: compute the gradients

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            # Optimization step: update the model parameters
            optimizer.step()
            train_loader_tqdm.set_postfix(loss=loss.item(), accuracy=accuracy)
            # Train
            # Log
        print(f"TRAIN   Epoch {epoch+1}/{epochs} loss {np.mean(trainLoss)} and accuracy {np.mean(trainAcc)}")
        val_loader_tqdm = tqdm(DataLoader(val_loader, batch_size=32, shuffle=True), desc="Validation")  # Wrap the loader

        model.eval()
        for batch_index, (data, targets) in enumerate(val_loader_tqdm):
            # Move data and targets to the device (GPU/CPU)
            data = data.to(device)
            targets = targets.to(device)
            print(data.shape, targets.shape)
            if data.dim() == 3:
                data = data.unsqueeze(1)  # shape: (B, H, W) → (B, 1, H, W)
            elif data.shape[1] != 1:
                data = data.unsqueeze(1)  # safer fallback

            # Forward pass: compute the model output
            out = model(data)
            preds = out.argmax(dim=1)
            correct = (preds == targets).sum().item()
            loss = criter(out, targets).item()
            accuracy = correct / targets.size(0)*100
            valLoss.append(loss)
            valAcc.append(accuracy)
            val_loader_tqdm.set_postfix(loss=loss, accuracy=accuracy)
        print(f"VALIDATION  Epoch {epoch+1}/{epochs} loss {np.mean(valLoss)} and accuracy {np.mean(valAcc)}")

        val_loss = np.mean(valLoss)
        train_loss = np.mean(trainLoss)
        train_acc = np.mean(trainAcc)
        val_acc = np.mean(valAcc)
        logger.log_epoch(train_loss, val_loss, train_acc, val_acc)
        scheduler.step()
    # Save
    logger.finalize(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_ResNet(classes=11)
model=model.to(device)
# Create logger with model name
logger = ModelLogger("model_1", DataLoader(test_loader, batch_size=32, shuffle=True), device=device)
criter = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)#, weight_decay=0.001)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=20, steps_per_epoch=len(train_loader), pct_start=0.1, div_factor=10, final_div_factor=100,anneal_strategy="cos")

# This will load your data from file rather than training it. 
# Make sure to set skip_training to True before submitting.
if skip_training:
    model = logger.load_model(model)
    logger.report()
else:
    train_model_1(model, logger,criter, optimizer,scheduler, epochs=20)