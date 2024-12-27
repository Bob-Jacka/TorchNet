import os
from os.path import exists

import torch
import torch.cuda
import torchvision
from PIL import Image
from torch import nn
from torch.ao.nn.quantized import Softmax
from torch.nn import Sequential, Dropout, BCELoss
from torch.optim import Optimizer
from torchvision import transforms
from torchvision.transforms import Lambda


class torch_nn(nn.Module):
    model: Sequential
    optimizer: Optimizer
    loss_fn: BCELoss
    data_loader: tuple
    model_name: str = 'model'

    ext: str = '.pth'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        super().__init__()
        self.transform_func2 = Lambda(lambda y: torch.zeros(
            10, dtype=torch.float32).scatter_(dim=0, index=torch.tensor(y), value=1))
        self.transform_func3 = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.create_model_dense()
        self.create_optim()

    def create_model(self, dropout_rate=0.25, is_static: bool = False):
        self.model = Sequential(
            nn.Conv2d(784, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Linear(32, 16),
            nn.Linear(8, 2),
            Dropout(p=dropout_rate),
            Softmax()).to(self.device)
        if is_static:
            return self.model

    def create_model_dense(self):
        self.model = Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1),
            Softmax()
        ).to(self.device)

    def create_optim(self, learning_rate=0.001):
        learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = BCELoss()

    def get_data_loaders(self, batch_size=64, shuffle_sets=True, isdownload_dataset=False, root: str = '.'):
        """
        first binary_train_loader,
        second binary_test_loader
        :return: binary_train, binary_test
        """
        try:
            train_set = torchvision.datasets.MNIST(
                root=root,
                train=True,
                download=isdownload_dataset,
                transform=self.transform_func3)

            test_set = torchvision.datasets.MNIST(root=root,
                                                  train=False, download=isdownload_dataset,
                                                  transform=self.transform_func3)
            binary_train_set = [x for x in train_set if x[1] in [0, 9]]
            binary_test_set = [x for x in test_set if x[1] in [0, 9]]
            binary_train_loader = torch.utils.data.DataLoader(
                binary_train_set,
                batch_size=batch_size,
                shuffle=shuffle_sets)
            binary_test_loader = torch.utils.data.DataLoader(
                binary_test_set,
                batch_size=batch_size, shuffle=shuffle_sets)
            return binary_train_loader, binary_test_loader

        except RuntimeError:
            print("Datasets are not loaded, set True to download")

    def train_model(self, data_loader, epochs=50, after_train_save: bool = False):
        if self.model is not None:
            for epoch in range(epochs):
                running_loss = 0.0
                for images, labels in data_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                    print(f'Epoch [{epoch + 1}/{epoch}], Loss: {running_loss / len(data_loader):.4f}')
        if after_train_save:
            self.save_model(get_save_path())
        else:
            print("Load model first")

    def save_model(self, save_path: str):
        path = save_path + self.model_name + self.ext
        torch.save(self.model.state_dict(), path)
        print("model save on path ", path)

    def load_model(self, load_path: str):
        if exists(load_path) and len(os.listdir(load_path)) != 0:
            state_dict = torch.load(load_path + self.model_name + self.ext, weights_only=True)
            loaded_model = torch.nn.Module()  ##torch_nn()
            if 'module.' in next(iter(state_dict.keys())):
                state_dict = {k.replace('module.', ''):
                                  v for k, v in state_dict.items()}
            loaded_model.load_state_dict(state_dict, strict=False)
            loaded_model.eval()
            print("model loaded and ready")
            return loaded_model
        else:
            print("Model is not exist")

    def predict(self, image_path: str):
        print("Net thinks that ")
        with torch.no_grad():
            image = Image.open(image_path)
            image = self.transform_func3(image).unsqueeze(0)  # Добавление размерности батча
            output = self.model(image)
            _, predicted = torch.max(output.data, 1)
            return predicted.item()


def get_save_path():
    return 'core/saveModel/'
