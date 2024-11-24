import torch
import torch.cuda
import torchvision
# from torch import T
from torch import nn
from torch.nn import Sequential, Linear, ReLU, Dropout, Sigmoid, BCELoss
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

        # self.transform_func1 = T.Compose([
        #     T.ToTensor(),
        #     T.Normalize([0.5], [0.5])])
        self.transform_func2 = Lambda(lambda y: torch.zeros(
            10, dtype=torch.float32).scatter_(dim=0, index=torch.tensor(y), value=1))
        self.transform_func3 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.create_model()
        self.create_optim()

    def create_model(self, dropout_rate=0.25, is_static: bool = False):
        self.model = Sequential(
            Linear(28 * 28, 256),
            ReLU(),  # D
            Linear(256, 128),
            ReLU(),
            Linear(128, 32),
            ReLU(),
            Linear(32, 1),
            Dropout(p=dropout_rate),
            Sigmoid()).to(self.device)
        if is_static:
            return self.model

    def create_optim(self, learning_rate=0.01):
        learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = BCELoss()

    def get_data_loaders(self, batch_size=64, shuffle_sets=True, isdownload_dataset=False, root: str = '.'):
        """
        first binary_train_loader,
        second binary_test_loader
        :return: binary_train, binary_test
        """
        train_set = torchvision.datasets.FashionMNIST(
            root=root,
            train=True,
            download=isdownload_dataset,
            transform=self.transform_func3)
        test_set = torchvision.datasets.FashionMNIST(root=root,
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

    def train_modele(self, data_loader, epochs=50, after_train_save: bool = False):
        for i in range(epochs):  # A
            tloss = 0
            for n, (imgs, labels) in enumerate(data_loader):
                imgs = imgs.reshape(-1, 28 * 28)
                imgs = imgs.to(self.device)
                labels = torch.FloatTensor(
                    [x if x == 0 else 1 for x in labels])
                labels = labels.reshape(-1, 1).to(self.device)
                preds = self.model(imgs)
                loss = self.loss_fn(preds, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tloss += loss
            tloss = tloss / n
            print(f"at epoch {i}, loss is {tloss}")
            if after_train_save:
                self.save_model('.')

    def save_model(self, save_path: str):
        path = save_path + self.model_name + self.ext
        torch.save(self.model.state_dict(), path)
        print("model save on path ", path)

    def load_model(self, load_path: str):
        state_dict = torch.load(load_path + self.model_name + self.ext)
        loaded_model = torch_nn()
        loaded_model.load_state_dict(state_dict)
        loaded_model.eval()
        print("model loaded and ready")
        return loaded_model

    def predict(self, image_path: str):
        pass