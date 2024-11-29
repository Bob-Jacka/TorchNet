from core.entities.TorchNet import torch_nn

tnet: torch_nn = torch_nn()

three_pic = 'core/data/three.png'

def load():
    global tnet
    tnet = tnet.load_model(load_path='core/saveModel/')


def after_load_train():
    train_dl = tnet.get_data_loaders(isdownload_dataset=True)[0]
    tnet.train_model(train_dl, after_train_save=True)

def predict_numbers():
    tnet.predict(three_pic)

def main_logic():
    global tnet
    tnet = torch_nn()
    data_loaders = tnet.get_data_loaders(root='../core/data', isdownload_dataset=True)

    train_dataloader = data_loaders[0]
    test_dataloader = data_loaders[1]

    tnet.train_model(train_dataloader, epochs=10)
    tnet.save_model(save_path='core/saveModel')


if __name__ == '__main__':
    after_load_train()
