from core.entities.TorchNet import torch_nn

tnet: torch_nn = torch_nn()


def load():
    global tnet
    tnet = tnet.load_model(load_path='core/saveModel/')


def main_logic():
    global tnet
    tnet = torch_nn()
    data_loaders = tnet.get_data_loaders(root='../core/data', isdownload_dataset=True)

    train_dataloader = data_loaders[0]
    test_dataloader = data_loaders[1]

    tnet.train_modele(train_dataloader, epochs=10)
    tnet.save_model(save_path='core/saveModel')


if __name__ == '__main__':
    load()
