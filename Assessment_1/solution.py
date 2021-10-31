import torch


class Solution:
    """
    solution for our task, point the model then we can train
    :param neural_model: the model we have defined
    :param lr: default is 1e-3
    :param epochs: default is 100
    :param optimizer: the optimizer for model
    :train_set: train dataset
    :val_set: validation dataset
    :test_set: test dataset
    :batch_size: how many sample we have to feed into the model for each gradient decent
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, **kwargs):
        self.model = kwargs['model'].to(self.device)
        self.lr = kwargs['lr']
        self.epochs = kwargs['epochs']
        self.optimizer = kwargs['optimizer']
        self.loss_func = kwargs['loss_func']
        self.train_set = kwargs['train_set']
        self.val_set = kwargs['val_set']
        self.test_set = kwargs['test_set']
        self.batch_size = kwargs['batch_size']

    def fit(self):
        train_loader = torch.utils.data.DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=1,
                                                   pin_memory=True)
        self.model.train()

        train_loss = 0.0
        train_counter = 0

        for _, data in enumerate(train_loader):
            train_counter += 1
            inputs, targets = data[0].to(self.device), data[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, targets)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        train_epoch_loss = train_loss / train_counter
        return train_epoch_loss

    def validate(self):
        val_loader = torch.utils.data.DataLoader(self.val_set, self.batch_size, shuffle=True, num_workers=1,
                                                 pin_memory=True)
        val_counter = 0
        val_loss = 0.0
        self.model.evl()

        with torch.no_grad():
            for _, data in enumerate(val_loader):
                val_counter += 1
                inputs, targets = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, targets)
                val_loss += loss.item()
            val_epoch_loss = val_loss / val_counter
            return val_epoch_loss

    def test(self):
        test_loader = torch.utils.data.DataLoader(self.test_set, self.batch_size, shuffle=True, num_workers=1,
                                                  pin_memory=True)
        test_counter = 0
        test_loss = 0.0
        self.model.evl()

        with torch.no_grad():
            for _, data in enumerate(test_loader):
                test_counter += 1
                inputs, targets = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, targets)
                test_loss += loss.item()
            test_epoch_loss = test_loss / test_counter
            return test_epoch_loss
