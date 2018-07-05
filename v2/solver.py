import math
import os
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable

from model import Model

class Solver(object):
    def __init__(self, config):

        self.model_type = config.model_type
        self.lr = config.lr
        self.reg = config.reg
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size

        self.num_workers = config.num_workers
        self.save_path = config.save_path
        self.load_path = config.load_path
        self.data_path = config.data_path
        self.log_step = config.log_step
        self.test_step = config.test_step
        self.use_gpu = config.use_gpu
        self.build_model()

    def build_model(self):
        if self.load_path == None:
            self.model = Model(vocab_size = 35000, embed_size = 100, hidden_size = 10,\
                               num_layers = 2)
        else:
            self.model = torch.load(self.load_path)
        self.optimizer = optim.Adam(self.model.parameters(),self.lr)
        if torch.cuda.is_available() and self.use_gpu:
            self.model.cuda()

    def to_variable(self, x):
        if torch.cuda.is_available() and self.use_gpu:
            x = x.cuda()
        return Variable(x)


    def train(self, train_loader, valid_loader):
        print("Start Train!!")
        print()
        total_step = len(train_loader)
        for epoch in range(self.num_epochs):
            for i, [data,label] in enumerate(train_loader):

                self.model.train()

                data = self.to_variable(data)
                label = self.to_variable(label)

                outputs = self.model(data)
                loss = self.model.mseloss(outputs, label)

                param_sum = 0
                for param in self.model.parameters():
                    param_sum += (param * param).sum()

                loss += self.reg * param_sum

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()


                correct = ((outputs>0.5).int() == label.int()).int()
                acc = correct.sum().float()/ len(correct)

                if (i + 1) % self.log_step == 0:
                    print('Epoch [%d/%d], Step[%d/%d], MSE_loss: %.4f, Train_Acc: %.4f'
                          % (epoch + 1, self.num_epochs, i + 1, total_step, loss, acc))

            if (epoch + 1) % self.test_step == 0:
                self.model.eval()
                for i, [data,label] in enumerate(valid_loader):
                    data = self.to_variable(data)
                    label = self.to_variable(label)
                    outputs = self.model(data)

                    correct = ((outputs > 0.5).int() == label.int()).int()
                    acc = correct.sum().float() / len(correct)

                    print()
                    print('Epoch [%d/%d], Valid_acc: %.4f'
                          % (epoch + 1, self.num_epochs, acc))
                    print()

            model_path = os.path.join(self.save_path, 'model-%d.pkl' % (epoch + 1))
            torch.save(self.model, model_path)
