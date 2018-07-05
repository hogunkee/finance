import argparse
import os
from solver import Solver
from torch.backends import cudnn
from data_loader import get_loader


def main(config):
    cudnn.benchmark = True

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    train_loader = get_loader(stock_path = config.train_path, batch_size = 100)
    valid_loader = get_loader(stock_path = config.valid_path, batch_size = 100000)

    solver = Solver(config)


    solver.train(train_loader, valid_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model type
    parser.add_argument('--model_type', type = str, default = None)

    # training hyper-parameters
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--reg', type = float, default = 0)
    parser.add_argument('--num_epochs', type = int, default = 1000)
    parser.add_argument('--batch_size', type = int, default = 1)

    # misc
    parser.add_argument('--mode', type = str, default = 'train')
    parser.add_argument('--num_workers', type = int, default = 2)
    parser.add_argument('--save_path', type = str, default = 'save')
    parser.add_argument('--load_path', type = str, default = None)
    parser.add_argument('--data_path', type = str, default = 'data')
    parser.add_argument('--train_path', type=str, default='train_stock.csv')
    parser.add_argument('--valid_path', type=str, default='valid_stock.csv')
    parser.add_argument('--log_step', type = int, default = 300)
    parser.add_argument('--test_step', type = int, default = 1)
    parser.add_argument('--use_gpu', type = bool, default = True)

    config = parser.parse_args()
    print(config)
    main(config)