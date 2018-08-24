import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', default=1, type=int, help='choose model')
parser.add_argument('--embed', default=0, type=int, help='choose model')

parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--beta', default=1e-5, type=float, help='learning rate')
parser.add_argument('--bs', default=500, type=int, help='batch size')
parser.add_argument('--ne', default=200, type=int, help='num of epochs')
parser.add_argument('--num_steps', default=500, type=int, help='num of epochs')

parser.add_argument('--start_date', default='2014-01-01')
#parser.add_argument('--start_date', default='2018-04-27')
parser.add_argument('--valid_date', default='2018-05-01')

parser.add_argument('--cnn_hidden_dim', type=int, default=150)

parser.add_argument('--cnn2_hidden_dim', type=int, default=150)

parser.add_argument('--rnn_hidden_dim', type=int, default=300)
parser.add_argument('--rnn_hidden_dim2', type=int, default=100)

parser.add_argument('--cnnrnn_hidden_dim', type=int, default=150)
parser.add_argument('--cnnrnn_hidden_dim2', type=int, default=64)

parser.add_argument('--drop', type=float, default=0.5)
parser.add_argument('--num_filters', type=int, default=128)
parser.add_argument('--num_filters2', type=int, default=32)

parser.add_argument('--title_len', default=30, type=int)
parser.add_argument('--em_dim', default=100, type=int)

def get_config():
    return parser.parse_args()
