import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--beta', default=1e-4, type=float, help='learning rate')
parser.add_argument('--bs', default=1000, type=int, help='batch size')
parser.add_argument('--ne', default=100, type=int, help='num of epochs')

parser.add_argument('--start_date', default='0000-00-00')
#parser.add_argument('--start_date', default='2018-04-27')
parser.add_argument('--valid_date', default='2018-05-01')

parser.add_argument('--cnn_hidden_dim', type=int, default=150)

parser.add_argument('--cnn2_hidden_dim', type=int, default=150)

parser.add_argument('--rnn_hidden_dim', type=int, default=150)
parser.add_argument('--rnn_hidden_dim2', type=int, default=64)

parser.add_argument('--cnnrnn_hidden_dim', type=int, default=150)
parser.add_argument('--cnnrnn_hidden_dim2', type=int, default=64)

parser.add_argument('--drop', type=float, default=0.0)
parser.add_argument('--num_filters', type=int, default=64)
parser.add_argument('--num_filters2', type=int, default=32)

parser.add_argument('--title_len', default=64, type=int)
parser.add_argument('--em_dim', default=100, type=int)

def get_config():
    return parser.parse_args()
