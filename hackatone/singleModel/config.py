import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--idx', required=True, type=int, help='company index')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--beta', default=1e-4, type=float, help='learning rate')
parser.add_argument('--bs', default=100, type=int, help='batch size')
parser.add_argument('--ne', default=100, type=int, help='num of epochs')

parser.add_argument('--hidden_dim', type=int, default=150)
parser.add_argument('--num_filters', type=int, default=64)

parser.add_argument('--voca_size', default=70000, type=int)
parser.add_argument('--title_len',default=20, type=int)
parser.add_argument('--text_len', default=300, type=int)
parser.add_argument('--em_dim', default=300, type=int)

parser.add_argument('--evaluate', default=0, type=int)

def get_config():
    return parser.parse_args()
