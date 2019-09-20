import argparse
from solver import Solver

def parse():
    parser = argparse.ArgumentParser(description="tree transformer")
    parser.add_argument('-model_dir',default='train_model',help='output model weight dir')
    parser.add_argument('-seq_length', type=int, default=50, help='sequence length')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-num_step', type=int, default=100000, help='sequence length')
    parser.add_argument('-data_dir',default='data_dir',help='data dir')
    parser.add_argument('-load',action='store_true',help='load pretrained model')
    parser.add_argument('-train', action='store_true',help='whether train the model')
    parser.add_argument('-test', action='store_true',help='whether test')
    parser.add_argument('-valid_path',default='data/valid.txt',help='validation data path')
    parser.add_argument('-train_path',default='data/train.txt',help='training data path')
    parser.add_argument('-test_path',default='data/test.txt',help='testing data path')
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse()
    solver = Solver(args)
    
    if args.train:
        solver.train()
    elif args.test:
        solver.test()