import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, help='Path to the config file', type=str)

parser.add_argument('--dataset', type=str, default='NYUv2', help='Name of the dataset')
parser.add_argument('--data-dir', type=str, default='/path/to/the/dataset', help='Root directory of the dataset')
parser.add_argument('--num-workers', type=int, default=8, metavar='N', help='Number of dataloader worker processes')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--crop-size', type=int, default=256, help='Size of the input (squared) patches')
parser.add_argument('--scaling', type=int, default=4, help='Scaling factor')
parser.add_argument('--in-memory', default=False, action='store_true', help='Hold data in memory during evaluation')

parser.add_argument('--cuda', default=0, type=int, help='Num of the GUP to run')
parser.add_argument('--checkpoint', type=str, default='', help='Checkpoint path to evaluate')
