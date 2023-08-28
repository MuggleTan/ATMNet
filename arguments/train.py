import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, help='Path to the config file', type=str)

# general
parser.add_argument('--save-dir', default='out', help='Saving path of models and logs')
parser.add_argument('--logstep-train', default=10, type=int, help='Training log interval in steps')
parser.add_argument('--save-model', default='both', choices=['last', 'best', 'no', 'both'])
parser.add_argument('--val-every-n-epochs', type=int, default=1, help='Validation interval in epochs')
parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

# data
parser.add_argument('--dataset', type=str, default='NYUv2', help='Name of the dataset', choices=['NYUv2', 'DIML', 'Middlebury'])
parser.add_argument('--data-dir', type=str, default='/path/to/the/dataset', help="Root directory of the dataset")
parser.add_argument('--num-workers', type=int, default=8, metavar='N', help='Number of dataloader worker processes')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--crop-size', type=int, default=256, help='Size of the input (squared) patches')
parser.add_argument('--scaling', type=int, default=4, help='Scaling factor', choices=[4, 8, 16])
parser.add_argument('--max-rotation', type=float, default=15., help='Maximum rotation angle (degrees)')
parser.add_argument('--no-flip', action='store_true', default=False, help='Switch off random flipping')
parser.add_argument('--in-memory', action='store_true', default=False, help='Hold data in memory during training')

# training
parser.add_argument('--loss', default='udl', type=str, choices=['l1', 'mse', 'udl'])
parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam'])
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--w-decay', type=float, default=1e-5)
parser.add_argument('--lr-scheduler', type=str, default='step', choices=['no', 'step', 'plateau'])
parser.add_argument('--lr-gamma', type=float, default=0.9, help='LR decay rate')
parser.add_argument('--skip-first', action='store_true', help='Don\'t optimize during first epoch')
parser.add_argument('--gradient-clip', type=float, default=0.01, help='If > 0, clips gradient norm to that value')
parser.add_argument('--cuda', default=1, type=int, help='Num of the GUP to run')
parser.add_argument('--step2', action='store_true', default=False, help='The training step')
parser.add_argument('--step1_path', default='.', type=str, help='The model path of step1')

