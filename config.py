""" Config class for training the InvNet """

import argparse

def get_parser(name):
    """

    :param name: String for Config Name
    :return: parser
    """

    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser

class InvNetConfig(argparse.Namespace):
    def build_parser(self):
        parser = get_parser("InvNet config")
        parser.add_argument('--dataset', required=True, default='circle', help='circle / polycrystalline')
        parser.add_argument('--trainset_path', default='./datasets/circle/train_toyCircle_3Ch_128.h5',
                            help='Train dataset path')
        parser.add_argument('--validset_path', default='./datasets/circle/valid_toyCircle_3Ch_128.h5',
                            help='Valid dataset path')
        parser.add_argument('--output_path', required=True, help='output directory')
        parser.add_argument('--gpu', default=0, help='Selecting the gpu')
        parser.add_argument('--batch_size', default=32, help='Batch size for training')
        parser.add_argument('--critic_iter', default=5, help='Number of iter for descriminator')
        parser.add_argument('--proj_iter', default=5, help='Number of iteration for projection update.')
        parser.add_argument('--end_iter', default=100000, help='How many iterations to train for.')
        parser.add_argument('--lambda_gp', default=10, help='gradient penalty hyperparameter')
        parser.add_argument('--restore_mode', default=False,
                            help='If True, it will load saved model from OUT_PATH and continue to train')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

