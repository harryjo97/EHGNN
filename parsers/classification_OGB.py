import argparse

class Parser:

    def __init__(self):

        self.parser = argparse.ArgumentParser(description='HyperDrop')
        self.parser.add_argument('--type', type=str, required=True)

        self.set_arguments()

    def set_arguments(self):

        self.parser.add_argument('--data', default='ogbg-molhiv', type=str,
                            choices=['ogbg-molhiv', 'ogbg-moltox21', 'ogbg-moltoxcast', 'ogbg-molbbbp'],
                            help='dataset type')
        self.parser.add_argument("--model", type=str, default='HyperDrop', choices=['HyperDrop'])

        self.parser.add_argument('--seed', type=int, default=42, help='seed')

        self.parser.add_argument('--num-hidden', type=int, default=128, help='hidden size')
        self.parser.add_argument('--batch-size', default=128, type=int, help='train batch size')
        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        self.parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay')
        self.parser.add_argument("--grad-norm", type=float, default=1.0)
        self.parser.add_argument("--dropout", type=float, default=0.5)

        self.parser.add_argument('--num-epochs', default=500, type=int, help='train epochs number')
        self.parser.add_argument("--gpu", type=int, default=-1)
        self.parser.add_argument('--patience', type=int, default=50, help='patience for earlystopping')
        
        self.parser.add_argument('--experiment-number', default='001', type=str)
        self.parser.add_argument('--debug', default=False, type=bool)

        self.parser.add_argument('--num-convs', default=3, type=int)
        self.parser.add_argument("--lr-schedule", action='store_true')
        self.parser.add_argument('--edge-ratio', type=float, default=0.8)
        
    def parse(self):

        args, unparsed  = self.parser.parse_known_args()
        
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        
        return args