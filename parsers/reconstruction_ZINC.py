import argparse

class Parser:

    def __init__(self):

        self.parser = argparse.ArgumentParser(description='HyperCluster')
        self.parser.add_argument('--type', type=str, required=True)

        self.set_arguments()

    def set_arguments(self):
        
        self.parser.add_argument('--seed', type=int, default=42, help='seed')
        self.parser.add_argument('--data', type=str, default="ZINC",
                                 choices=["ZINC"])
        self.parser.add_argument('--model', type=str, default="HyperCluster",
                                 choices=['HyperCluster'])

        self.parser.add_argument('--num-hidden', type=int, default=32, help='hidden size')
        self.parser.add_argument('--num-convs', type=int, default=2)

        self.parser.add_argument('--batch-size', default=128, type=int, help='train batch size')
        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        self.parser.add_argument("--dropout", type=float, default=0.0)

        self.parser.add_argument('--edge-ratio', type=float, default=0.15, help='pooling ratio')

        self.parser.add_argument('--num-epochs', default=500, type=int, help='train epochs number')
        self.parser.add_argument("--gpu", type=int, default=-1)
        self.parser.add_argument('--patience', type=int, default=200, help='patience for earlystopping')
        self.parser.add_argument('--test-only', action='store_true', default=False)

        self.parser.add_argument('--experiment-number', default='001', type=str)

        self.parser.add_argument('--num-heads', type=int, default=1, help='attention head size')
        self.parser.add_argument("--ln", action='store_true')
        self.parser.add_argument("--cluster", action='store_true')
        self.parser.add_argument('--debug', action='store_true', default=False)
        self.parser.add_argument("--lr-schedule", action='store_true')

    def parse(self):

        args, unparsed  = self.parser.parse_known_args()
        
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        
        return args
