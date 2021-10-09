import time
import numpy as np
import torch
import random
import os


class Logger:
    def __init__(self, filepath, mode, lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock

    def log(self, str):
        if self.lock:
            self.lock.acquire()

        try:
            with open(self.filepath, self.mode) as f:
                f.write(str + '\n')
        except Exception as e:
            print(e)

        if self.lock:
            self.lock.release()


def set_seed(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return args.seed
    

def set_experiment_name(args):

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    log_folder_name = os.path.join(*[args.data, args.model, args.experiment_number])

    if not(os.path.isdir('./checkpoints/{}'.format(log_folder_name))):
        os.makedirs(os.path.join('./checkpoints/{}'.format(log_folder_name)))

    if not(os.path.isdir('./results/{}'.format(log_folder_name))):
        os.makedirs(os.path.join('./results/{}'.format(log_folder_name)))

    if not(os.path.isdir('./logs/{}'.format(log_folder_name))):
        os.makedirs(os.path.join('./logs/{}'.format(log_folder_name)))

    print("Make Directory {} in Logs, Checkpoints and Results Folders".format(log_folder_name))

    exp_name = str()
    exp_name += f"{ts}"
    # exp_name += "ER={}".format(args.edge_ratio)

    # Save training arguments for reproduction
    torch.save(args, os.path.join('./checkpoints/{}'.format(log_folder_name), 'training_args.bin'))

    return log_folder_name, exp_name


def set_logger(log_folder_name, exp_name, seed):

    logger = Logger(str(os.path.join(f'./logs/{log_folder_name}/', 
                    f'exp-{exp_name}_seed-{seed}.log')), mode='a')

    return logger


def set_logger_fold(log_folder_name, exp_name, seed, fold_number):

    logger = Logger(str(os.path.join(f'./logs/{log_folder_name}/', 
                    f'exp-{exp_name}_seed-{seed}_fold-{fold_number}.log')), mode='a')

    return logger


def set_device(args):

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'

    return device