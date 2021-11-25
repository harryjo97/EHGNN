import torch
import torch.nn.functional as F
import time
from tqdm import tqdm, trange
import numpy as np
from transformers.optimization import get_cosine_schedule_with_warmup

from utils.node_utils import accuracy
from utils.setting import set_seed, set_logger, set_experiment_name, set_device
from utils.loader_node import load_data, load_model

class Trainer(object):

    def __init__(self, args):

        super(Trainer, self).__init__()

        self.args = args

        ### Set seed, logger, and gpu device
        self.seed = set_seed(self.args)
        self.log_folder_name, self.exp_name = set_experiment_name(self.args)
        self.logger = set_logger(self.log_folder_name, self.exp_name, self.seed)
        self.device = set_device(self.args)

        A, features, labels, idx_train, idx_val, idx_test = load_data(self.args)
        self.dataset = {
            'A': A.to(self.device),
            'features': features.to(self.device),
            'labels': labels.to(self.device),
            'idx_train': idx_train.to(self.device),
            'idx_val': idx_val.to(self.device),
            'idx_test': idx_test.to(self.device)
        }

    def train(self):

        self.model = load_model(self.args).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.lr, 
                                            weight_decay = self.args.weight_decay)

        if self.args.lr_schedule:
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 
                                                                self.args.patience, 
                                                                self.args.num_epochs)


        best_acc = 0.0
        train_curve, valid_curve, test_curve = [], [], []
        patience = 0

        t_start = time.perf_counter()
        for epoch in trange(0, (self.args.num_epochs), desc = '[Epoch]', position = 1):

            self.model.train()

            out = self.model(self.dataset['features'], self.dataset['A'])
            loss = F.nll_loss(
                out[self.dataset['idx_train']], 
                self.dataset['labels'][self.dataset['idx_train']]
            )
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)                
            self.optimizer.step()

            if self.args.lr_schedule:
                self.scheduler.step()

            results = self.eval()
            train_curve.append(results['train_acc'])
            valid_curve.append(results['valid_acc'])
            test_curve.append(results['test_acc'])

            self.logger.log(f"[Val: Epoch {epoch:03d}] Loss: {loss.item():.4f} Train: {train_curve[-1]:.4f} "
                            f"Valid: {valid_curve[-1]:.4f} Test: {test_curve[-1]:.4f}")

            if results['valid_acc'] >= best_acc:
                best_acc = results['valid_acc']
                torch.save(self.model.state_dict(), 
                        f'./checkpoints/{self.log_folder_name}/experiment-{self.exp_name}_'
                        f'seed-{self.seed}_best-model.pth')
                patience = 0
            else:
                patience += 1

            if patience > self.args.patience: break

        t_end = time.perf_counter()

        best_train = max(train_curve)
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_val = valid_curve[best_val_epoch]
        test_score = test_curve[best_val_epoch]

        self.logger.log(f"Train: {best_train:4f} Valid: {best_val:4f} Test: {test_score:4f} "
                        f"with Time: {t_end - t_start:2f}")

        result_file = f"./results/{self.log_folder_name}/{self.exp_name}.txt"
        with open(result_file, 'a+') as f:
            f.write(f"SEED={self.seed}  Test Score:{test_score*100:.2f}\n")

        import pdb; pdb.set_trace()

    def eval(self):
        
        self.model.eval()

        out = self.model(self.dataset['features'], self.dataset['A'])

        return {
            'train_loss': F.nll_loss(out[self.dataset['idx_train']], self.dataset['labels'][self.dataset['idx_train']]).item(),
            'train_acc': accuracy(out[self.dataset['idx_train']], self.dataset['labels'][self.dataset['idx_train']]).item(),
            'valid_loss': F.nll_loss(out[self.dataset['idx_val']], self.dataset['labels'][self.dataset['idx_val']]).item(),
            'valid_acc': accuracy(out[self.dataset['idx_val']], self.dataset['labels'][self.dataset['idx_val']]).item(),
            'test_loss': F.nll_loss(out[self.dataset['idx_test']], self.dataset['labels'][self.dataset['idx_test']]).item(),
            'test_acc': accuracy(out[self.dataset['idx_test']], self.dataset['labels'][self.dataset['idx_test']]).item()
        }