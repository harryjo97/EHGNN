import torch
import torch.nn.functional as F
import time
from tqdm import tqdm, trange
import numpy as np
from transformers.optimization import get_cosine_schedule_with_warmup

from utils.data import num_graphs
from utils.setting import set_seed, set_logger_fold, set_experiment_name, set_device
from utils.loader_TU import load_data, load_model, load_dataloader


class Trainer(object):

    def __init__(self, args):

        super(Trainer, self).__init__()

        self.args = args

        ### Set seed and gpu device
        self.seed = set_seed(self.args)
        self.log_folder_name, self.exp_name = set_experiment_name(self.args)
        self.device = set_device(self.args) 

        ### Load dataset 
        self.dataset = load_data(self.args)
        

    def train(self):

        ### Train
        overall_results = {
            'val_acc': [],
            'test_acc': []
        }

        train_fold_iter = tqdm(range(1, 11), desc='Training')
        val_fold_iter = [i for i in range(1, 11)]

        for fold_number in train_fold_iter:
            
            ### Set logger, loss and accuracy for each fold
            logger = set_logger_fold(self.log_folder_name, self.exp_name, self.seed, fold_number)

            patience = 0
            best_loss_epoch = 0
            best_acc_epoch = 0
            best_loss = 1e9
            best_loss_acc = -1e9
            best_acc = -1e9
            best_acc_loss = 1e9

            val_fold_number = val_fold_iter[fold_number - 2]

            train_loader, val_loader, test_loader = load_dataloader(self.args, self.dataset, 
                                                                    fold_number, val_fold_number)

            # Load model and optimizer
            self.model = load_model(self.args).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.lr, 
                                                weight_decay = self.args.weight_decay)

            if self.args.lr_schedule:
                self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 
                                                                    self.args.patience * len(train_loader), 
                                                                    self.args.num_epochs * len(train_loader))

            
            t_start = time.perf_counter()
            ### K-Fold Training
            for epoch in trange(0, (self.args.num_epochs), desc = '[Epoch]', position = 1):

                self.model.train()
                total_loss = 0

                for _, data in enumerate(train_loader):

                    self.optimizer.zero_grad()
                    data = data.to(self.device)
                    out = self.model(data)
                    loss = F.nll_loss(out, data.y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
                    total_loss += loss.item() * num_graphs(data)
                    
                    self.optimizer.step()

                    if self.args.lr_schedule:
                        self.scheduler.step()

                total_loss = total_loss / len(train_loader.dataset)

                ### Validation
                val_acc, val_loss = self.eval(val_loader)
                
                if val_loss < best_loss:
                    torch.save(self.model.state_dict(), 
                        f'./checkpoints/{self.log_folder_name}/experiment-{self.exp_name}_'
                        f'fold-{fold_number}_seed-{self.seed}_best-model.pth')
                    
                    best_loss_acc = val_acc
                    best_loss = val_loss
                    best_loss_epoch = epoch
                    patience = 0
                else:
                    patience += 1

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_acc_loss = val_loss
                    best_acc_epoch = epoch

                ### Validation log
                logger.log(f'[Val: Fold {fold_number}-Epoch {epoch}] TrL: {total_loss:.4f} ' 
                            f'VaL: {val_loss:.4f} VaAcc: {val_acc:.4f}')
                logger.log(f"[Val: Fold {fold_number}-Epoch {epoch}] Loss: {best_loss:.4f} "
                            f"Acc: {best_loss_acc:.4f} at Epoch: {best_loss_epoch} / "
                            f"Loss: {best_acc_loss:.4f} Acc: {best_acc:.4f} at Epoch: {best_acc_epoch}")

                train_fold_iter.set_description(f'[Fold {fold_number}] TrL: {total_loss:.4f} '
                                                f'VaL: {val_loss:.4f} VaAcc: {val_acc:.4f}')
                train_fold_iter.refresh()
                if patience > self.args.patience: break

            t_end = time.perf_counter()

            ### Test log
            checkpoint = torch.load(f'./checkpoints/{self.log_folder_name}/experiment-{self.exp_name}_'
                                    f'fold-{fold_number}_seed-{self.seed}_best-model.pth')
            self.model.load_state_dict(checkpoint)
            
            test_acc, test_loss = self.eval(test_loader)
            
            logger.log(f"[Test: Fold {fold_number}] Loss: {best_loss:4f} Acc: {best_loss_acc:4f} "
                        f"at Epoch: {best_loss_epoch} / (Acc) Loss: {best_acc_loss:4f} "
                        f"Acc: {best_acc:4f} at Epoch: {best_acc_epoch}")
            logger.log(f"[Test: Fold {fold_number}] Test Acc: {test_acc:4f} with Time: {t_end-t_start:.2f}")

            test_result_file = "./results/{}/{}-results.txt".format(self.log_folder_name, self.exp_name)
            with open(test_result_file, 'a+') as f:
                f.write(f"[FOLD {fold_number}] {self.seed}: {best_loss:.4f} "
                        f"{best_acc:.4f} {test_loss:.4f} {test_acc:.4f}\n")

            ### Report results
            overall_results['val_acc'].append(best_acc)
            overall_results['test_acc'].append(test_acc)

            final_result_file = f"./results/{self.log_folder_name}/{self.exp_name}.txt"
            with open(final_result_file, 'a+') as f:
                f.write(f"{self.seed}: {np.array(overall_results['val_acc']).mean():.4f} "
                        f"{np.array(overall_results['val_acc']).std():.4f} " 
                        f"{np.array(overall_results['test_acc']).mean():.4f} "
                        f"{np.array(overall_results['test_acc']).std():.4f}\n")


    ### Evaluate
    def eval(self, loader):

        self.model.eval()

        correct = 0.
        loss = 0.

        for data in loader:

            data = data.to(self.device)
            with torch.no_grad():
                out = self.model(data)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            loss += F.nll_loss(out, data.y, reduction='sum').item()
        
        return correct / len(loader.dataset), loss / len(loader.dataset)

