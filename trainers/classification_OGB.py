import torch
import time
from tqdm import tqdm, trange
import numpy as np
from ogb.graphproppred import Evaluator
from transformers.optimization import get_cosine_schedule_with_warmup

from utils.data import num_graphs
from utils.setting import set_seed, set_logger, set_experiment_name, set_device
from utils.loader_OGB import load_data, load_model, load_dataloader


class Trainer(object):

    def __init__(self, args):

        super(Trainer, self).__init__()

        self.args = args

        ### Set seed, logger, and gpu device
        self.seed = set_seed(self.args)
        self.log_folder_name, self.exp_name = set_experiment_name(self.args)
        self.logger = set_logger(self.log_folder_name, self.exp_name, self.seed)
        self.device = set_device(self.args) 

        ### Load dataset and evaluator
        self.dataset = load_data(args)
        self.evaluator = Evaluator(args.data)


    def train(self):

        train_loader, val_loader, test_loader = load_dataloader(self.args, self.dataset)

        ### Load model and optimizer
        self.model = load_model(self.args).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.lr, 
                                            weight_decay = self.args.weight_decay)

        self.cls_criterion = torch.nn.BCEWithLogitsLoss()
        self.reg_criterion = torch.nn.MSELoss()

        if self.args.lr_schedule:
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 
                                                                self.args.patience * len(train_loader), 
                                                                self.args.num_epochs * len(train_loader))

        ### Train
        train_curve = []
        valid_curve = []
        test_curve = []
        eval_metric = self.dataset.eval_metric

        t_start = time.perf_counter()
        for epoch in trange(0, (self.args.num_epochs), desc = '[Epoch]', position = 1):

            self.model.train()
            total_loss = 0

            for _, data in enumerate(tqdm(train_loader, desc="[Iteration]")):

                if data.x.shape[0] == 1 or data.batch[-1] == 0: pass

                self.optimizer.zero_grad()
                data = data.to(self.device)
                out = self.model(data)

                is_labeled = data.y == data.y
                if "classification" in self.args.task_type: 
                    loss = self.cls_criterion(out.to(torch.float32)[is_labeled], 
                                                data.y.to(torch.float32)[is_labeled])
                else:
                    loss = self.reg_criterion(out.to(torch.float32)[is_labeled], 
                                                data.y.to(torch.float32)[is_labeled])

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
                total_loss += loss.item() * num_graphs(data)
                self.optimizer.step()

                if self.args.lr_schedule:
                    self.scheduler.step()

            total_loss = total_loss / len(train_loader.dataset)

            train_curve.append(self.eval(train_loader)[eval_metric])
            valid_curve.append(self.eval(val_loader)[eval_metric])
            test_curve.append(self.eval(test_loader)[eval_metric])

            self.logger.log(f"[Val: Epoch {epoch:03d}] Loss: {total_loss:.4f} Train: {train_curve[-1]:.4f} "
                            f"Valid: {valid_curve[-1]:.4f} Test: {test_curve[-1]:.4f}")

        t_end = time.perf_counter()

        ### Report results for highest validation result
        if 'classification' in self.dataset.task_type:
            best_val_epoch = np.argmax(np.array(valid_curve))
            best_train = max(train_curve)
        else:
            best_val_epoch = np.argmin(np.array(valid_curve))
            best_train = min(train_curve)

        best_val = valid_curve[best_val_epoch]
        test_score = test_curve[best_val_epoch]

        self.logger.log(f"Train: {best_train:4f} Valid: {best_val:4f} Test: {test_score:4f} "
                        f"with Time: {t_end - t_start:2f}")

        result_file = f"./results/{self.log_folder_name}/{self.exp_name}.txt"
        with open(result_file, 'a+') as f:
            f.write(f"SEED={self.seed}   Test Score:{test_score*100:.2f}\n")


        ### Save trained model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'Val': best_val,
            'Train': train_curve[best_val_epoch],
            'Test': test_score,
            'BestTrain': best_train
            }, './checkpoints/{}/best-model_{}.pth'.format(self.log_folder_name, self.seed))


    ### Evaluate
    def eval(self, loader):

        self.model.eval()

        y_true = []
        y_pred = []

        for _, batch in enumerate(tqdm(loader, desc="[Iteration]")):
            batch = batch.to(self.device)

            if batch.x.shape[0] == 1: pass

            with torch.no_grad():
                pred = self.model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim = 0).numpy()
        y_pred = torch.cat(y_pred, dim = 0).numpy()

        input_dict = {"y_true": y_true, "y_pred": y_pred}

        return self.evaluator.eval(input_dict)
