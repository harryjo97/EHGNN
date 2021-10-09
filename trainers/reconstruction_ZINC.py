import os
import time
import rdkit
from tqdm import tqdm, trange
import torch
from torch_geometric.utils import to_dense_adj
from transformers.optimization import get_cosine_schedule_with_warmup

from utils.molecule_utils import mol_from_graphs
from utils.setting import set_seed, set_logger, set_experiment_name, set_device
from utils.loader_ZINC import load_data, load_model, load_dataloader


class Trainer(object):

    def __init__(self, args):

        super(Trainer, self).__init__()

        self.args = args
        
        ### Set seed, logger, gpu device and checkoint
        self.seed = set_seed(self.args)
        self.log_folder_name, self.exp_name = set_experiment_name(self.args)
        self.logger = set_logger(self.log_folder_name, self.exp_name, self.seed) 
        self.device = set_device(self.args) 
        self.ckpt = os.path.join(f'./checkpoints/{self.log_folder_name}', 
                                    f"best_molecule_{self.seed}.pth")

        ### Load dataloader
        self.train_loader, self.val_loader, self.test_loader = load_dataloader(self.args)


    def train(self):

        ### Load model and optimizer
        self.model = load_model(self.args).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = torch.nn.CrossEntropyLoss()

        if self.args.lr_schedule:
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 
                                                            self.args.patience * len(self.train_loader), 
                                                            self.args.num_epochs * len(self.train_loader))

        ### Train
        best_loss = 1e+6
        patience = self.args.patience

        t_start = time.perf_counter()
        epoch_iter = trange(0, self.args.num_epochs, desc='[EPOCH]', position=1)

        for epoch in epoch_iter:

            self.model.train()
            
            for _, data in enumerate(tqdm(self.train_loader, desc='[Train]', position=0)):
                self.optimizer.zero_grad()
                data = data.to(self.device)
                out = self.model(data)
                target = data.edge_attr.argmax(-1)
                
                loss = self.criterion(out, target)
                loss.backward()
                self.optimizer.step()

                if self.args.lr_schedule:
                    self.scheduler.step()

                desc = f"[Train] Train Loss {loss.item():.4f}"
                epoch_iter.set_description(desc)
                epoch_iter.refresh()

            valid_acc, valid_loss = self.eval(self.val_loader)
            self.logger.log(f"[Epoch {epoch}] Loss: {valid_loss:.4f}, Acc: {valid_acc:.4f}")

            if valid_loss < best_loss:
                torch.save(self.model.state_dict(), self.ckpt)
                patience = self.args.patience
                best_loss = valid_loss
            else:
                patience -= 1
                if patience == 0: break

        t_end = time.perf_counter()

        ### Load best model
        self.model = load_model(self.args)
        self.model.load_state_dict(torch.load(self.ckpt))
        self.model = self.model.to(self.device)

        ### Compute exact match, validity, and test accuracy
        num_valid, num_invalid, exact_match, validity, test_acc = self.test(self.test_loader)

        ### Report results
        self.logger.log(f"GT Valid Molecules: {num_valid}, Invalid Molecules: {num_invalid}")
        self.logger.log(f"EM: {exact_match:.4f}, Validity: {validity:.4f}, Acc: {test_acc:.4f} "
                        f"with Time: {t_end - t_start:.2f}")

        result_file = f"./results/{self.log_folder_name}/{self.exp_name}.txt"
        with open(result_file, 'a+') as f:
            f.write(f"{self.seed}: ExactMatch={exact_match:.4f}  Validity={validity:.4f}  "
                    f"TestAcc={test_acc:.4f}\n")


    ### Evaluate
    def eval(self, loader):

        self.model.eval()

        valid_loss = 0
        valid_acc = 0
        total_node_num = 0

        for _, data in enumerate(tqdm(loader, desc='[Eval]', position=0)):
            data = data.to(self.device)
            out = self.model(data)
            pred_logit = torch.softmax(out, dim=-1)
            target = data.edge_attr.argmax(-1)
            loss = self.criterion(pred_logit, target)
            pred = pred_logit.argmax(-1)
            valid_loss += loss
            valid_acc += (pred == target).sum().item()
            total_node_num += float(pred.shape[0])

        valid_loss = valid_loss / float(len(loader))
        valid_acc = valid_acc / total_node_num * 100

        return valid_acc, valid_loss.item()

    
    ### Compute exact match, validity, and test accuracy
    def test(self, loader):

        test_acc, _ = self.eval(loader)

        exact_match = 0
        validity = 0
        num_valid = 0
        num_invalid = 0

        for _, data in enumerate(tqdm(loader, desc='[Test]')):
            nodes = data.x.argmax(-1)
            a = to_dense_adj(data.edge_index, edge_attr=data.edge_attr.argmax(-1))[0]
            mol = mol_from_graphs(nodes, a)
            
            if mol is None:
                num_invalid += 1
                continue
            else: num_valid += 1
            
            smiles = rdkit.Chem.MolToSmiles(mol)
            data = data.to(self.device)
            out = self.model(data)
            pred = torch.softmax(out, dim=-1).argmax(-1)

            a_pred = to_dense_adj(data.edge_index, edge_attr=pred)[0].to('cpu')
            pred_mol = mol_from_graphs(nodes, a_pred)
            
            if pred_mol is not None:
                pred_smiles = rdkit.Chem.MolToSmiles(pred_mol)
            else: pred_smiles = ''

            _exact_match = smiles == pred_smiles
            _validity = pred_smiles != ''

            exact_match += _exact_match
            validity += _validity

        exact_match = exact_match / float(num_valid) * 100
        validity = validity / float(num_valid) * 100

        return num_valid, num_invalid, exact_match, validity, test_acc

