import os
import sys
import torch
import yaml
import wandb

from collections import namedtuple
from tqdm import tqdm
from pytorch_metric_learning import losses

from morphofeatures.shape.network import DeepGCN
from morphofeatures.shape.data_loading.loader import get_train_val_loaders


class ShapeTrainer:

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(config['device'])
        self.ckpt_dir = os.path.join(config['experiment_dir'], 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.build_loaders()
        self.reset()

    def reset(self):
        self.build_model()
        self.best_val_loss = None
        self.epoch = 0
        self.step = 0

    def build_loaders(self):
        dataset_config = self.config['data']
        loader_config = self.config['loader']
        loaders = get_train_val_loaders(dataset_config, loader_config)
        self.train_loader = loaders['train']
        self.val_loader = loaders['val']

    def build_model(self):
        self.model = DeepGCN(**self.config['model']['kwargs'])
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(
                self.model,
                device_ids=[i for i in range(torch.cuda.device_count())]
            )
            self.model.cuda()
        else:
            self.model = self.model.to(self.device)

        self.optimizer = getattr(
            torch.optim, self.config['optimizer']['name']
        )(self.model.parameters(), **self.config['optimizer']['kwargs'])
        
        # Handle ContrastiveLoss with proper distance function
        criterion_config = self.config['criterion']
        if criterion_config['name'] == 'ContrastiveLoss':
            from pytorch_metric_learning import distances
            # Check if distance is a dictionary with function key
            if isinstance(criterion_config['kwargs'].get('distance'), dict) and 'function' in criterion_config['kwargs']['distance']:
                # Get the distance function class from pytorch_metric_learning.distances
                distance_name = criterion_config['kwargs']['distance']['function']
                distance_fn = getattr(distances, distance_name)()
                # Replace the dictionary with the actual distance function
                criterion_config['kwargs']['distance'] = distance_fn
        
        self.criterion = getattr(
            losses, criterion_config['name']
        )(**criterion_config['kwargs'])
        
        # Initialize scheduler - use StepLR by default
        scheduler_config = self.config.get('scheduler', {})
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=scheduler_config.get('step_size', 10),
            gamma=scheduler_config.get('gamma', 0.5)
        )

    def checkpoint(self, force=True):
        save = force or (self.epoch % self.config['training']['checkpoint_every'] == 0)
        if save:
            info = {
                'epoch': self.epoch,
                'iteration': self.step,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'config/model/name': self.config['model']['name'],
                'config/model/kwargs': self.config['model']['kwargs'],
            }
            ckpt_name = f'best_ckpt_iter_{self.step}.pt' if force else f'ckpt_iter_{self.step}.pt'
            ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
            torch.save(info, ckpt_path)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for data in tqdm(self.train_loader, desc='Training'):
            out, h = self.model(data['points'], data['features'])
            labels = torch.arange(out.size(0) // 2) \
                          .repeat_interleave(2) \
                          .to(self.device)

            loss = self.criterion(out, labels)
            if torch.isnan(loss).item():
                print(f'Loss: {loss.item()}')
                continue

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            # Make wandb logging optional
            if hasattr(self, 'use_wandb') and self.use_wandb:
                wandb.log({'training/loss': loss.item()}, step=self.step)
            self.step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {self.epoch} - Average training loss: {avg_loss:.6f}")
        return avg_loss

    def validate_epoch(self):
        if self.epoch % self.config['training']['validate_every'] != 0:
            return
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data in tqdm(self.val_loader, desc='Validation'):
                out, h = self.model(data['points'], data['features'])
                labels = torch.arange(out.size(0) // 2) \
                            .repeat_interleave(2) \
                            .to(self.device)
                loss = self.criterion(out, labels)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.val_loader)
        print(f"Epoch {self.epoch} - Validation loss: {avg_loss:.6f}")

        if self.best_val_loss is None or avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.checkpoint(True)
            print(f"New best validation loss: {avg_loss:.6f}")

        # Make wandb logging optional
        if hasattr(self, 'use_wandb') and self.use_wandb:
            wandb.log({'validation/loss': avg_loss}, step=self.step)
        return avg_loss

    def train(self):
        for epoch_num in tqdm(range(self.config['training']['epochs']), desc='Epochs'):
            self.epoch = epoch_num
            self.train_epoch()
            self.validate_epoch()
            self.scheduler.step()
            self.checkpoint(False)

    def run(self):
        # Check if wandb should be used
        use_wandb = self.config.get('use_wandb', True)
        self.use_wandb = use_wandb
        
        if use_wandb:
            try:
                import wandb
                with wandb.init(project=self.config.get('wandb_project', 'MorphoFeatures')):
                    self.validate_epoch()
                    self.train()
            except (ImportError, AttributeError) as e:
                print(f"Warning: Unable to use wandb for logging: {e}")
                print("Continuing without wandb logging...")
                self.use_wandb = False
                self.validate_epoch()
                self.train()
        else:
            print("Wandb logging disabled in config. Using console output only.")
            self.validate_epoch()
            self.train()


if __name__ == '__main__':
    path_to_config = sys.argv[1]
    with open(path_to_config, 'r') as f: 
        config = yaml.load(f, Loader=yaml.FullLoader)

    trainer = ShapeTrainer(config)
    trainer.run()
