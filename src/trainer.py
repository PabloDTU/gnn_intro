from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.utils import dropout_edge

class SemiSupervisedEnsemble:
    def __init__(
        self,
        supervised_criterion,
        consistency_criterion,
        optimizer,
        scheduler,
        device,
        student_model,
        teacher_model,
        logger,
        datamodule,
        ema_decay=0.999,
        max_consistency_weight=10.0,
        rampup_epochs=80,
    ):

        self.device = device
        # self.models = models
        self.student = student_model
        self.teacher = teacher_model

        # Initialize teacher as an exact copy of student
        self.teacher.load_state_dict(self.student.state_dict())

        # Optim related things
        self.supervised_criterion = supervised_criterion
        self.consistency_criterion = consistency_criterion
        self.ema_decay = ema_decay
        self.max_consistency_weight = max_consistency_weight
        self.rampup_epochs = rampup_epochs
        #all_params = [p for m in self.models for p in m.parameters()]
        #self.optimizer = optimizer(params=all_params)
        self.optimizer = optimizer(params=self.student.parameters()) # only student is optimized - teacher is updated via EMA
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        self.train_dataloader = datamodule.train_dataloader()
        self.train_unlabeled = datamodule.unlabeled_dataloader() 
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Logging
        self.logger = logger
    
    # Exponential Moving Average update for teacher model
    def update_teacher(self):
        ema = self.ema_decay # ema is the decay rate
        for t, s in zip(self.teacher.parameters(), self.student.parameters()):
            t.data = ema * t.data + (1 - ema) * s.data
    
    # Consistency weight ramp-up function
    def get_consistency_weight(self, epoch: int) -> float:
        """Exponential ramp-up from 0 to max_consistency_weight over rampup_epochs."""
        if epoch >= self.rampup_epochs:
            return self.max_consistency_weight
        phase = 1.0 - epoch / float(self.rampup_epochs)
        return self.max_consistency_weight * float(np.exp(-5.0 * phase * phase))
    
    # Data augmentation functions
    def augment_features(self, batch, drop_prob=0.005):
        x = batch.x
        mask = torch.rand_like(x) < drop_prob
        x_aug = x.clone()
        x_aug[mask] = 0.0
        batch_aug = batch.clone()
        batch_aug.x = x_aug
        return batch_aug


    def augment_edges(self, batch, drop_prob=0.00):
        """Randomly drops edges (and matching edge_attr) with probability drop_prob."""
        # If not training or drop_prob is zero, skip
        if drop_prob <= 0.0 or not self.student.training:
            return batch

        edge_index = batch.edge_index
        num_edges = edge_index.size(1)

        # Sample a mask: True = keep edge
        keep_mask = torch.rand(num_edges, device=edge_index.device) >= drop_prob

        edge_index_aug = edge_index[:, keep_mask]

        batch_aug = batch.clone()
        batch_aug.edge_index = edge_index_aug

        # If edge attributes exist, keep them in sync
        if getattr(batch, "edge_attr", None) is not None:
            batch_aug.edge_attr = batch.edge_attr[keep_mask]

        return batch_aug

    
    def augment_graph(self, batch):
        # Student sees noise, teacher sees clean input
        batch_aug = self.augment_features(batch, drop_prob=0.005)
        batch_aug = self.augment_edges(batch_aug, drop_prob=0.00)
        return batch_aug
    

    def validate(self):
        self.teacher.eval()
        val_losses = []

        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                #preds = self.teacher(x)
                preds = self.student(x) # validate student performance
                val_loss = torch.nn.functional.mse_loss(preds, targets)
                val_losses.append(val_loss.item())

        return {"val_MSE": np.mean(val_losses)}


    def train(self, total_epochs, validation_interval):
        #self.logger.log_dict()
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            # for model in self.models:
            #     model.train()
            self.student.train()
            self.teacher.eval()   # teacher does NOT train

            consistency_w = self.get_consistency_weight(epoch) # get current consistency weight

            supervised_losses_logged = []
            consistency_losses_logged = []

            unlabeled_iter = iter(self.train_unlabeled)
            
            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
    
                # --- Fetch unlabeled batch ---
                try:
                    x_u, _ = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(self.train_unlabeled)
                    x_u, _ = next(unlabeled_iter)

                x_u = x_u.to(self.device)


                # --- Forward passes for labelled & unlabelled ---
                self.optimizer.zero_grad()

                # Apply augmentations to input for the unlabeled student ONLY
                #x_aug     = self.augment_graph(x)
                x_u_aug   = self.augment_graph(x_u)

                # labelled
                student_l = self.student(x) # CLEAN graph for supervised path
                with torch.no_grad():
                    teacher_l = self.teacher(x)

                # unlabelled
                student_u = self.student(x_u_aug)
                with torch.no_grad():
                    teacher_u = self.teacher(x_u)
                # Supervised loss
                # supervised_losses = [self.supervised_criterion(model(x), targets) for model in self.models]
                # supervised_loss = sum(supervised_losses)
                
                # --- Supervised loss (labelled only) ---
                supervised_loss = self.supervised_criterion(student_l, targets)
                
                # --- Consistency loss (labelled + unlabelled) ---
                consistency_loss_u = self.consistency_criterion(student_u, teacher_u.detach())
                consistency_loss = consistency_loss_u

                # Burn-in: disable consistency for first 40 epochs
                if epoch < self.rampup_epochs:
                    consistency_loss = consistency_loss_u.detach() * 0.0

                # --- Total loss ---
                loss = supervised_loss + consistency_w * consistency_loss

                supervised_losses_logged.append(supervised_loss.detach().item())
                consistency_losses_logged.append(consistency_loss.detach().item())


                # Backprop on student
                loss.backward()
                self.optimizer.step()

                # EMA update of teacher
                self.update_teacher()
            
            self.scheduler.step()
            supervised_losses_logged = np.mean(supervised_losses_logged)
            consistency_losses_logged = np.mean(consistency_losses_logged)

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
                "consistency_loss": consistency_losses_logged,
                "consistency_weight": consistency_w,
            }

            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
            self.logger.log_dict(summary_dict, step=epoch)