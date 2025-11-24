from functools import partial

import numpy as np
import torch
from tqdm import tqdm

class SemiSupervisedEnsemble:
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        student_model,
        teacher_model,
        logger,
        datamodule,
    ):
    

        self.device = device
        # self.models = models
        self.student = student_model
        self.teacher = teacher_model


        # Optim related things
        self.supervised_criterion = supervised_criterion
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
    def update_teacher(self, ema): # ema is
        for t, s in zip(self.teacher.parameters(), self.student.parameters()):
            t.data = ema * t.data + (1 - ema) * s.data


    # def validate(self):
    #     for model in self.models:
    #         model.eval()

    #     val_losses = []
        
    #     with torch.no_grad():
    #         for x, targets in self.val_dataloader:
    #             x, targets = x.to(self.device), targets.to(self.device)
                
    #             # Ensemble prediction
    #             preds = [model(x) for model in self.models]
    #             avg_preds = torch.stack(preds).mean(0)
                
    #             val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
    #             val_losses.append(val_loss.item())
    #     val_loss = np.mean(val_losses)
    #     return {"val_MSE": val_loss}
    def validate(self):
        self.teacher.eval()
        val_losses = []

        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                preds = self.teacher(x)
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

            supervised_losses_logged = []

            unlabeled_iter = iter(self.train_unlabeled)
            
            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)

                # --- Fetch unlabelled batch ---
                try:
                    x_u = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(self.train_unlabeled)
                    x_u = next(unlabeled_iter)

                x_u = x_u.to(self.device)

                # --- Supervised Forward pass ---
                self.optimizer.zero_grad()
                # Supervised loss
                # supervised_losses = [self.supervised_criterion(model(x), targets) for model in self.models]
                # supervised_loss = sum(supervised_losses)
                student_pred = self.student(x)
                supervised_loss = self.supervised_criterion(student_pred, targets)

                supervised_losses_logged.append(supervised_loss.detach().item())
                
                supervised_loss.backward()
                self.optimizer.step()
                #loss = supervised_loss
                #loss.backward()  # type: ignore
                #self.optimizer.step()
                
            self.scheduler.step()
            supervised_losses_logged = np.mean(supervised_losses_logged)

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
            }
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
            self.logger.log_dict(summary_dict, step=epoch)