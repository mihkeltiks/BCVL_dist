from __future__ import print_function

import gc
import logging
import os
import sys
from contextlib import nullcontext
import timeit

from tqdm import tqdm

import numpy as np
import torch

import torch.optim as optim

from monai.networks.nets import resnet18

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# Training settings

class Trainer:
    def __init__(self, optimizer, loss_function, check: bool = False):

        self.check = check
        self.device = torch.device("cuda")
        self.optimizer = optimizer
        self.loss_function = loss_function
    def run_one_epoch(self, model, train: bool = True):
        epoch_loss = 0.
        step = 0
        model.train()

        scaler = torch.cuda.amp.GradScaler()
        self.optimizer.zero_grad(set_to_none=True)
        for i in tqdm(range(5)): # lets say we have 100 data points in dataset
            depth =  300
            data = torch.rand((1,1,depth,512,512)) # batch size, input channels, depth, height, width
            label = torch.randint(2,(1,1)) # label, either 0 or 1

            step += 1
            gc.collect()

            data = data.to(self.device, dtype=torch.float16, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            

            # calculate loss and metrics
            with torch.cuda.amp.autocast(), torch.no_grad() if not train else nullcontext():
                Y_prob = model.forward(data)
                loss = self.loss_function(Y_prob, label.float())

            if train:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

            epoch_loss += loss.item()

            if step >= 5 and self.check:
                break
        return


def main():

    print(f"Work in {os.getcwd()}")

    np.random.seed(0)
    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        print('\nGPU is ON!')

    logging.info('Init Model')
    model = resnet18(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=1)

    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999),
                           weight_decay=5e-4)
    loss_function = torch.nn.BCEWithLogitsLoss().cuda()

    trainer = Trainer(optimizer=optimizer, loss_function=loss_function, check=True)

    logging.info('Start Training')

    nr_of_epochs = 10
    for epoch in range(1, nr_of_epochs):
        print("epoch ", epoch)
        print("TRAIN")
        trainer.run_one_epoch(model,train=True)
        print("TEST")
        trainer.run_one_epoch(model,train=False)

if __name__ == "__main__":
    main()
