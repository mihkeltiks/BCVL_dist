from __future__ import print_function

import gc
import logging
import os
import sys
import argparse

from contextlib import nullcontext

import deepspeed
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import StepLR
import torch.utils.data.distributed


from tqdm import tqdm

from resnet3d import generate_model

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Deepspeed Training')
parser.add_argument('--local_rank', type=int, default=-1, help="local rank for distributed training on gpus")
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser = deepspeed.add_config_arguments(parser)






# Training settings
class Trainer:
    def __init__(self, optimizer, loss_function, device,args, check: bool = False):

        self.check = check
        self.device = device
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.args = args

    def run_one_epoch(self, model, dataloader, train: bool = True):

        epoch_loss = 0.
        step = 0

        model.train()

        scaler = torch.cuda.amp.GradScaler()
        self.optimizer.zero_grad(set_to_none=True)

        for data,label in tqdm(dataloader):

            step += 1
            gc.collect()
            print("DATA SHAPE",data.shape)
            print("RANK",self.args.local_rank,"DATA", data)
            print("LABEL",label)

            data = data.to(self.device, dtype=torch.float16, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            
            out = model(data)
        #    print("OUT",out)
            loss = self.loss_function(out, label.float())
            print("LOSS",loss)
            model.backward(loss)
            model.step()
            # calculate loss and metrics
        #    with torch.cuda.amp.autocast(), torch.no_grad() if not train else nullcontext():
        #        Y_prob = model.forward(data)
        #        loss = self.loss_function(Y_prob, label.float())
#
 #           if train:
  #              scaler.scale(loss).backward()
   #             scaler.step(self.optimizer)
    #            scaler.update()
#                self.optimizer.zero_grad()
#                torch.cuda.empty_cache()
#
            epoch_loss += loss.item()

            if step >= 3 and self.check:
                break
        return


class MockDataSet(torch.utils.data.Dataset):
    def __init__(self, depth=100, dataset_size=4):
        super().__init__()

        self.depth = depth
        class1_labels = [[False]] * int(dataset_size / 2)
        class2_labels = [[True]] * int(dataset_size / 2)

        self.labels = class1_labels + class2_labels
        self.classes = ["zeros", "random"]

    def __len__(self):
        # a DataSet must know its size
        return len(self.labels)

    def __getitem__(self, index):

        label = self.labels[index]
        print("GETTING INDEX ", index)

        #if label == True:
        #x = torch.rand((1, self.depth, 512, 512)) # input channels, depth, height, width
        x = torch.full((1, self.depth, 512, 512), index, dtype=torch.float16)  # input channels, depth, height, width
        #else:
        #    x = torch.zeros((1, self.depth, 512, 512))

        label = torch.Tensor(label)

        return x, label


def main():
    args = parser.parse_args()

    print(f"Work in {os.getcwd()}")


    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
        print('\nGPU is ON! GPUS PER NODE', ngpus_per_node)
        
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()



    logging.info('Init Model')
    model = generate_model(model_depth=18, n_input_channels=1, n_classes=1)
    
    if torch.cuda.is_available():
        model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999),
                           weight_decay=5e-4)
    loss_function = torch.nn.BCEWithLogitsLoss().to(device)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    model, optimizer, _, _ = deepspeed.initialize(
        model = model,
        optimizer = optimizer,
        args = args,
        lr_scheduler = None,#scheduler,
        dist_init_required=True
        )



    train_dataset = MockDataSet(depth=500, dataset_size=4)
    loader_kwargs = {'num_workers': 7, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = data_utils.DataLoader(train_dataset, shuffle=(train_sampler is None), sampler=train_sampler,  batch_size=args.batch_size, **loader_kwargs)

    trainer = Trainer(optimizer=optimizer, loss_function=loss_function, device=device, args=args, check=True)
    logging.info('Start Training')

    nr_of_epochs = 4
    for epoch in range(1, nr_of_epochs):
        print("TRAIN")
        trainer.run_one_epoch(model, dataloader=train_loader, train=True)
        print("NOTRAIN")
        trainer.run_one_epoch(model, dataloader=train_loader,
                              train=False)  # tavaliselt kasutatakse siin test andmestiku, aga demo jaoks ühest dataloaderist piisab


if __name__ == "__main__":
    main()
