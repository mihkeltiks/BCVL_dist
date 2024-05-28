from __future__ import print_function

import gc
import logging
import os
import sys
from contextlib import nullcontext
import timeit
import argparse
from tqdm import tqdm

import numpy as np
import torch
import deepspeed

import torchvision
import torch.optim as optim


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
            loss = model(data)
#                Y_prob = model.forward(data)
 #               loss = self.loss_function(Y_prob, label.float())

            #if train:
            #    scaler.scale(loss).backward()
            #    scaler.step(self.optimizer)
            #    scaler.update()
            #    self.optimizer.zero_grad()
            #    torch.cuda.empty_cache()

            print(loss.mean())
#            epoch_loss += np.mean(loss)

            model.backward(loss.mean())
            model.step()

            if step >= 5 and self.check:
                break
        return


def main():
    print(f"Work in {os.getcwd()}")
    parser = argparse.ArgumentParser()
    parser.add_argument('-lrank', '--local_rank', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.rank = int(os.environ['RANK'])
    args.local_rank = args.rank

    ds_config = {
    "train_batch_size": 2,
    "steps_per_print": 2000,
    "optimizer": {
        "type": "Adam",
        "params": {
        "lr": 0.001,
        "betas": [
            0.8,
            0.999
        ],
        "eps": 1e-8,
        "weight_decay": 3e-7
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
        "warmup_min_lr": 0,
        "warmup_max_lr": 0.001,
        "warmup_num_steps": 1000
        }
    },
    "gradient_clipping": 1.0,
    "prescale_gradients": False,
    "bf16": {
        "enabled": False
    },
    "fp16": {
        "enabled": True,
        "fp16_master_weights_and_grads": False,
        "loss_scale": 0,
        "loss_scale_window": 500,
        "hysteresis": 2,
        "min_loss_scale": 1,
        "initial_scale_power": 15
    },
    "wall_clock_breakdown": False,
    "zero_optimization": {
        "stage": 1,
        "allgather_partitions": True,
        "reduce_scatter": True,
        "allgather_bucket_size": 50000000,
        "reduce_bucket_size": 50000000,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "cpu_offload": False
    }
    }
    
    print(torch.cuda.device_count())
    deepspeed.init_distributed()
    #model = torchvision.models.resnet18(num_classes=10).cuda()
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False).cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters()) # note that we need to pre-process parameters

    model, _, _, __ = deepspeed.initialize(args=args, model=model, model_parameters=parameters, config=ds_config)


    np.random.seed(0)
    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        print('\nGPU is ON!')

    logging.info('Init Model')

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
