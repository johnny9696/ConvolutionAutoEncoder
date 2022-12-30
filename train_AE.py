import json
import os
import argparse
import math
import sys

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from mel_data import Mel_loader, MelCollate
from StackedAE import Convolution_Auto_Encoder as model


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import commons
import utils
import audio_processing as ap

import torch.multiprocessing as mp

global_step=1
config_path='/home/caijb/Desktop/Zero shot GLow TTS/Stacked Auto Encoder/configs/base.json'
save_path='/config.json'

def main():
    assert torch.cuda.is_available()
    n_gpus = torch.cuda.current_device()
    
    #writing config data on the model dir
    with open(config_path,"r") as f:
        data = f.read()
    config = json.loads(data)
    hps = utils.HParams(**config)
    #if no dir 
    if not os.path.exists(hps.train.model_dir):
        os.makedirs(hps.train.model_dir)
    with open(hps.train.model_dir+save_path,'w') as f:
        f.write(data)
    
    torch.manual_seed(hps.train.seed)
    hps.n_gpus = torch.cuda.device_count()
  
    hps.batch_size=int(hps.train.batch_size/hps.n_gpus)
    if hps.n_gpus>1:
        mp.spawn(train_and_eval,nprocs=hps.n_gpus,args=(hps.n_gpus,hps,))
    else:   
        train_and_eval(0,hps.n_gpus,hps)

def train_and_eval(rank,n_gpu, hps):
    global global_step

    if hps.n_gpus>1:
        os.environ["MASTER_ADDR"]="localhost"
        os.environ["MASTER_PORT"]="12355"
        dist.init_process_group(backend='nccl',init_method='env://',world_size=n_gpu,rank=rank)

    if rank == 0:
        logger = utils.get_logger(hps.train.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.train.model_dir)
        writer = SummaryWriter(log_dir=hps.train.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.train.model_dir, "eval"))

    device = torch.device("cuda:{:d}".format(rank))

    #load_train_dataset
    train_dataset = Mel_loader(hps.data.training_files,hps.data)
    collate_fn = MelCollate(hps.data.slice_length, hps)
    train_loader = DataLoader(train_dataset, num_workers=4, shuffle=False,
      batch_size=hps.train.batch_size, pin_memory=True,
      drop_last=True, collate_fn=collate_fn)
    if rank == 0 :
        eval_dataset = Mel_loader(hps.data.validation_files,hps.data)
        eval_loader = DataLoader(eval_dataset, num_workers=4, shuffle=False,
      batch_size=hps.train.batch_size, pin_memory=True,
      drop_last=True, collate_fn=collate_fn)

    AE_model = model(encoder_dim=hps.model.encoder_dim, hidden_1dim=hps.model.hidden_dim1,
    hidden_2dim=hps.model.hidden_dim2, kernel=hps.model.kernel).to(device)

    #mutli_gpu_Set
    if hps.n_gpus>1:
        print("Multi GPU Setting Start")
        AE_model=DistributedDataParallel(AE_model,device_ids=[rank]).to(device)
        print("Multi GPU Setting Finish")

    optimizer = torch.optim.Adam(AE_model.parameters(), lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

    epoch_str = 1
    global_step = 0

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train(rank, device, epoch, hps, AE_model, optimizer, train_loader,logger, writer)
            eval(rank, device, epoch, hps, AE_model, optimizer, eval_loader, logger, writer_eval)

            utils.save_checkpoint(AE_model, optimizer, hps.train.learning_rate, epoch, os.path.join(hps.train.model_dir, "G_{}.pth".format(epoch)))
        else : 
            train(rank, device, epoch,  hps, AE_model, optimizer, train_loader, None, None)

def train(rank, device, epoch, hps, model, optimizer, train_loader, logger, writer):
    global global_step

    model.train()
    loss = nn.MSELoss()

    for batch_id,(mel_padded) in enumerate(train_loader):
        mel_padded = mel_padded.to(device)
        

        optimizer.zero_grad()
        mel_hat = model(mel_padded)
        output = loss(mel_padded,mel_hat)
        output.backward()
        optimizer.step()

        if rank == 0 :
            if batch_id % hps.train.log_interval == 0:
                image_ori = mel_padded[0].squeeze()
                image_hat = mel_hat[0].squeeze()

                logger.info('Train Epoch : {}, step : {} , Loss : {}'.format(epoch, batch_id*epoch, output.item()))

                utils.summarize(
                    writer = writer,
                    global_step = global_step,
                    images = {"mel_original" : utils.plot_spectrogram_to_numpy(image_ori.data.cpu().numpy()),
                    "mel_predict" : utils.plot_spectrogram_to_numpy(image_hat.data.cpu().numpy())},
                     scalars = {"/Loss" : output.item()}
                )
        global_step += 1


def eval(rank, device, epoch, hps, model, optimizer, eval_loader, logger,  writer):
    global global_step

    model.eval()
    loss = nn.MSELoss()
    with torch.no_grad():
        for batch_id, (mel_padded) in enumerate(eval_loader):
            mel_padded = mel_padded.to(device)

            optimizer.zero_grad()
            mel_hat = model(mel_padded)
            output = loss(mel_padded,mel_hat)

            optimizer.step()

            if rank == 0 :
                if batch_id % hps.train.log_interval == 0 :
                    image_ori = mel_padded[0].squeeze()
                    image_hat = mel_hat[0].squeeze()

                    logger.info('Eval Epoch : {}, step : {} , Loss : {}'.format(epoch, batch_id*epoch, output.item()))

                    utils.summarize(
                        writer = writer,
                        global_step = global_step,
                        images = {"mel_original" : utils.plot_spectrogram_to_numpy(image_ori.data.cpu().numpy()),
                        "mel_predict" : utils.plot_spectrogram_to_numpy(image_hat.data.cpu().numpy())},
                        scalars = {"/Loss" : output.item()}
                        )
        global_step += 1

if __name__ == "__main__":
    main()