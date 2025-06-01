#!/usr/bin/python3

"""
This module is both: 
a commandline tool: Given sequence file, write model prediction to csv.
a python function: compute_predictions()
"""

from utils import find_available_gpu
import numpy as np
import pandas as pd
import torch
import os
import math
import argparse
from loguru import logger

from seq_ops import seq2x, resize_seq


BATCH_SIZE=4096


def compute_predictions(seqs,
                        # model=torch.load("/isdata/alab/people/pcr980/DeepCompare/Models/AstigCRConv6D_Dataset_final_rep_dat_CR_MT/model.h5"),
                        model=torch.load("/isdata/alab/people/pcr980/DeepCompare/DeepCompare_model/model.h5"),
                        device=torch.device("cuda:"+find_available_gpu()),
                        verbose=False):
    """
    Args:
        seqs: a list of strings
        model: optional, loaded torch model. If absent, just use default model.
        device: optional, a torch.device(). If absent, find first GPU that has more than 5GB memory.
    Output: 
        numpy array of model output of all tracks, shape (len(seqs), 16)
    """
    if verbose:
        logger.info(f"Use device {device} to compute predictions")
    if isinstance(seqs,str):
        seqs=[seqs]
    if hasattr(seqs,"__iter__"):
        seqs=list(seqs)
    if isinstance(device,str):
        device=torch.device(f"cuda:{device}")
    model.to(device)
    model.eval()
    res=np.empty((0, 16))
    # process BATCH_SIZE sequences at a time
    for i in range(math.ceil(len(seqs) / BATCH_SIZE)):
        start_idx, end_idx = BATCH_SIZE*i, min(BATCH_SIZE*(i+1), len(seqs))
        X=seq2x(seqs[start_idx:end_idx],device)
        with torch.no_grad():
            res=np.concatenate((res,model(X).cpu().detach().numpy()),axis=0)
    return np.array(res)



def write_predictions(data_path,seq_colname,out_path,
                      model=torch.load("/isdata/alab/people/pcr980/DeepCompare/DeepCompare_model/model.h5"),
                      device=torch.device("cuda:"+find_available_gpu()),
                      variable_length=False,
                      batch_size=BATCH_SIZE):
    """
    Aimed to be used as a independent commandline tool. 
    Args:
        gpu_idx: optional, gpu index, from "0" to "7", or a torch.device()
        model_dir: optional, need to be absolute path of the directory that stores "model.h5"
    Output:
        NULL, but write model output to csv file. No header
    """
    df=pd.read_csv(data_path,chunksize=batch_size,header=0)
    query_seqs=df[seq_colname].tolist()
    if variable_length:
        query_seqs=list(map(resize_seq,query_seqs))
    df_res=pd.DataFrame(compute_predictions(query_seqs,model,device))
    df_res.to_csv(out_path, mode='a',index=False,header=False)
    



# export functions
__all__ = ['compute_predictions','write_predictions']

if __name__=="__main__":
    # parse parameters
    parser = argparse.ArgumentParser(description="Input parameters for prediction")
    parser.add_argument("-d", "--data_path", type=str, required=True, help="Data path")
    parser.add_argument("-c", "--seq_colname", type=str, required=True, help="Name of column that contains the sequences")
    parser.add_argument("-o", "--out_path", type=str, help="Output path")
    parser.add_argument("-m", "--model", type=str, default="/isdata/alab/people/pcr980/DeepCompare/DeepCompare_model/",help="Model name")
    parser.add_argument("-g", "--gpu_index", type=str, default="infer",help="GPU index")
    parser.add_argument("-v", "--variable_length", type=bool, default=False, help="Is input sequence of variable lengths? (default False)")
    parser.add_argument("-b", "--batch_size", type=int, default=16384, help="Batch size (default 16384)")
    
    args=parser.parse_args()
    
    if args.gpu_index=="infer":
        args.gpu_idx=find_available_gpu()
    
    write_predictions(args.data_path, args.seq_colname, args.out_path, 
                      torch.device(f"cuda:{args.gpu_idx}"), args.model, args.variable_length, args.batch_size)


