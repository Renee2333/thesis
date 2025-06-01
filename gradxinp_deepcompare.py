"""
This modules calculates gradxinp.
Output format: 
    Row names are "SeqX_TrackY", where X is the index of sequence, Y is the index of track.
"""

import pandas as pd
import numpy as np
import torch
import math
from captum.attr import InputXGradient

from seq_ops import SeqExtractor
from utils import find_available_gpu
from seq_ops import seq2x


BATCH_SIZE=8192



def backprop(interpreter, seqs, target, device):
    res=np.empty((0, len(seqs[0])))
    for i in range(math.ceil(len(seqs) / BATCH_SIZE)):
        start_idx, end_idx = BATCH_SIZE*i, min(BATCH_SIZE*(i+1), len(seqs))
        x=seq2x(seqs[start_idx:end_idx],device)
        x.requires_grad_()
        temp=interpreter.attribute(x, target=target).cpu().detach().numpy().sum(axis=1)
        res=np.concatenate((res,temp),axis=0)
    return res




def compute_gradxinp(bed_file,
                     seq_extractor=SeqExtractor("/isdata/alab/people/pcr980/Resource/hg38.fa"),targets=list(range(16)),
                     model=torch.load("/isdata/alab/people/pcr980/DeepCompare/DeepCompare_model/model.h5"),
                     device=torch.device("cuda:"+find_available_gpu())):
    """
    Return gradxinp in pandas dataframe
    Args:
        seqs: sequence, either a string or a list of strings.
        target: number(s) of track to use. If empty, return gradxinp for all tracks
    Return:
        df_gradxinp: gradxinp in pandas dataframe, with "SeqX_trackY" as row names.
                      SeqX has same order as seqs
    """
    if isinstance(bed_file,str):
        bed_file=pd.read_csv(bed_file,sep="\t",header=None)
    if isinstance(targets,int):
        targets=[targets]
    if isinstance(device,str):
        device=torch.device(f"cuda:{device}")
    
    # resize region, fix start
    bed_file.iloc[:,2]=bed_file.iloc[:,1]+599
    # get sequences
    seqs=bed_file.apply(lambda row: seq_extractor.get_seq(row.tolist()), axis=1).tolist()

    model.to(device)
    interpreter = InputXGradient(model)
    gradxinp=[]
    for target in targets:
        gradxinp.append(backprop(interpreter, seqs, target, device))
    gradxinp=np.array(gradxinp) # shape: (num_targets, num_seqs, seq_len)
    gradxinp=np.transpose(gradxinp, (1,0,2))
    gradxinp=gradxinp.reshape((len(targets)*len(seqs),len(seqs[0])))
    df_gradxinp=pd.DataFrame(gradxinp)
    # create indices: chr:start-end_Track{}
    indices=[f"{bed_file.iloc[i,0]}:{bed_file.iloc[i,1]}-{bed_file.iloc[i,2]}_Track{target}" for i in range(len(bed_file)) for target in targets]
    df_gradxinp.index=indices
    return df_gradxinp.round(3)





