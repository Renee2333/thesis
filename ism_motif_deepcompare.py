import pandas as pd
import numpy as np
import torch
from loguru import logger
import itertools
import csv

import sys
sys.path.insert(1,"/isdata/alab/people/pcr980/Scripts_python/")
from utils import find_available_gpu
from prediction import compute_predictions
from seq_ops import ablate_motifs
from seq_ops import SeqExtractor

# set log
logger.add("ism_motif2000.log", rotation="500MB", level="INFO")

#------------------------------
# Base level ISM and ISA
#------------------------------



def flatten_list(l):
    return [item for sublist in l for item in sublist]


def _mutate_single_base(seq,location,alt_base):
    """
    allows seq[location]==alt_base
    """
    return seq[:location]+alt_base+seq[location+1:] 



def _mutate_one_seq(seq,idx,mode):
    """
    Returns a dataframe of position, ref_seq, mut_seq
    """
    positions=flatten_list([[i]*5 for i in range(len(seq))])
    ref_bases=flatten_list([[seq[i]]*5 for i in range(len(seq))])
    mut_bases=flatten_list([['A','C','G','T','N'] for i in range(len(seq))])
    mut_seqs=flatten_list([[_mutate_single_base(seq,i,alt_base) for alt_base in ['A','C','G','T','N']] for i in range(len(seq))])
    df=pd.DataFrame({"ref_idx":f"Seq{idx}",
                     "position":positions,
                     "ref_base":ref_bases,
                     "mut_base":mut_bases,
                     "ref_seq":seq,
                     "mut_seq":mut_seqs})
    if mode=="isa":
        df=df[df["mut_base"]=="N"].reset_index(drop=True)
    elif mode=="ism":
        df=df[df["mut_base"]!="N"].reset_index(drop=True)
        df=df[df["ref_base"]!=df["mut_base"]].reset_index(drop=True)
    else:
        raise ValueError("mode must be isa or ism")
    return df




def _mutate_all_seqs(seqs,mode):
    """
    Returns a dataframe of position, ref_seq, mut_seq
    """
    if isinstance(seqs,str):
        seqs=[seqs]
    # get an empty dataframe
    df=pd.DataFrame(columns=["ref_idx","position","ref_base","mut_base","ref_seq","mut_seq"])
    for idx,seq in enumerate(seqs):
        df=df.append(_mutate_one_seq(seq,idx,mode))
    df=df.reset_index(drop=True)
    return df


def abs_max(input_list):
    return max(input_list, key=abs)





def compute_mutagenesis_score(input,mode,
                              seq_extractor=SeqExtractor("/isdata/alab/people/pcr980/Resource/hg38.fa"),
                              aggregation_method="mean",
                              device=torch.device("cuda:"+find_available_gpu())):
    """
    input format may be:
    1. a bed file with 3 columns: chromosome, start, end
    2. a list of sequences
    3. a string of sequences
    4. a dataframe with columns ["chromosome","start","end","name","region"]
    """
    try:
        try:
            # case 1: input is a bed file path
            df_bed=pd.read_csv(input,sep="\t",header=None)
        except Exception as e:
            # case 2: input is a dataframe
            logger.warning(f"{input} is not a file path, try to read it as a dataframe")
            df_bed=input.copy()
        # resize, fix the start
        df_bed.iloc[:,2]=df_bed.iloc[:,1]+599
        seqs=df_bed.apply(lambda row: seq_extractor.get_seq(row.tolist()), axis=1).tolist()
        df_ism=compute_mutagenesis_score_from_seqs(seqs,mode,aggregation_method,device)
        indices=[f"{df_bed.iloc[i,0]}:{df_bed.iloc[i,1]}-{df_bed.iloc[i,2]}_Track{target}" for i in range(len(df_bed)) for target in range(16)]
        df_ism.index=indices
    except Exception as e:
        # case 3: input is a list of sequences
        logger.warning(f"{input} is not a bed file, try to read it as a list of sequences")
        # if seqs is a string, convert it to a list
        if isinstance(input,str):
            seqs=[input]
        else:
            seqs=input
        df_ism=compute_mutagenesis_score_from_seqs(seqs,mode,aggregation_method,device)
        indices=[f"Seq{i}_Track{target}" for i in range(len(seqs)) for target in range(16)]
        df_ism.index=indices
    return df_ism
    

    




def compute_mutagenesis_score_from_seqs(seqs,mode,aggregation_method,device):
    """
    Workhorse for compute_mutagenesis_score
    mode: isa or ism
    """
    #
    df=_mutate_all_seqs(seqs,mode)
    pred_ref=compute_predictions(df["ref_seq"],device=device)
    pred_mut=compute_predictions(df["mut_seq"],device=device)
    delta=pd.DataFrame(pred_ref-pred_mut)
    delta.columns=[f"Track{i}" for i in range(16)]
    # merge df with delta
    df=pd.concat([df,delta],axis=1)
    df.drop(columns=["ref_base","mut_base","ref_seq","mut_seq"],inplace=True)
    # group by position and refseq, and calculate mean of each track
    if aggregation_method=="mean":
        df=df.groupby(["ref_idx","position"]).mean()
    if aggregation_method=="max":
        df=df.groupby(["ref_idx","position"]).agg(abs_max)
    # # Reshape the DataFrame from (num_seqs*600,16)to (num_seqs*16,600)
    num_sequences = len(df.index.get_level_values(0).unique())
    num_positions = len(df.index.get_level_values(1).unique())
    num_tracks = df.shape[1]
    array_3d = df.values.reshape(num_sequences, num_positions, num_tracks)
    array_3d = np.transpose(array_3d, (0, 2, 1))
    array_2d = array_3d.reshape(num_sequences * num_tracks, num_positions)
    df_flat = pd.DataFrame(array_2d)
    return df_flat





#------------------------------
# Motif level ISA
# Input: motif_df output by jaspar_annotator.annotate()
#------------------------------



def get_motif_isa(seq_extractor,motif_df,track_num=None, device=torch.device("cuda:"+find_available_gpu())):
    """
    Args:
        seq_extractor: SeqExtractor
        motif_df: a dataframe with columns ["chromosome","start","end","name","region"]
        track_num: int, the track number to use
    Returns:
        extra columns in motif_df: ["isa_track{track_num}","start_rel","end_rel"...]
        
    """
    if motif_df.shape[0]==0:
        logger.warning("Empty motif_df")
        return motif_df
    
    motif_df["start_seq"]=motif_df["region"].str.split(":").str[1].str.split("-").str[0].astype(int)
    motif_df["start_rel"]=motif_df["start"]-motif_df["start_seq"]
    motif_df["end_rel"]=motif_df["end"]-motif_df["start_seq"]
    # get the original and mutated sequence
    motif_df["seq_orig"]= motif_df["region"].apply(lambda x: seq_extractor.get_seq(x))
    motif_df["seq_mut"] = motif_df.apply(lambda row: ablate_motifs(row['seq_orig'], row['start_rel'], row['end_rel']), axis=1)
    # compute predictions
    pred_orig=compute_predictions(motif_df["seq_orig"],device=device)
    pred_mut=compute_predictions(motif_df["seq_mut"],device=device)
    # compute isa
    isa=pred_orig-pred_mut
    motif_df.drop(columns=["seq_orig","seq_mut"],inplace=True)
    cols=[f"isa_track{i}" for i in range(16)]
    df_isa=pd.DataFrame(isa,columns=cols)
    if track_num is not None:
        df_isa=df_isa[f"isa_track{track_num}"]
    motif_df=pd.concat([motif_df,df_isa],axis=1)
    return motif_df


def get_ism_motif(seq_extractor, motif_df, track_num=None, device=torch.device("cuda:"+find_available_gpu())):
    """
    Args:
        seq_extractor: SeqExtractor
        motif_df: a dataframe with columns ["chromosome", "start", "end", "name", "region"]
        track_num: int, the track number to use
        device: Torch device (e.g., "cuda:0" or "cpu")
    Returns:
        motif_df with additional columns: ["ism_track{track_num}", "start_rel", "end_rel", ...]
    """
    if motif_df.shape[0] == 0:
        logger.warning("Empty motif_df")
        return motif_df

    # 解析 region，提取 start 和 end
    motif_df["start_seq"] = motif_df["region"].str.split(":").str[1].str.split("-").str[0].astype(int)
    motif_df["end_seq"] = motif_df["region"].str.split(":").str[1].str.split("-").str[1].astype(int) + 1

    # 计算 motif 在序列中的相对位置
    motif_df["start_rel"] = motif_df["start"] - motif_df["start_seq"]
    motif_df["end_rel"] = motif_df["end"] - motif_df["start_seq"]

    # 获取原始序列
    motif_df["seq_orig"] = motif_df["region"].apply(lambda x: seq_extractor.get_seq(x))

    # 生成突变序列
    def mutate_motif(seq, start_rel, end_rel):
        motif_length = end_rel - start_rel
        motif_seq = seq[start_rel:end_rel]
        mutated_seqs = []

        for bases in itertools.product("ACGT", repeat=motif_length):
            new_motif = ''.join(bases)
            if new_motif == motif_seq:
                continue
            mutated = list(seq)
            for i, base in enumerate(bases):
                mutated[start_rel + i] = base
            mutated_seqs.append(''.join(mutated))

        return mutated_seqs

    for idx, row in motif_df.iterrows():
        logger.info(f"Processing row {idx + 1}/{len(motif_df)}: {row['region']}")
        try:
            seq_orig = row["seq_orig"]
            start_rel = row["start_rel"]
            end_rel = row["end_rel"]
            motif_seq = seq_orig[start_rel:end_rel]

            # 生成突变序列
            mutated_seqs = mutate_motif(seq_orig, start_rel, end_rel)
            logger.info(f"Original motif: {motif_seq}")
            logger.info(f"Number of mutated motifs: {len(mutated_seqs)}")

            # 预测原始和突变序列
            pred_ref = compute_predictions([seq_orig], device=device)
            pred_mut = compute_predictions(mutated_seqs, device=device)

            # 检查预测结果合法性
            if not isinstance(pred_mut, np.ndarray):
                pred_mut = np.array(pred_mut)
            if pred_mut.ndim != 2 or pred_mut.shape[1] != 16:
                logger.error(f"Unexpected prediction shape: {pred_mut.shape}")
                motif_df.at[idx, "ism_score"] = np.nan
                continue

            # 计算 ISM 分数：ref_track0 - 平均(mutated_track0)
            pred_ref_track0 = pred_ref[0,0]
            pred_mut_track0_mean = pred_mut[:, 0].mean()
            ism_score = pred_ref_track0 - pred_mut_track0_mean

            logger.info(f"ISM motif score: {ism_score}")

            motif_df.at[idx, "ism_score"] = ism_score

        except Exception as e:
            logger.error(f"Error processing row {idx + 1}: {e}")
            motif_df.at[idx, "ism_score"] = np.nan

    return motif_df


if __name__ == "__main__":
    # 输入文件路径和输出文件路径
    input_file = "/binf-isilon/alab/data/students/ruiningcui/thesis/Scripts_python/motif_ism_shortest2000_unique.csv"  # 替换为实际输入文件路径
    output_file = "/binf-isilon/alab/data/students/ruiningcui/thesis/Scripts_python/ism_motif2000.csv"  # 替换为实际输出文件路径

    # 初始化 GPU 设备
    gpu_index = find_available_gpu()
    device = torch.device(f"cuda:{gpu_index}" if gpu_index != "cpu" else "cpu")

    # 初始化 SeqExtractor
    seq_extractor = SeqExtractor("/isdata/alab/people/pcr980/Resource/hg38.fa")

    # 读取输入文件
    logger.info(f"Reading input file: {input_file}")
    motif_df = pd.read_csv(input_file)

    # 打开输出文件，逐行写入结果
    with open(output_file, mode='w', newline='') as f:
        # 写入表头
        writer = csv.writer(f)
        writer.writerow(["chromosome", "start", "end", "protein", "region", "ism_score"])

        # 逐行处理每个 motif
        for idx, row in motif_df.iterrows():
            try:
                # 提取序列并计算 ISM 分数
                result_row = get_ism_motif(seq_extractor, pd.DataFrame([row]), device=device)

                # 将结果写入文件
                columns_to_write = ["chromosome", "start", "end", "protein", "region", "ism_score"]
                writer.writerow(result_row[columns_to_write].iloc[0].tolist())
                logger.info(f"Row {idx + 1} written to {output_file}")
            except Exception as e:
                logger.error(f"Error processing row {idx + 1}: {e}")
                # 写入错误行，ISM 分数为 NaN
                writer.writerow([row["chromosome"], row["start"], row["end"], row["protein"], row["region"], "NaN"])