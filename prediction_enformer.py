import tensorflow as tf
import tensorflow_hub as hub
import joblib
import gzip
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
from loguru import logger
import csv


# set log
logger.add("prediction.log", rotation="500MB", level="INFO")

# --------------- #
# set path 
# --------------- #
# set the base directory
BASE_DIR = "/binf-isilon/alab/data/students/ruiningcui/thesis/enformer/data"
# other files
model_path = "https://tfhub.dev/deepmind/enformer/1"
# set the paths for fasta and vcf
fasta_file = os.path.join(BASE_DIR, "genome/genome.fa")
clinvar_vcf = os.path.join(BASE_DIR, "clinvar/clinvar.vcf.gz")

# --------------- #
# functions
# --------------- #
SEQUENCE_LENGTH = 393216
class Enformer:

  def __init__(self, tfhub_url):
    self._model = hub.load(tfhub_url).model

  def predict_on_batch(self, inputs):
    predictions = self._model.predict_on_batch(inputs)
    return {k: v.numpy() for k, v in predictions.items()}

  @tf.function
  def contribution_input_grad(self, input_sequence,
                              target_mask, output_head='human'):
    input_sequence = input_sequence[tf.newaxis]

    target_mask_mass = tf.reduce_sum(target_mask)
    with tf.GradientTape() as tape:
      tape.watch(input_sequence)
      prediction = tf.reduce_sum(
          target_mask[tf.newaxis] *
          self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

    input_grad = tape.gradient(prediction, input_sequence) * input_sequence
    input_grad = tf.squeeze(input_grad, axis=0)
    return tf.reduce_sum(input_grad, axis=-1)

# --------------- #
# extract fasta sequence
# --------------- #
class FastaStringExtractor:
    
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()

def one_hot_encode(sequence):
    import kipoiseq.transforms.functional
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

# --------------- #
# compute contribution scores
# --------------- #
def compute_contribution_scores(model, fasta_extractor, target_interval):
    sequence_one_hot = kipoiseq.transforms.functional.one_hot_dna(
        fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH))
    ).astype(np.float32)

    predictions = model.predict_on_batch(sequence_one_hot[np.newaxis])['human'][0]
    return (predictions[446:450]).mean()


# --------------- #
# process csv
# --------------- #
def process_csv(input_csv, output_csv, model, fasta_extractor):
    logger.info("Reading input CSV...")
    df = pd.read_csv(input_csv)

    # 读取并合并三个 ISM 文件
    ism_files = [
        "/binf-isilon/alab/data/students/ruiningcui/thesis/enformer/ism_colab.csv",
        "/binf-isilon/alab/data/students/ruiningcui/thesis/enformer/ism2_formatted.csv",
        "/binf-isilon/alab/data/students/ruiningcui/thesis/enformer/ism.csv"
    ]
    logger.info("Reading and merging ISM files...")
    ism_dfs = [pd.read_csv(file) for file in ism_files]
    combined_ism_df = pd.concat(ism_dfs).drop_duplicates(subset=["region"])
    logger.info(f"Combined ISM regions: {len(combined_ism_df)}")

    # 筛选 input_csv，只保留 region 能在合并后的 ISM 表格中找到的行
    filtered_df = df[df["region"].isin(combined_ism_df["region"])]

    # 去重，确保 region 唯一
    filtered_df = filtered_df.drop_duplicates(subset=["region"])
    logger.info(f"Filtered and unique input regions: {len(filtered_df)}")

    # 检查是否有已存在的输出文件
    processed_regions = set()
    if os.path.exists(output_csv):
        logger.info(f"Resuming from existing file: {output_csv}")
        with open(output_csv, mode='r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                processed_regions.add(row[0])  # 第一列是 region

    logger.info(f"Already processed {len(processed_regions)} regions. Resuming...")

    # 获取所有的 region
    regions = []
    for region in filtered_df["region"].dropna():
        if region in processed_regions:
            logger.info(f"Skipping already processed region: {region}")
            continue
        chrom, coords = region.split(':')
        start, end = map(int, coords.split('-'))
        end += 1  # 让区间包含最后一个碱基
        regions.append((region, chrom, start, end))

    logger.info(f"Processing {len(regions)} regions...")

    # 打开输出文件，追加写入
    with open(output_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if len(processed_regions) == 0:
            # 如果文件是新建的，写入表头
            writer.writerow(['region'] + [f'pos_{i}' for i in range(600)])

        # 逐行处理每个 region
        for i, (region, chrom, start, end) in enumerate(regions, 1):
            target_interval = Interval(chrom, start, end)
            contribution_scores = compute_contribution_scores(model, fasta_extractor, target_interval)

            # 如果 contribution_scores 是单一值，则直接转换为浮点数
            if isinstance(contribution_scores, (np.float32, float)):
                contribution_scores = [float(contribution_scores)]  # 转换为单元素列表
            else:
                contribution_scores = list(map(float, contribution_scores))  # 转换为浮点数列表

            writer.writerow([region] + contribution_scores)
            logger.info(f"Processed {i}/{len(regions)}: {region}")

    logger.info(f"Saved results to {output_csv}")


# 主函数
if __name__ == "__main__":
    logger.info("Initializing model...")
    model = Enformer(model_path)
    fasta_extractor = FastaStringExtractor(fasta_file)

    input_csv = "/binf-isilon/alab/data/students/ruiningcui/thesis/motif_info_thresh_500_enhancers_hepg2.csv"
    output_csv = "prediction.csv"
    process_csv(input_csv, output_csv, model, fasta_extractor)
