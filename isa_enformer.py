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
import matplotlib as mpl
import seaborn as sns
import csv
from loguru import logger
import os
import subprocess

# set log
logger.add("isa.log", rotation="500MB", level="INFO")

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

def calculate_changes(reference_prediction, alternate_prediction):
    positions_i = [446, 447, 448, 449, 450]
    positions_j = [27, 91, 234]
    
    changes = []
    for i in positions_i:
        for j in positions_j:
            change = alternate_prediction[i][j] - reference_prediction[i][j]
            changes.append(change)
    
    return np.mean(changes)

def process_csv(input_csv, output_csv, model, fasta_extractor):
    logger.info("Reading input CSV...")
    df = pd.read_csv(input_csv)

    # 打开 CSV 文件写入
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['region', 'protein', 'start_motif', 'end_motif', 'average_change'])

        for i, row in df.iterrows():
            region = row['region']
            protein = row['protein']
            start_motif, end_motif = row['start'], row['end']

            # 解析 region，获取染色体和 start_region
            chrom, coords = region.split(':')
            start_region, end_region = map(int, coords.split('-'))
            end_region += 1  # 让区间包含最后一个碱基

            # 提取参考序列
            target_interval = kipoiseq.Interval(chrom, start_region, end_region)
            reference = fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH))
            reference_sequence = list(reference)

            # 计算 start_motif 在 reference_sequence 里的相对索引
            relative_start = start_motif - start_region
            relative_end = end_motif - start_region

            # 替换 motif 位置的碱基为 'N'
            mutated_sequence = reference_sequence[:]
            mutated_sequence[196308+relative_start:196308+relative_end] = ['N'] * (relative_end - relative_start)
            mutated_reference = ''.join(mutated_sequence)

            # One-hot 编码
            mutated_reference_one_hot = one_hot_encode(mutated_reference)
            reference_one_hot = one_hot_encode(reference)

            # 提取变异后的预测值
            alternate_prediction = model.predict_on_batch(mutated_reference_one_hot[np.newaxis])['human'][0]
            reference_prediction = model.predict_on_batch(reference_one_hot[np.newaxis])['human'][0]

            # 计算变化值
            average_change = calculate_changes(reference_prediction, alternate_prediction)

            # 写入 CSV
            writer.writerow([region, protein, start_motif, end_motif, average_change])
            logger.info(f"Processed {i+1}/{len(df)}: {region}, {protein}, {start_motif}-{end_motif}, Change: {average_change}")

    logger.info(f"Results saved to {output_csv}")

# --------------- #
# main function
# --------------- #
if __name__ == "__main__":
    logger.info("Initializing model...")
    model = Enformer(model_path)
    fasta_extractor = FastaStringExtractor(fasta_file)

    input_csv = "/binf-isilon/alab/data/students/ruiningcui/thesis/motif_info_thresh_500_enhancers_hepg2.csv"
    output_csv = "ISA.csv"
    process_csv(input_csv, output_csv, model, fasta_extractor)
