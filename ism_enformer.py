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
import tensorflow as tf
import csv
from loguru import logger
import tensorflow as tf
import tensorflow_hub as hub
import joblib
import gzip
import kipoiseq
import os
import subprocess

# set log
logger.add("ism.log", rotation="500MB", level="INFO")

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

import numpy as np
import matplotlib.pyplot as plt
import csv
from loguru import logger

# 假设`fasta_extractor`和`seq_extractor`已经定义，`model`已经加载

# 目标区间
target_interval = kipoiseq.Interval('chr6', 33183099, 33183698)
reference = fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH))

# 参考序列的碱基列表
reference_sequence = list(reference)

# 需要计算的变异位置
start_pos = 196308
end_pos = 196908

# 保存每个碱基替换后的平均变化值
average_changes = []

# 计算变化的函数
def calculate_changes(reference_prediction, alternate_prediction):
    # 计算在指定的 i 和 j 位置的变化值的平均
    positions_i = [446, 447, 448, 449, 450]
    positions_j = [27, 91, 234]
    
    changes = []
    for i in positions_i:
        for j in positions_j:
            change = alternate_prediction[i][j] - reference_prediction[i][j]
            changes.append(change)
    
    # 返回这些变化的平均值
    return np.mean(changes)

# 遍历每个位置（从start_pos到end_pos），替换该位置的碱基并计算变化
for pos in range(start_pos, end_pos + 1):
    # 当前位点的原始碱基
    original_base = reference_sequence[pos - start_pos]
    
    # 记录该位置的三个替代碱基（A, T, C, G）的变化
    base_changes = []
    
    # 遍历所有其他三个碱基 (A, T, C, G) 替换当前位点
    for new_base in ['A', 'T', 'C', 'G']:
        if new_base != original_base:
            # 替换碱基
            reference_sequence[pos - start_pos] = new_base
            
            # 生成替换后的序列
            mutated_reference = ''.join(reference_sequence)
            mutated_interval = kipoiseq.Interval(target_interval.chrom, target_interval.start, target_interval.end)
            mutated_interval_sequence = fasta_extractor.extract(mutated_interval.resize(SEQUENCE_LENGTH))

            # 提取变异后的序列并计算预测
            alternate_prediction = model.predict_on_batch(one_hot_encode(mutated_reference)[np.newaxis])['human'][0]

            # 计算参考序列和替代序列之间的变化值
            reference_prediction = model.predict_on_batch(one_hot_encode(reference)[np.newaxis])['human'][0]
            
            # 计算变化值并记录
            change = calculate_changes(reference_prediction, alternate_prediction)
            base_changes.append(change)

            # 恢复原始序列的碱基，准备下一次替换
            reference_sequence[pos - start_pos] = original_base
    
    # 计算该位置三个变化的平均值
    average_changes.append(np.mean(base_changes))

    # 输出日志信息，告知该位置的计算已经完成
    logger.info(f"Successfully processed position {pos} with average change: {np.mean(base_changes)}")

# 绘制图表
positions = list(range(start_pos, end_pos + 1))  # x轴为位置

plt.figure(figsize=(10, 6))
plt.plot(positions, average_changes, label="Average Change", color='b')
plt.xlabel('Position')
plt.ylabel('Average Change')
plt.title('Average Change for Each Position')
plt.grid(True)

# **保存图片**
image_filename = "average_changes.png"
plt.savefig(image_filename, dpi=300)
logger.info(f"Plot saved as {image_filename}")

# 输出到CSV文件
csv_filename = 'average_changes.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 写入表头
    header = ['Interval'] + [f'Position_{i}' for i in range(start_pos, end_pos + 1)]
    writer.writerow(header)
    
    # 写入数据（interval 名称及其所有的变化值）
    writer.writerow([str(target_interval)] + average_changes)

# 输出日志信息，告知CSV文件已成功保存
logger.info(f"Average changes have been written to {csv_filename}")
