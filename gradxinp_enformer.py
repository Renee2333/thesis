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
logger.add("script.log", rotation="500MB", level="INFO")

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
    target_mask = np.zeros_like(predictions)

    # 27, 91, 234 are the tracks for Hepg2
    # 446, 447, 448, 449, 450 are the target intervals
    for idx in [446, 447, 448, 449, 450]:
        target_mask[idx, 27] = 1
        target_mask[idx, 91] = 1
        target_mask[idx, 234] = 1

    contribution_scores = model.contribution_input_grad(sequence_one_hot, target_mask).numpy()
    return contribution_scores[196308:196908]

# --------------- #
# process csv
# --------------- #
def process_csv(input_csv, output_csv, model, fasta_extractor):
    logger.info("Reading input CSV...")
    df = pd.read_csv(input_csv)

    df['region'] = df['region'].drop_duplicates()
    regions = []

    for region in df['region'].dropna():
        chrom, coords = region.split(':')
        start, end = map(int, coords.split('-'))
        end += 1  

        if end - start == 600:
            regions.append((region, chrom, start, end))

    logger.info(f"Processing {len(regions)} regions...")

    # Open the output CSV file in write mode
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Write the header
        writer.writerow(['region'] + [f'pos_{i}' for i in range(600)])

        # Process each region and write to CSV line by line
        for i, (region, chrom, start, end) in enumerate(regions, 1):
            target_interval = Interval(chrom, start, end)
            contribution_scores = compute_contribution_scores(model, fasta_extractor, target_interval)
            writer.writerow([region] + list(contribution_scores))
            logger.info(f"Processed {i}/{len(regions)}: {region}")

    logger.info(f"Saved results to {output_csv}")

# --------------- #
# main function
# --------------- #
if __name__ == "__main__":
    logger.info("Initializing model...")
    model = Enformer(model_path)
    fasta_extractor = FastaStringExtractor(fasta_file)

    input_csv = "/binf-isilon/alab/data/students/ruiningcui/thesis/motif_info_thresh_500_enhancers_hepg2.csv"  
    output_csv = "Enformer_gradxinp.csv"
    process_csv(input_csv, output_csv, model, fasta_extractor)

