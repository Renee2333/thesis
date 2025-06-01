import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
prediction_df = pd.read_csv("/binf-isilon/alab/data/students/ruiningcui/thesis/enformer/prediction.csv")
gradxinp_df = pd.read_csv("/binf-isilon/alab/data/students/ruiningcui/thesis/enformer/Enformer_gradxinp.csv")
isa_df1 = pd.read_csv("/binf-isilon/alab/data/students/ruiningcui/thesis/enformer/ISA_colab.csv")
isa_df2 = pd.read_csv("/binf-isilon/alab/data/students/ruiningcui/thesis/enformer/ISA.csv")
isa_df3 = pd.read_csv("/binf-isilon/alab/data/students/ruiningcui/thesis/enformer/ISA_colab2.csv")  # 新增 ISA 文件
ism_df1 = pd.read_csv("/binf-isilon/alab/data/students/ruiningcui/thesis/enformer/ism_colab.csv")
ism_df2 = pd.read_csv("/binf-isilon/alab/data/students/ruiningcui/thesis/enformer/ism.csv")
ism_df3 = pd.read_csv("/binf-isilon/alab/data/students/ruiningcui/thesis/enformer/ism2_formatted.csv")  # 新增 ISM 文件
ism_df2.columns = ism_df1.columns
ism_df3.columns = ism_df1.columns
isa_df2.columns = isa_df1.columns
isa_df3.columns = isa_df1.columns
# 分别合并 ISA 和 ISM 数据
isa_df = pd.concat([isa_df1, isa_df2, isa_df3]).drop_duplicates(subset=["region"])
ism_df = pd.concat([ism_df1, ism_df2, ism_df3]).drop_duplicates(subset=["region"])

# 根据 ISM 数据筛选 ISA、Gradxinp 和 Prediction 数据
filtered_regions = ism_df["region"]
isa_df = isa_df[isa_df["region"].isin(filtered_regions)]
gradxinp_df = gradxinp_df[gradxinp_df["region"].isin(filtered_regions)]
prediction_df = prediction_df[prediction_df["region"].isin(filtered_regions)]

# 处理 Gradxinp 和 ISM 数据：取最大绝对值和最大值
def extract_abs_max(values):
    return max(abs(v) for v in values)

def extract_max(values):
    return max(values)

# 计算相对位置并提取 Gradxinp 和 ISM 数据
def get_relative_values(row, gradxinp_df, ism_df):
    region_start = int(row["region"].split(":")[1].split("-")[0])  # 获取 region 起始位置
    motif_start = int(row["start_motif"]) - region_start  # 计算 motif 相对起始位置
    motif_end = int(row["end_motif"]) - region_start  # 计算 motif 相对结束位置

    # 提取 Gradxinp 和 ISM 数据
    gradxinp_values = gradxinp_df.loc[gradxinp_df["region"] == row["region"]].iloc[0, 1:].values[motif_start:motif_end]
    ism_values = ism_df.loc[ism_df["region"] == row["region"]].iloc[0, 1:].values[motif_start:motif_end]

    gradxinp_abs_max = extract_abs_max(gradxinp_values)
    ism_max = extract_max(ism_values)

    return pd.Series({"gradxinp_abs_max": gradxinp_abs_max, "ism_max": ism_max})

# 提取 Gradxinp 和 ISM 的最大值
gradxinp_ism_values = isa_df.apply(get_relative_values, axis=1, gradxinp_df=gradxinp_df, ism_df=ism_df)
isa_df = pd.concat([isa_df, gradxinp_ism_values], axis=1)

# 合并 Prediction 数据
isa_df = isa_df.merge(prediction_df, on="region", how="inner")

# 绘制散点图
plt.figure(figsize=(15, 5))

# Prediction vs ISA
plt.subplot(1, 3, 1)
isa_data = isa_df[["pos_0", "average_change"]].dropna().replace([np.inf, -np.inf], np.nan).dropna()
correlation_isa = pearsonr(isa_data["pos_0"], isa_data["average_change"])[0]
plt.scatter(isa_data["pos_0"], isa_data["average_change"], alpha=0.5, color="skyblue")
plt.title("Prediction vs ISA", fontsize=16)  # 标题字号
plt.xlabel("Prediction", fontsize=14)  # X轴标签字号
plt.ylabel("ISA", fontsize=14)  # Y轴标签字号
plt.xticks(fontsize=12)  # X轴刻度字号
plt.yticks(fontsize=12)  # Y轴刻度字号
plt.grid(True)
plt.text(0.95, 0.95, f"Corr: {correlation_isa:.2f}", transform=plt.gca().transAxes,
         fontsize=14, verticalalignment='top', horizontalalignment='right')

# Prediction vs Gradxinp
plt.subplot(1, 3, 2)
gradxinp_data = isa_df[["pos_0", "gradxinp_abs_max"]].dropna().replace([np.inf, -np.inf], np.nan).dropna()
correlation_gradxinp = pearsonr(gradxinp_data["pos_0"], gradxinp_data["gradxinp_abs_max"])[0]
plt.scatter(gradxinp_data["pos_0"], gradxinp_data["gradxinp_abs_max"], alpha=0.5, color="lightgreen")
plt.title("Prediction vs Gradxinp", fontsize=16)  # 标题字号
plt.xlabel("Prediction", fontsize=14)  # X轴标签字号
plt.ylabel("Gradxinp", fontsize=14)  # Y轴标签字号
plt.xticks(fontsize=12)  # X轴刻度字号
plt.yticks(fontsize=12)  # Y轴刻度字号
plt.grid(True)
plt.text(0.95, 0.95, f"Corr: {correlation_gradxinp:.2f}", transform=plt.gca().transAxes,
         fontsize=14, verticalalignment='top', horizontalalignment='right')

# Prediction vs ISM
plt.subplot(1, 3, 3)
ism_data = isa_df[["pos_0", "ism_max"]].dropna().replace([np.inf, -np.inf], np.nan).dropna()
correlation_ism = pearsonr(ism_data["pos_0"], ism_data["ism_max"])[0]
plt.scatter(ism_data["pos_0"], ism_data["ism_max"], alpha=0.5, color="salmon")
plt.title("Prediction vs ISM", fontsize=16)  # 标题字号
plt.xlabel("Prediction", fontsize=14)  # X轴标签字号
plt.ylabel("ISM", fontsize=14)  # Y轴标签字号
plt.xticks(fontsize=12)  # X轴刻度字号
plt.yticks(fontsize=12)  # Y轴刻度字号
plt.grid(True)
plt.text(0.95, 0.95, f"Corr: {correlation_ism:.2f}", transform=plt.gca().transAxes,
         fontsize=14, verticalalignment='top', horizontalalignment='right')

plt.tight_layout()
plt.savefig("enformer_prediction_correlation.pdf")
plt.close()