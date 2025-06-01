import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv("/binf-isilon/alab/data/students/ruiningcui/thesis/motif_info_thresh_500_enhancers_hepg2.csv")

# 处理 gradxinp_0 列：取最大绝对值
def extract_abs_max(s):
    values = [float(i) for i in str(s).split(":")]
    max_abs_val = max(abs(v) for v in values)  # 找到绝对值最大的数
    return max_abs_val  # 返回绝对值

# 处理 ism_0 列：直接取最大值
def extract_max(s):
    values = [float(i) for i in str(s).split(":")]
    max_val = max(values)  # 直接取最大值
    return max_val

# 新列
df["gradxinp_0_max"] = df["gradxinp_0"].apply(extract_abs_max)
df["ism_0_max"] = df["ism_0"].apply(extract_max)

# 取用于相关性的三列
df_corr = df[["isa_track0", "gradxinp_0_max", "ism_0_max"]].copy()

# 每1000行为一组计算相关性
group_size = 1000
n_groups = df_corr.shape[0] // group_size

cor_isa_grad, cor_isa_ism, cor_grad_ism = [], [], []

for i in range(n_groups):
    chunk = df_corr.iloc[i*group_size:(i+1)*group_size]
    cor_isa_grad.append(pearsonr(chunk["isa_track0"], chunk["gradxinp_0_max"])[0])
    cor_isa_ism.append(pearsonr(chunk["isa_track0"], chunk["ism_0_max"])[0])
    cor_grad_ism.append(pearsonr(chunk["gradxinp_0_max"], chunk["ism_0_max"])[0])

# 构建画图用的 DataFrame
cor_df = pd.DataFrame({
    "Correlation": cor_isa_grad + cor_isa_ism + cor_grad_ism,
    "Type": (["ISA vs Gradxinp"] * len(cor_isa_grad) +
             ["ISA vs ISM"] * len(cor_isa_ism) +
             ["Gradxinp vs ISM"] * len(cor_grad_ism))
})

# 保存总图（所有曲线在一张图中）
plt.figure(figsize=(10, 6))
sns.kdeplot(data=cor_df, x="Correlation", hue="Type", fill=True, alpha=0.4, common_norm=False)
plt.title("Correlation Between ISA / Gradxinp / ISM at motif level (Grouped by 1000 rows)", fontsize=16)  # 标题字号
plt.xlabel("Correlation", fontsize=14)  # X轴标签字号
plt.ylabel("Density", fontsize=14)  # Y轴标签字号
plt.xticks(fontsize=12)  # X轴刻度字号
plt.yticks(fontsize=12)  # Y轴刻度字号
plt.xlim(-1, 1)
plt.tight_layout()
plt.savefig("correlation_motif_combined_abs_max.pdf")
plt.close()

# 保存子图（1x3）
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

sns.kdeplot(x=cor_isa_grad, fill=True, ax=axs[0], color="skyblue")
axs[0].set_title("ISA vs Gradxinp", fontsize=16)  # 标题字号
axs[0].set_xlim(-1, 1)
axs[0].set_xlabel("Correlation", fontsize=14)  # X轴标签字号
axs[0].set_ylabel("Density", fontsize=14)  # Y轴标签字号
axs[0].tick_params(axis='both', labelsize=12)  # 刻度字号
axs[0].grid(False)

sns.kdeplot(x=cor_isa_ism, fill=True, ax=axs[1], color="lightgreen")
axs[1].set_title("ISA vs ISM", fontsize=16)  # 标题字号
axs[1].set_xlim(-1, 1)
axs[1].set_xlabel("Correlation", fontsize=14)  # X轴标签字号
axs[1].tick_params(axis='both', labelsize=12)  # 刻度字号
axs[1].grid(False)

sns.kdeplot(x=cor_grad_ism, fill=True, ax=axs[2], color="salmon")
axs[2].set_title("Gradxinp vs ISM", fontsize=16)  # 标题字号
axs[2].set_xlim(-1, 1)
axs[2].set_xlabel("Correlation", fontsize=14)  # X轴标签字号
axs[2].tick_params(axis='both', labelsize=12)  # 刻度字号
axs[2].grid(False)

plt.tight_layout()
plt.savefig("correlation_motif_faceted_abs_max.pdf")
plt.close()