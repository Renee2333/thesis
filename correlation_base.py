import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 读取 CSV 文件
df = pd.read_csv("/binf-isilon/alab/data/students/ruiningcui/thesis/motif_info_thresh_500_enhancers_hepg2.csv")

# 解析冒号分隔的数值字符串为数组
def parse_col(col):
    return col.astype(str).str.split(":").apply(lambda x: np.array(list(map(float, x))))

# 转换列
gradxinp = parse_col(df["gradxinp_0"])
ism = parse_col(df["ism_0"])
isa = parse_col(df["isa_0"])

# 每 1000 行为一组，展开为一维并计算相关系数
group_size = 1000
n_groups = len(df) // group_size

cor_isa_gradxinp, cor_isa_ism, cor_gradxinp_ism = [], [], []

for i in range(n_groups):
    start = i * group_size
    end = start + group_size

    gradxinp_block = np.concatenate(gradxinp[start:end].values)
    isa_block = np.concatenate(isa[start:end].values)
    ism_block = np.concatenate(ism[start:end].values)

    cor_isa_gradxinp.append(pearsonr(isa_block, gradxinp_block)[0])
    cor_isa_ism.append(pearsonr(isa_block, ism_block)[0])
    cor_gradxinp_ism.append(pearsonr(gradxinp_block, ism_block)[0])

# 构建 DataFrame
cor_df = pd.DataFrame({
    "Correlation": cor_isa_gradxinp + cor_isa_ism + cor_gradxinp_ism,
    "Type": (["ISA vs Gradxinp"] * len(cor_isa_gradxinp) +
             ["ISA vs ISM"] * len(cor_isa_ism) +
             ["Gradxinp vs ISM"] * len(cor_gradxinp_ism))
})

# 保存叠加密度图
plt.figure(figsize=(10, 6))
sns.kdeplot(data=cor_df, x="Correlation", hue="Type", fill=True, common_norm=False, alpha=0.4)
plt.title("Correlation Between ISA / Gradxinp / ISM at base level (Grouped by 1000 rows)", fontsize=16)  # 标题字号
plt.xlabel("Correlation", fontsize=14)  # X轴标签字号
plt.ylabel("Density", fontsize=14)  # Y轴标签字号
plt.xticks(fontsize=12)  # X轴刻度字号
plt.yticks(fontsize=12)  # Y轴刻度字号
plt.xlim(-1, 1)
plt.grid(False)
plt.tight_layout()
plt.savefig("correlation_base_combined.pdf")
plt.close()

# 保存一行三个图的密度图
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

sns.kdeplot(x=cor_isa_gradxinp, ax=axes[0], fill=True, color="skyblue")
axes[0].set_title("ISA vs Gradxinp", fontsize=16)  # 标题字号
axes[0].set_xlim(-1, 1)
axes[0].set_xlabel("Correlation", fontsize=14)  # X轴标签字号
axes[0].set_ylabel("Density", fontsize=14)  # Y轴标签字号
axes[0].tick_params(axis='both', labelsize=12)  # 刻度字号
axes[0].grid(False)

sns.kdeplot(x=cor_isa_ism, ax=axes[1], fill=True, color="lightgreen")
axes[1].set_title("ISA vs ISM", fontsize=16)  # 标题字号
axes[1].set_xlim(-1, 1)
axes[1].set_xlabel("Correlation", fontsize=14)  # X轴标签字号
axes[1].tick_params(axis='both', labelsize=12)  # 刻度字号
axes[1].grid(False)

sns.kdeplot(x=cor_gradxinp_ism, ax=axes[2], fill=True, color="salmon")
axes[2].set_title("Gradxinp vs ISM", fontsize=16)  # 标题字号
axes[2].set_xlim(-1, 1)
axes[2].set_xlabel("Correlation", fontsize=14)  # X轴标签字号
axes[2].tick_params(axis='both', labelsize=12)  # 刻度字号
axes[2].grid(False)

plt.tight_layout()
plt.savefig("correlation_base_triple.pdf")
plt.close()
