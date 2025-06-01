import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.ticker as ticker


import sys
sys.path.insert(1,"/binf-isilon/alab/data/students/ruiningcui/thesis/Scripts_python")
from seq_ops import SeqExtractor
from region_ops import resize_region
from seq_annotators import JasparAnnotator
from in_silico_mutagenesis import compute_mutagenesis_score, get_motif_isa
from gradxinp import compute_gradxinp

matplotlib.rcParams['pdf.fonttype']=42

#---------------
# Load annotators
#---------------
seq_extractor=SeqExtractor("/isdata/alab/people/pcr980/Resource/hg38.fa")
jaspar_hepg2_annotator=JasparAnnotator("/isdata/alab/people/pcr980/Resource/JASPAR2022_tracks/JASPAR2022_hg38.bb",
                                       chip_file="/isdata/alab/people/pcr980/Resource/ReMap2022/ReMap2022_hg38_hepg2.bed",
                                       rna_file="/isdata/alab/people/pcr980/DeepCompare/RNA_expression/expressed_tf_list_hepg2.tsv"
                                       )
jaspar_k562_annotator=JasparAnnotator("/isdata/alab/people/pcr980/Resource/JASPAR2022_tracks/JASPAR2022_hg38.bb",
                                       chip_file="/isdata/alab/people/pcr980/Resource/ReMap2022/ReMap2022_hg38_k562.bed",
                                       rna_file="/isdata/alab/people/pcr980/DeepCompare/RNA_expression/expressed_tf_list_k562.tsv"
                                       )

#---------------
# Helper functions
#---------------



def get_motifs(seq_extractor,jaspar_annotator,region,track_num,score_threshold):
    df_motif=jaspar_annotator.annotate(region)
    # remove uncertain proteins
    df_motif=df_motif[~df_motif["protein"].str.contains("::")].copy().reset_index(drop=True)
    df_chip=df_motif.loc[df_motif["chip_evidence"]==True,:].reset_index(drop=True)
    df_rest=df_motif.loc[df_motif["chip_evidence"]==False,:].reset_index(drop=True)
    df_rna=df_rest.loc[df_rest["rna_evidence"]==True,:].reset_index(drop=True)
    df_rna=df_rna.loc[df_rna["score"]>=score_threshold,:].reset_index(drop=True)
    df_motif=pd.concat([df_chip,df_rna],axis=0).reset_index(drop=True)
    # sort df_motif by "start"
    df_motif=df_motif.sort_values(by="start").reset_index(drop=True)
    # get motif isa
    df_motif=get_motif_isa(seq_extractor,df_motif,track_num)
    df_motif.rename(columns={f"isa_track{track_num}":"isa"},inplace=True)
    return df_motif



def reduce_protein_names(protein_list):
    protein_list=list(set(protein_list))
    # order alphabetically
    protein_list.sort()
    # if there are more than 2 proteins sharing same prefix of length > 4
    # only keep the prefix, followed by "s"
    # eg: hoxa9, hoxa9b, hoxa9c -> hoxa9s
    protein_dict={}
    for protein in protein_list:
        prefix=protein[:4]
        if prefix in protein_dict:
            protein_dict[prefix].append(protein)
        else:
            protein_dict[prefix]=[protein]
    prefix_list=[]
    for prefix in protein_dict:
        if len(protein_dict[prefix])>1:
            prefix_list.append(prefix+"s")
        else:
            prefix_list.append(protein_dict[prefix][0])
    # concatenate by "\n"
    return "\n".join(prefix_list)




def reduce_motifs(df_motif,window=4):
    # if start is within 3bp of another start
    # choose the top 3 based on "score"
    # concatenate protein with "\n", use largest "end" as end
    df_res=pd.DataFrame()
    while df_motif.shape[0]>0:
        current_start=df_motif.loc[0,"start_rel"]
        df_temp=df_motif[(df_motif["start_rel"]>=current_start) & (df_motif["start_rel"]<=(current_start+window))].copy().reset_index(drop=True)
        df_temp=df_temp.sort_values(by="score",ascending=False).reset_index(drop=True)
        df_temp=df_temp.iloc[:2,:]
        df_temp["protein"]=reduce_protein_names(df_temp["protein"])
        df_temp["isa"]=df_temp["isa"].mean()
        df_temp["end_rel"]=df_temp["end_rel"].max()
        df_temp["start_rel"]=df_temp["start_rel"].min()
        df_temp["start"]=df_temp["start"].min()
        df_temp["end"]=df_temp["end"].max()
        df_res=df_res.append(df_temp.iloc[0,:],ignore_index=True)
        # remove the rows in df_temp from df_motif
        df_motif=df_motif[df_motif["start_rel"]>current_start+window].copy().reset_index(drop=True)
    return df_res



def plot_motif_imp(df_motif,ax,ylim):
    # only relative position matters
    # Hide spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Plot ISA score
    ax.bar(df_motif["start_rel"], df_motif["isa"], width=df_motif["end_rel"] - df_motif["start_rel"], color="#1f77b4", alpha=0.5, align='edge')
    ax.axhline(0, color='black', lw=0.5)
    # Add text labels for proteins
    prev_text_pos=0
    for idx, row in df_motif.iterrows():
        current_text_pos=row["start_rel"]
        if current_text_pos-prev_text_pos<5:
            current_text_pos=prev_text_pos+5
            if current_text_pos>row["end_rel"]:
                pass
                # raise ValueError("Annotation overlap cannot be resolved")
        ax.text(current_text_pos,row["isa"], row["protein"], rotation=90, fontsize=5)
        prev_text_pos=current_text_pos
    #  set labels
    ax.set_xlabel("Motif ISA",fontsize=7,labelpad=0)
    ax.tick_params(axis='y', which='major', labelsize=7)
    ax.set_xticks([])
    ax.set_ylim(ylim)

def plot_base_imp(df,ax,markersize,xlabel,ylim=None):
    """
    df have columns "position", "base", "imp"
    """
    # Hide spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.bar(df["position"], df["imp"], color="#1f77b4", alpha=0.5)
    ax.axhline(0, color='black', lw=0.5)
    # Create color map for bases
    color_dict = {"A": "#1f77b4", "C": "#ff7f0e", "G": "#2ca02c", "T": "#d62728"}
    # Plot markers for each base
    for row in df.itertuples():
        ax.plot(row.position, row.imp, marker="o", color=color_dict[row.base], markersize=markersize)
    ax.set_xlabel(xlabel,fontsize=7,labelpad=0)
    ax.tick_params(axis='y', which='major', labelsize=7)
    ax.set_xticks([])
    #
    # Create legend
    handles = [mpatches.Patch(color=color, label=base) for base, color in color_dict.items()]
    ax.legend(handles=handles, title="Bases", title_fontsize=5, fontsize=5, loc="upper right")
    if ylim:
        ax.set_ylim(ylim)




def plot_region(seq_extractor, jaspar_annotator, element_name, region, track_num, score_threshold, markersize):
    region_resized = resize_region(region, 599, fix="center")
    left_shift = region[1] - region_resized[1]
    # get seq
    seq = seq_extractor.get_seq(region)
    seq_resized = seq_extractor.get_seq(region_resized)
    # get base importance
    isa = compute_mutagenesis_score(seq_resized, "isa", "mean").loc[f"Seq0_Track{track_num}", :]
    isa = isa[left_shift:(left_shift + region[2] - region[1] + 1)].reset_index(drop=True)
    ism = compute_mutagenesis_score(seq_resized, "ism", "mean").loc[f"Seq0_Track{track_num}", :]
    ism = ism[left_shift:(left_shift + region[2] - region[1] + 1)].reset_index(drop=True)
    # get df_motif
    df_motif = get_motifs(seq_extractor, jaspar_annotator, region_resized, track_num, score_threshold)
    df_motif = reduce_motifs(df_motif)
    # subset to region
    df_motif = df_motif[(df_motif.loc[:, "start"] >= region[1]) & (df_motif.loc[:, "end"] <= region[2])].copy().reset_index(drop=True)
    df_motif["start_rel"] -= left_shift
    df_motif["end_rel"] -= left_shift

    # 保存 get_motifs 的输出到文件
    output_file = f"{element_name}_motifs_track{track_num}.csv"
    df_motif.to_csv(output_file, index=False)
    print(f"Motif data saved to {output_file}")
    
    # Compute gradxinp
    bed_file = pd.DataFrame([[region[0], region[1], region[2]]])  # Create a BED-like DataFrame
    df_gradxinp = compute_gradxinp(bed_file, seq_extractor=seq_extractor)  # 不提供 targets 参数

    # 构造目标行的索引
    target_row = f"{region[0]}:{region[1]}-{region[2]}_Track0"
    if target_row in df_gradxinp.index:
        gradxinp = df_gradxinp.loc[target_row, :].values  # 提取目标行的数据
    else:
        raise KeyError(f"Row '{target_row}' not found in gradxinp DataFrame.")

    # Create subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(180 / 25.4, 110 / 25.4))

    # 添加标题
    fig.suptitle(element_name, fontsize=10, fontweight='bold')
    ymax = max(max(ism), max(isa), max(gradxinp))
    ymin = min(min(ism), min(isa), min(gradxinp))
    # Plot ISA score on the first axis
    plot_motif_imp(df_motif, ax1, (ymin, ymax))
    # plot isa on the second axis
    df_n = pd.DataFrame({"position": list(range(len(isa))), "base": list(seq), "imp": isa})
    plot_base_imp(df_n, ax2, markersize, "ISA (Base replaced by N)", ylim=(ymin, ymax))
    # plot ism on the third axis
    df_a = pd.DataFrame({"position": list(range(len(ism))), "base": list(seq), "imp": ism})
    plot_base_imp(df_a, ax3, markersize, "ISM (Average of 3 alternative bases)", ylim=(ymin, ymax))
    # plot gradxinp on the fourth axis
    df_g = pd.DataFrame({"position": list(range(len(gradxinp))), "base": list(seq), "imp": gradxinp})
    plot_base_imp(df_g, ax4, markersize, "Gradxinp", ylim=(ymin, ymax))
    
    # Add supra ticks and labels
    ax4.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax4.tick_params(axis='both', which='major', labelsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局以避免标题与子图重叠
    plt.savefig(f"{element_name}_track{track_num}.pdf", dpi=300)
    plt.close()

# main
plot_region(
    seq_extractor=seq_extractor,
    jaspar_annotator=jaspar_hepg2_annotator,
    element_name="chr2:161995555-161996154",
    region=("chr2",161995555,161996154),
    track_num=0,
    score_threshold=360,
    markersize=1
)