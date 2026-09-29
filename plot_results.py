
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_rq1(summary_csv,outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(summary_csv)
    if {"budget","method","search_auc"}.issubset(df.columns):
        g=df.groupby(["budget","method"])["search_auc"].mean().unstack()
        ax=g.plot(marker="o")
        ax.set_xlabel("Architecture evaluation budget")
        ax.set_ylabel("Mean search AUC")
        ax.figure.tight_layout()
        ax.figure.savefig(out/"rq1_search_auc.pdf",dpi=600)
        plt.close(ax.figure)

def plot_pareto(cost_csv,outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(cost_csv)
    fig,ax=plt.subplots()
    for beta,g in df.groupby("beta"):
        ax.scatter(g["flops"],g["test_acc"],label=f"beta={beta}")
    ax.set_xlabel("FLOPs")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"rq3_accuracy_flops.pdf",dpi=600)
    plt.close(fig)
