"""Publication-style snapshot of locked microscopic traces and inverse confirmations."""
from pathlib import Path
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt, font_manager

root=Path(__file__).resolve().parent
def load(name):
    return json.loads((root/name).read_text(encoding="utf-8-sig"))

font = Path(os.environ["WINDIR"]) / "Fonts" / "msyh.ttc" if "WINDIR" in os.environ else None
if font is not None and font.is_file():
    font_manager.fontManager.addfont(font)
    plt.rcParams["font.family"]=font_manager.FontProperties(fname=font).get_name()
plt.rcParams.update({"axes.unicode_minus":False, "font.size":10, "axes.spines.top":False,
                     "axes.spines.right":False, "text.color":"#172e41",
                     "axes.labelcolor":"#172e41"})
times=np.asarray(load("REGISTERED.json")["global_times"])
with np.load(root/"CONFIRMED.npz") as data:
    paths=data["path"]
chosen=load("POOLED_NOMINAL.json")["actions"][0]
fig, panels=plt.subplots(1,3,figsize=(14.6,4.25),layout="constrained")
for ax,prop,name in ((panels[0],0,"双键转化比例"),(panels[1],3,"最大反应团簇比例")):
    for arm,label,color in ((0,"不编辑","#8d9ba5"),(chosen,"提高引发与链增长","#16666b")):
        curves=paths[:,arm,:,:,prop].reshape(-1,len(times))
        lo,middle,hi=np.quantile(curves,[.25,.5,.75],axis=0)
        ax.plot(times,middle,color=color,label=label,lw=2)
        ax.fill_between(times,lo,hi,color=color,alpha=.17,linewidth=0)
    ax.set(xlabel="模型时间",ylabel=name,ylim=(0,1))
    ax.set_title(name,loc="left",fontweight="semibold")
panels[0].legend(frameon=False,loc="upper left",fontsize=9)
schedule=load("scheduled_edit/SCORED.json")["results"]
k="early_network_with_chain"
stats=schedule[k]["hits"]
samples=[stats[key] for key in ("zero","both_plus_plus","early_feed__both_plus_plus")]
heights=np.array([s["estimate"] for s in samples])
limits=np.array([s["ci95"] for s in samples]).T
bars=panels[2].bar(np.arange(3),heights, color=["#8d9ba5","#16666b","#2b8294"],width=.57)
panels[2].errorbar(np.arange(3),heights,yerr=(heights-limits[0],limits[1]-heights),
                   fmt="none",color="#142f40",capsize=4,lw=1.2)
panels[2].set(ylim=(0,.66),ylabel="达标比例",xticks=np.arange(3),
              xticklabels=("不编辑","固定规则编辑","同规则＋提前投料"))
panels[2].set_title("新初态的快速目标回验",loc="left",fontweight="semibold")
for bar,s in zip(bars,samples):
    panels[2].text(bar.get_x()+bar.get_width()/2,s["ci95"][1]+.013,
                   f'{s["n_success"]}/{s["n_paths"]}',ha="center",va="bottom",fontsize=9)
fig.suptitle("从固定材料规则到随时间形成的网络",x=.02,ha="left",fontweight="bold",fontsize=15)
fig.savefig(root/"CLOSED_LOOP_FIGURE.png",dpi=175,bbox_inches="tight")
