"""Visualize the factorial decisions, a half-null pair, and predictive coverage."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('score', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    cohorts = json.loads(args.score.read_text(encoding='utf-8'))['cohorts']
    rows = list(cohorts.values())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.right':False,'axes.spines.top':False})
    fig, axes = plt.subplots(1,3,figsize=(12.6,3.9), layout='constrained')
    colors = ['#59646e','#63959a','#b98a58','#1c667b']
    keys = ['timed_total','timed_half','dual_total','dual_half']
    labels = ['Timed + total','Timed + half','Dual + total','Dual + half']
    for j,(key,label,color) in enumerate(zip(keys,labels,colors,strict=True)):
        success = [sum(r['arms'][key]['count'] for r in data)/(32*len(data)) for data in rows]
        bars = axes[0].bar(np.arange(2)+(j-1.5)*.20,success,.19,label=label,color=color)
        axes[0].bar_label(bars,labels=[f'{100*x:.1f}%' for x in success],fontsize=7,padding=3)
    axes[0].set(title='A   Matched-budget factorial',ylabel='Joint success fraction',ylim=(0,.93),xticks=[0,1],
                xticklabels=['Joint uncertainty','Interface + prep shift'])
    axes[0].legend(loc='upper left',frameon=False,fontsize=8,ncol=2)

    first = rows[0][0]['truth']['dry']
    for j,color in enumerate(('#2a7581','#a6572b')):
        label = 'Twin A' if j == 0 else 'Twin B'
        axes[1].stairs(first[j],np.linspace(0,1,9),label=label,color=color,linewidth=2)
    axes[1].axvline(.5,linestyle='--',color='#777777',linewidth=1)
    axes[1].set(title='B   Same half-layer inventories',xlabel='Depth / thickness',ylabel='Initial inventory fraction per bin')
    axes[1].legend(frameon=False,fontsize=8)

    for j,(tag,label,color) in enumerate((('dual_total','Total stock','#59646e'),('dual_half','Half stock','#1c667b'),
                                        ('dual_back_slice','Deep slice','#a6572b'))):
        coverage = [np.mean([r['arms'][tag]['profile_coverage'] for r in data]) for data in rows]
        bars = axes[2].bar(np.arange(2)+(j-1)*.23,coverage,.22,label=label,color=color)
        axes[2].bar_label(bars,labels=[f'{100*x:.1f}%' for x in coverage],fontsize=8,padding=3)
    axes[2].axhline(.9,color='#555555',linestyle='--',linewidth=1)
    axes[2].text(1.38,.918,'90% nominal',fontsize=8,ha='right')
    axes[2].set(title='C   Dry-profile interval coverage',ylabel='Empirical marginal coverage',ylim=(0,1.2),xticks=[0,1],
                xticklabels=['Joint uncertainty','Interface + prep shift'])
    axes[2].legend(loc='upper left',frameon=False,fontsize=8,ncol=2)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg'):
        target=args.out.with_suffix(f'.{ext}')
        if target.exists():
            raise FileExistsError(target)
        fig.savefig(target,dpi=200)


if __name__ == '__main__':
    main()
