"""Plot the completed capacity comparisons and independently repaired gate tests."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('thickness',type=Path)
    parser.add_argument('nine_actions',type=Path)
    parser.add_argument('calibrated_thickness',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--replicated',type=Path)
    parser.add_argument('--learned',type=Path)
    a=parser.parse_args()
    capacity=read(a.thickness/'SCORED.json')['summary']['shift_test']['fixed_area_conductance']['means']
    strict=read(a.nine_actions/'CERTIFICATE.json')
    test=read(a.nine_actions/'SCORED.json')['reports']['confirmation']
    loose=read(a.nine_actions/'LOOSE_CONFIRMED.json')
    strict_model=strict['policy']['primary_method']
    head=read(a.calibrated_thickness/'SCORED.json')['results']
    limits=read(a.calibrated_thickness/'CERTIFICATE.json')['catalogues']
    make_four=a.replicated is not None and a.learned is not None
    fig,axs=plt.subplots(2,2,figsize=(11.8,8.3),layout='constrained') if make_four else plt.subplots(1,3,figsize=(13.2,4.65),layout='constrained')
    axs=np.asarray(axs).ravel()
    color=( '#2f6586','#2f957d','#cc7751')
    cat=('grid25','thickness27')
    side=np.arange(2)
    for i,(label,key) in enumerate((('Any feasible design','feasible'),('Chosen design passes','success'))):
        vals=[100*capacity[name][key] for name in cat]
        bar=axs[0].bar(side+.34*(i-.5),vals,width=.31,color=color[i],label=label)
        axs[0].bar_label(bar,fmt='%.1f',padding=3,fontsize=9)
    axs[0].set_xticks(side,['25 fixed thickness','27 variable thickness'])
    axs[0].set(ylim=(0,88),ylabel='Percentage of requests',title='a   Design feasibility in a changed mechanism')
    axs[0].legend(frameon=False,fontsize=9)
    points=[test[strict_model]['locked'],loose['confirmation']]
    for i,item in enumerate(points):
        scale=item['requests']/100
        delivered=item['successful_requests']/scale
        bad=item['unsafe_accepted_requests']/scale
        axs[1].bar(i,delivered,color=color[1],width=.64,label='Meets window' if i==0 else None)
        axs[1].bar(i,bad,bottom=delivered,color=color[2],width=.64,label='Violates window' if i==0 else None)
        axs[1].text(i,delivered+bad+1,f"{item['accepted_requests']}/{item['requests']}",ha='center',fontsize=9)
    axs[1].set_xticks(side,['10% risk target','25% risk target'])
    axs[1].set(ylim=(0,57),ylabel='Issued requests (%)',title='b   Two verified 9-action operating points')
    axs[1].legend(frameon=False,fontsize=9)
    if make_four:
        replication=read(a.replicated/'SCORED.json')['results']
        delta=[head['safe_issued_delta_fraction'],replication['paired_delta']]
        interval=[head['paired_ci'],replication['paired_ci_95']]
        for i,(value,(lo,hi)) in enumerate(zip(delta,interval,strict=True)):
            axs[2].errorbar(100*value,i,xerr=[[100*(value-lo)],[100*(hi-value)]],fmt='o',capsize=4,color=color[0])
        axs[2].set_yticks(side,['128 batches','640 new batches'])
        axs[2].axvline(0,ls='--',lw=1,color='#9aa4ac')
        axs[2].set_ylim(1.5,-.5)
        axs[2].set(xlabel='Correctly issued request difference (percentage points)',title='c   Variable minus fixed thickness')
        added=read(a.learned/'EVALUATED.json')['results']['thickness27']
        for i,key in enumerate(('upper_margin_control','learned')):
            counts=added[key]
            ratio=100*counts['successful_requests']/counts['requests']
            ok=(key=='upper_margin_control' or added['certificate']['certified'])
            rect=axs[3].bar(i,ratio,width=.6,color=color[1] if ok else '#a5aab0')
            axs[3].bar_label(rect,labels=[f"{counts['successful_requests']}/{counts['requests']}"],padding=4,fontsize=9)
        axs[3].set_xticks(side,['Frozen upper margin','Learned geometry score'])
        cap=max(100*added[key]['successful_requests']/added[key]['requests'] for key in ('upper_margin_control','learned'))
        axs[3].set(ylabel='Issued & meets window (%)',ylim=(0,max(15,1.27*cap)),title='d   Alternative score on a separate test partition')
        if not added['certificate']['certified']:
            axs[3].text(.5,.91,'Grey gate not certified',transform=axs[3].transAxes,ha='center',fontsize=9)
    else:
        for i,key in enumerate(cat):
            arm=head['catalogues'][key]
            primary=arm['primary']
            rule=limits[key]['calibration'][primary]['fixed_policy']
            v=arm['policies'][primary]['locked']
            result=100*v['successful_requests']/v['requests']
            bar=axs[2].bar(i,result,color=color[1] if rule['certified'] else '#a5aab0',width=.60)
            axs[2].bar_label(bar,labels=[f"{v['successful_requests']}/{v['requests']}"],padding=4,fontsize=9)
        axs[2].set_xticks(side,['25 fixed thickness','27 variable thickness'])
        hi=max(100*x['policies'][x['primary']]['locked']['successful_requests']/x['policies'][x['primary']]['locked']['requests'] for x in head['catalogues'].values())
        axs[2].set(ylim=(0,max(20,1.28*hi)),ylabel='Issued & meets window (%)',title='c   10% risk target on a new cohort')
        if not head['both_certified']:
            axs[2].text(.5,.89,'Grey gate not certified',transform=axs[2].transAxes,ha='center',fontsize=9)
    for ax in axs:
        ax.grid(axis='y',color='#e3e9ed')
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Changes in feasible design capacity and risk-constrained process decisions',weight='bold',fontsize=14)
    fig.savefig(a.output,dpi=180,facecolor='white')


if __name__=='__main__':
    main()
