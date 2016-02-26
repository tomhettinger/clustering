# -*- coding: utf-8 -*-
"""
@author: Tom
"""
import os

import seaborn as sns
sns.set(style="whitegrid", font_scale=1.6)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FIG_PATH = u'C:\\Users\\Tom\\Desktop\\clustering\\figures\\'

def swarmplot(df):
    fig = plt.figure(figsize=[12,8])
    ax = fig.gca()
    df_melted = pd.melt(frame = df, 
                        id_vars = ["group"],
                        value_vars = ["height", "heartrate", "weight", "age"],
                        var_name="measurement")
    sns.swarmplot(x="measurement", y="value", hue="group", data=df_melted)
    ax.set_xticklabels(['height (cm)', 'heart rate (bpm)', 'weight (kg)', 'age (years)'])
    ax.set_xlabel('')
    fig.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, 'swarmplot.png'), dpi=100)
    

def heatmap(df):
    corr = df.drop(['group', 'id'], axis=1).corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    fig = plt.figure(figsize=[8,8])
    #cmap = sns.cubehelix_palette(8, light=0.8, dark=0.2, as_cmap=True)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, vmax=.8, square=True, cmap=cmap, linewidths=0.8, annot=True)
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(FIG_PATH, 'heatmap.png'))
    
    
def pairplot(df, group="group"):
    sns.pairplot(data=df.drop('id', axis=1),
                 vars=['age', 'weight', 'heartrate', 'height'],
                 hue=group,
                 diag_kind='kde', 
                 size=5,
                 diag_kws=dict(shade=True, linewidth=2),
                 plot_kws=dict(s=50) )
    if group == "group":
        plt.savefig(os.path.join(FIG_PATH, 'pairplot.png'), dpi=100)
    else:
        plt.savefig(os.path.join(FIG_PATH, 'pairplot_%s.png' % group), dpi=100)


def pairplot_kde(df):
    g = sns.PairGrid(df.drop(['id'], axis=1), diag_sharey=False, size=5)
    g.map_lower(sns.kdeplot, cmap="Blues_d", shade=True, shade_lowest=False)
    g.map_upper(plt.scatter)
    g.map_diag(sns.kdeplot, lw=3, shade=True)
    plt.savefig(os.path.join(FIG_PATH, 'pairplot_kde.png'), dpi=100)
    
    
def violin_subplot(ax, df, p, ylab):
    sns.violinplot(x='group', y=p, hue='gender', axis=1, data=df, 
                   split=True, inner="quart", ax=ax)
    plt.xticks(rotation=10)
    plt.legend(loc=2)
    plt.xlabel('')
    plt.ylabel(ylab)


def violin(df):
    fig = plt.figure(figsize=[18,14])
    ax = plt.subplot(221)
    violin_subplot(ax, df, 'weight', 'weight (kg)')
    ax = plt.subplot(222)
    violin_subplot(ax, df, 'height', 'height (cm)')
    ax = plt.subplot(223)
    violin_subplot(ax, df, 'age', 'age (years)')
    ax = plt.subplot(224)
    violin_subplot(ax, df, 'heartrate', 'heart rate (bpm)')
    fig.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, 'violin.png'), dpi=100)
    
    
def inertia(k_value, inertia):
    plt.figure(figsize=[8,8])
    plt.plot(k_value, inertia, color='dodgerblue', marker='o')
    plt.xlim(0, 12)
    plt.ylim(0, 800)
    plt.xlabel('k')
    plt.ylabel('inertia')
    plt.savefig(os.path.join(FIG_PATH, 'inertia.png'), dpi=100)
    
    
def d_inertia(k_value, inertia):
    x = k_value[1:]
    y = np.diff(inertia)
    plt.figure(figsize=[8,8])
    plt.plot(x, y, color='dodgerblue', marker='o')
    plt.xlim(0, 12)
    plt.ylim(-200, 0)
    plt.xlabel('k')
    plt.ylabel(r'$\Delta$ inertia / $\Delta$ k')
    plt.savefig(os.path.join(FIG_PATH, 'inertia_derivative.png'), dpi=100)
    
    
def compare_model(df, x, y, model='k5_class'):
    g = sns.FacetGrid(data=df, hue='group', size=8)
    g = g.map(plt.scatter, x, y, s=100, marker='*')
    plt.legend(loc=4, markerscale=2, shadow=True, frameon=True, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, '%s_v_%s_true.png' % (y, x)), dpi=100)

    g = sns.FacetGrid(data=df, hue=model, size=8, hue_order=[3, 1, 2, 0, 4])
    g = g.map(plt.scatter, x, y, s=100)
    plt.legend(loc=4, markerscale=1.5, shadow=True, frameon=True, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, '%s_v_%s_pred.png' % (y, x)), dpi=100)
