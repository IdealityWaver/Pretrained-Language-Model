import os
import sys
import re
import seaborn as sns
import numpy as np

sns.set_theme()
pattern = ".*: (0.*)\}: \(([0-9]*),([0-9]*)\)"
pattern = re.compile(pattern)
data = np.empty((12, 12))

with open('./eval_results_ablation.txt', 'r') as f:
    for line in f:
        line = line.strip()
        res = pattern.match(line)
        acc, i, l = float(res[1]), int(res[2]), int(res[3])
        data[l,i] = (acc)


#ax = sns.heatmap(data, cmap="YlGnBu")
ax = sns.heatmap(data)
fig = ax.get_figure()
fig.savefig('/mnt/e/importance_map.png')
