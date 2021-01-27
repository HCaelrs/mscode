# encoding:utf-8
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings;

warnings.filterwarnings(action='once')

large = 22;
med = 16;
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

# Version
print(mpl.__version__)  # > 3.0.0
print(sns.__version__)  # > 0.9.0

# ../TestData/datasets/

# Import Data
df = pd.read_csv("../TestData/datasets/mpg_ggplot2.csv")
df_counts = df.groupby(['hwy', 'cty']).size().reset_index(name='counts')

# Draw Stripplot

# sns.stripplot(x=df_counts.cty, y=df_counts.hwy, s=df_counts.counts, ax=ax)
plt.scatter(x=df_counts.cty, y=df_counts.hwy, s=df_counts.counts*20, c=(df_counts.cty+df_counts.hwy)/2)

# xx = np.random.random(100)
# yy = np.random.random(100)
# sz = np.random.random(100) * 1000
# plt.scatter(xx, yy, s=sz)

# Decorations
plt.title('Counts Plot - Size of circle is bigger as more points overlap', fontsize=22)
plt.show()
