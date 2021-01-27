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

# Import dataset
# ../TestData/datasets/
midwest = pd.read_csv("../TestData/datasets/midwest_filter.csv")
# 将数据进行传入


# Prepare Data
# Create as many colors as there are unique midwest['category']
categories = np.unique(midwest['category'])
# 将所有的类别进行归类
# 将所有的类别都存放在category

colors = [plt.cm.tab10(i / float(len(categories) - 1)) for i in range(len(categories))]
# 特殊语法


# Draw Plot for Each Category
plt.figure(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')

"""
下面需要修改 
其中的area 表示第一个维度 poptotal表示的是第二个维度 即纵轴
其中选择的是数据中的列的名称 将这些数据进行输入
"""
for i, category in enumerate(categories):
    plt.scatter('area', 'poptotal',
                data=midwest.loc[midwest.category == category, :],
                s=20, cmap=colors[i], label=str(category))
    # data是一个pandas的dataframe类型的 所以前两参数在指定列 data就是要被找的目标
    # s是点的大小 cmap指的是这个颜色 label知名这个颜色所表示的名字

# Decorations
# plt.gca().set(xlim=(0.0, 0.1), ylim=(0, 90000),
#               xlabel='Area', ylabel='Population')
# 可以不进行指定，这样通用性更强
"""
需要进行修改
"""
plt.gca().set(xlabel='Area', ylabel='Population')
"""
可以进行修改
调整坐标字体大小
"""
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
"""
需要进行修改标题
"""
plt.title("Scatterplot of Midwest Area vs Population", fontsize=22)
plt.legend(fontsize=12)
plt.show()
