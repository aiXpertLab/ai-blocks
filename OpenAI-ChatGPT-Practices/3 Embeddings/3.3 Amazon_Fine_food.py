import matplotlib               # matplotlib 是一个 Python 的 2D 绘图库，pyplot 是其子库，提供了一种类似 MATLAB 的绘图框架。
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE # TSNE (t-Distributed Stochastic Neighbor Embedding) 是一种用于数据可视化的降维方法，尤其擅长处理高维数据的可视化。# 它可以将高维度的数据映射到 2D 或 3D 的空间中，以便我们可以直观地观察和理解数据的结构。
from sklearn.cluster  import KMeans 
import pandas as pd
import numpy  as np ## 导入 NumPy 包，NumPy 是 Python 的一个开源数值计算扩展。这种工具可用来存储和处理大型矩阵，比 Python 自身的嵌套列表（nested list structure)结构要高效的多。
import tiktoken, ast             # 导入 tiktoken 库。Tiktoken 是 OpenAI 开发的一个库，用于从模型生成的文本中计算 token 数量。
from utils.embeddings_utils import get_embedding, cosine_similarity

from openai import OpenAI
client = OpenAI()

input_datapath = "3 Embeddings/data/fine_food_reviews_1k.csv"
df = pd.read_csv(input_datapath, index_col=0)
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
df = df.dropna()
 
# 将 "Summary" 和 "Text" 字段组合成新的字段 "combined"
df["combined"] = (
    "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip())
print(df.head(2))

embedding_model = "text-embedding-3-small"
embedding_encoding = "cl100k_base"      # text-embedding-ada-002 模型对应的分词器（TOKENIZER）
max_tokens = 8000                       # text-embedding-ada-002 模型支持的输入最大 Token 数是8191，向量维度 1536# 在我们的 DEMO 中过滤 Token 超过 8000 的文本

top_n = 1000                                # 设置要筛选的评论数量为1000
df = df.sort_values("Time").tail(top_n * 2) # 对DataFrame进行排序，基于"Time"列，然后选取最后的2000条评论。# 这个假设是，我们认为最近的评论可能更相关，因此我们将对它们进行初始筛选。
df.drop("Time", axis=1, inplace=True)       # 丢弃"Time"列，因为我们在这个分析中不再需要它。
encoding = tiktoken.get_encoding(embedding_encoding)    # 从'embedding_encoding'获取编码
df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))   # 计算每条评论的token数量。我们通过使用encoding.encode方法获取每条评论的token数，然后把结果存储在新的'n_tokens'列中。
df = df[df.n_tokens <= max_tokens].tail(top_n)         # 如果评论的token数量超过最大允许的token数量，我们将忽略（删除）该评论。# 我们使用.tail方法获取token数量在允许范围内的最后top_n（1000）条评论。

print(len(df))# 打印出剩余评论的数量。

# 实际生成会耗时几分钟
# 提醒：非必须步骤，可直接复用项目中的嵌入文件 fine_food_reviews_with_embeddings_1k
# df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model))
# output_datapath = "data/fine_food_reviews_with_embeddings_1k_0904.csv"
# df.to_csv(output_datapath)

embedding_datapath = "3 Embeddings/data/fine_food_reviews_with_embeddings_1k.csv"
df_embedded = pd.read_csv(embedding_datapath, index_col=0)
df_embedded["embedding_vec"] = df_embedded["embedding"].apply(ast.literal_eval) # 将字符串转换为向量
print(len(df_embedded["embedding_vec"][0]))
print(df_embedded.head(2))

# 使用 t-SNE 可视化 1536 维 Embedding 美食评论
print(type(df_embedded["embedding_vec"]))

assert df_embedded['embedding_vec'].apply(len).nunique() == 1   #首先，确保你的嵌入向量都是等长的
matrix = np.vstack(df_embedded['embedding_vec'].values)         #将嵌入向量列表转换为二维 numpy 数组
# 创建一个 t-SNE 模型，t-SNE 是一种非线性降维方法，常用于高维数据的可视化。
# n_components 表示降维后的维度（在这里是2D）# perplexity 可以被理解为近邻的数量
# random_state 是随机数生成器的种子# init 设置初始化方式# learning_rate 是学习率。
tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200) 
vis_dims = tsne.fit_transform(matrix) #使用 t-SNE 对数据进行降维，得到每个数据点在新的2D空间中的坐标
colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]  #  定义了五种不同的颜色，用于在可视化中表示不同的等级

x = [x for x,y in vis_dims]#从降维后的坐标中分别获取所有数据点的横坐标和纵坐标
y = [y for x,y in vis_dims]

color_indices = df_embedded.Score.values - 1  # 根据数据点的评分（减1是因为评分是从1开始的，而颜色索引是从0开始的）获取对应的颜色索引
assert len(vis_dims) == len(df_embedded.Score.values)   # 确保你的数据点和颜色索引的数量匹配

colormap = matplotlib.colors.ListedColormap(colors) #创建一个基于预定义颜色的颜色映射对象
plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)    #使用 matplotlib 创建散点图，其中颜色由颜色映射对象和颜色索引共同决定，alpha 是点的透明度
plt.title("Amazon ratings visualized in language using t-SNE")
# plt.show()

# 使用 K-Means 聚类，然后使用 t-SNE 可视化
# np.vstack 是一个将输入数据堆叠到一个数组的函数（在垂直方向）。# 这里它用于将所有的 ada_embedding 值堆叠成一个矩阵。
# matrix = np.vstack(df.ada_embedding.values)

n_clusters = 4          # 定义要生成的聚类数。
# 创建一个 KMeans 对象，用于进行 K-Means 聚类。# n_clusters 参数指定了要创建的聚类的数量；
# init 参数指定了初始化方法（在这种情况下是 'k-means++'）；
# random_state 参数为随机数生成器设定了种子值，用于生成初始聚类中心。
# n_init=10 消除警告 'FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4'
kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=42, n_init=10)

kmeans.fit(matrix)  # 使用 matrix（我们之前创建的矩阵）来训练 KMeans 模型。这将执行 K-Means 聚类算法。

df_embedded['Cluster'] = kmeans.labels_ # kmeans.labels_ 属性包含每个输入数据点所属的聚类的索引。# 这里，我们创建一个新的 'Cluster' 列，在这个列中，每个数据点都被赋予其所属的聚类的标签。
print(df_embedded["Cluster"])

# 首先为每个聚类定义一个颜色。
colors = ["red", "green", "blue", "purple"]
 
# 然后，你可以使用 t-SNE 来降维数据。这里，我们只考虑 'embedding_vec' 列。
tsne_model = TSNE(n_components=2, random_state=42)
vis_data = tsne_model.fit_transform(matrix)
 
# 现在，你可以从降维后的数据中获取 x 和 y 坐标。
x = vis_data[:, 0]
y = vis_data[:, 1]
 
color_indices = df_embedded['Cluster'].values           # 'Cluster' 列中的值将被用作颜色索引。
colormap = matplotlib.colors.ListedColormap(colors)     # 创建一个基于预定义颜色的颜色映射对象

plt.scatter(x, y, c=color_indices, cmap=colormap)   # 使用 matplotlib 创建散点图，其中颜色由颜色映射对象和颜色索引共同决定
plt.title("Clustering visualized in 2D using t-SNE")
 
plt.show()

