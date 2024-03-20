import pandas as pd
import numpy as np
from ast import literal_eval

datafile_path = "3 Embeddings/data/fine_food_reviews_with_embeddings_1k.csv"

df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

from utils.embeddings_utils import get_embedding, cosine_similarity

# search through the reviews for a specific product
def search_reviews(df, product_description, n=3, pprint=True):      # Pandas DataFrame 产品描述，数量，以及一个 pprint 标志（默认值为 True）。
    product_embedding = get_embedding(
        product_description,
        model="text-embedding-3-small"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results

results = search_reviews(df, "delicious beans", n=3)# 使用 'delicious beans' 作为产品描述和 3 作为数量，# 调用 search_reviews 函数来查找与给定产品描述最相似的前3条评论。# 其结果被存储在 res 变量中。