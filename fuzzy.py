import os
import openai
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def get_embedding(text):
    result = openai.embeddings.create(
      model='text-embedding-ada-002',
      input=text
    )
    return result.data[0].embedding

def fuzzy_match(target_df, source_df, columns, threshold=0.85):
    """
    Function to perform fuzzy matching between two dataframes on specified columns.

    Parameters:
    target_df (pd.DataFrame): The dataframe to be matched to.
    source_df (pd.DataFrame): The dataframe to be matched from.
    columns (list of str): The columns to perform fuzzy matching on.
    threshold (float, optional): The cosine similarity threshold for a match to be considered 'good'. Defaults to 0.85.

    Returns:
    pd.DataFrame: A new dataframe where each specified column in source_df is matched against the corresponding column in target_df,
                  with similarity scores and 'good'/'bad' match indicators for each column.
    """

    matched_results = source_df[columns].copy()

    for column in columns:
        target_df[column + '_embeddings'] = target_df[column].apply(get_embedding)
        source_df[column + '_embeddings'] = source_df[column].apply(get_embedding)

        nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(target_df[column + '_embeddings'].to_list())
        distances, indices = nn.kneighbors(source_df[column + '_embeddings'].to_list(), return_distance=True)

        matched_results[column + '_matched_to'] = [target_df.loc[indices[i, 0], column] for i in range(source_df.shape[0])]
        matched_results[column + '_similarity'] = 1 - distances
        matched_results[column + '_is_good_match'] = ['good' if 1 - distances[i, 0] >= threshold else 'bad' for i in range(source_df.shape[0])]

    return matched_results


# Assuming you have two dataframes df1 and df2 with 'name' and 'address' fields
df1 = pd.DataFrame({
    'name': ['John Doe', 'Mary Jane', 'Peter Parker'],
    'address': ['123 Main St', '456 Oak St', '789 Pine St'],
    'age': ['10','25','100'],
})

df2 = pd.DataFrame({
    'name': ['Pete Parker', 'Jon Doe', 'John Doe', 'Marry Jane', 'Anita Smith'],
    'address': ['123 Main Street', '457 Oak St', '457', '789 Pine Street', '789'],
    'age': ['10','30','40','90',''],
})

#get matches for age
print(fuzzy_match(df1,df2,['age'],0.85))

#get matches for name
print(fuzzy_match(df1,df2,['name'],0.85))

#get matches for address
print(fuzzy_match(df1,df2,['address'],0.85))
