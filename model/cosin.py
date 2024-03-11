import openai
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import ast
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type
import tiktoken
from itertools import islice

cve_sid = pd.read_csv("../output/finaldata/emb/cve_sid_em_test.csv")
code_diff = pd.read_csv("../output/finaldata/emb/code_diffs_clean.csv")
message = pd.read_csv("../output/finaldata/emb/message_f.csv")
accracy_case = open("accuracy.txt","a+")
final_result = open("final_result.txt","a+")


def convert_to_list(string_representation):
    try:
        return ast.literal_eval(string_representation)
    except (SyntaxError, ValueError):
        return []

#Compute cosine similarity
def cosin(df):
    print("start")
    # keep track accurate cases
    accurate_cases = 0
    save_l = []


    # Loop through each row in cvesside and compute cosine similarity with each row in message
    for index, row in cve_sid.iterrows():
        cve_p = row['CVE_ID']
        cve_embedding = np.array(ast.literal_eval(row['embedding']))

        result_list = []

        for index_d, row_d in df.iterrows():
            message_p = row_d['CVE_ID']
            message_embedding = np.array(ast.literal_eval(row_d['embedding']))
            #message_embedding = np.array(row_d['embedding'])
            message_label = row_d['label']
            #message_label = row_d['label_y']

            cosine_sim = cosine_similarity([cve_embedding], [message_embedding])[0][0]
            print("finish")

            result_dict = {
                'CVE_ID': cve_p,
                'message_ID': message_p,
                'message_label': message_label,
                'cosine_sim': cosine_sim
            }

            result_list.append(result_dict)

            # Sort based on cosine similarity and get top 5
        result_list = sorted(result_list, key=lambda x: x['cosine_sim'], reverse=True)[:1]

        save_l.append(result_list)


        # Check accuracy
        for result_dict in result_list:
            if result_dict['CVE_ID'] == result_dict['message_ID'] and result_dict['message_label'] == 1:
                accurate_cases += 1
                print("accurate_cases: ", accurate_cases)
                # write accurate_cases to file
                accracy_case.write('CVE_ID: ' + str(cve_p) + ' cosine_sim: ' + str(cosine_sim) + '\n')
                break

    for entry in save_l:
        final_result.write(str(entry) + '\n')
    accracy_case.write('FINISH'+'\n')
    return accurate_cases / len(cve_sid)

def average_pooling(embeddings):
    # Determine the max length of vectors in embeddings
    max_length = max(len(vector) for vector in embeddings)

    # Pad shorter vectors with NaN (which will be ignored in np.nanmean)
    padded_embeddings = [vector + [np.nan] * (max_length - len(vector)) for vector in embeddings]

    # Convert to NumPy array for efficient computation
    np_embeddings = np.array(padded_embeddings)

    # Compute the mean, ignoring NaN values
    mean_embedding = np.nanmean(np_embeddings, axis=0)

    return mean_embedding



'''code_diff['embedding'] = code_diff['embedding'].apply(convert_to_list)

code_diff["pooled"] = code_diff['embedding'].apply(average_pooling)
print("finish")

merged_df = pd.merge(code_diff, message, on='CVE_ID')
#merged_df['pooled'] =merged_df['pooled'].apply(convert_to_list)
merged_df['embedding_y'] =merged_df['embedding_y'].apply(convert_to_list)


# Function to compute the average of two embeddings
def average_embeddings(row):
    pooled = np.array(row['pooled'])
    embedding = np.array(row['embedding_y'])
    return (pooled + embedding) / 2

# Apply the function to each row
merged_df['embedding'] = merged_df.apply(average_embeddings, axis=1)
print(merged_df['embedding'][0])
print("merged")'''
#accuracy_rate = cosin(merged_df)
accuracy_rate = cosin(message)
print(accuracy_rate)


