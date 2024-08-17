import pandas as pd
import tiktoken
import torch
import tqdm
from transformers import pipeline


def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens


if __name__ == "__main__":
    # Use the `dtype` parameter of `pd.read_csv`.
    sampled_df = pd.read_csv("wiki_qa_bart_10000row_input.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    response_list = []
    # sampled_df = sampled_df.iloc[:5]
    for i, row in tqdm.tqdm(sampled_df.iterrows(), total=sampled_df.shape[0]):
        try:
            num_tokens = num_tokens_from_string(row["text"])
            min_length = 40
            max_length = max(int(num_tokens * 0.7), min_length) + 1
            response = summarizer(row["text"], max_length=max_length, min_length=min_length, do_sample=False)[0][
                "summary_text"
            ]
        except Exception as e:
            print(e)
            response = ""
        print(f"text (token: {num_tokens}, max_length: {max_length}, min_length: {min_length})")
        print(row["text"])
        print("output")
        print(response)
        response_list.append(response)
    sampled_df["response"] = response_list
    sampled_df.to_csv("wiki_qa_bart_10000row.csv", index=False)
