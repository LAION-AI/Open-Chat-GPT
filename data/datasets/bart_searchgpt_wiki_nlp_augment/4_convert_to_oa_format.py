import json

import pandas as pd

if __name__ == "__main__":
    # Use the `dtype` parameter of `pd.read_csv`.
    raw_df = pd.read_csv(r"...\wiki_qa_bart_10000row.csv")
    # print(raw_df.iloc[0])
    # print(raw_df.columns)
    instruction_list, response_list, metadata_list = [], [], []
    for index, row in raw_df.iterrows():
        instruction_list.append(row["query"])
        response_list.append(row["response"])
        metadata_json = {
            "wiki_id": row["id"],
            "text": row["text"],
            "title": row["title"],
            "url": row["url"],
            "question_parent": row["parent"],
        }
        metadata_list.append(json.dumps(metadata_json))
    final_df = pd.DataFrame(
        {"INSTRUCTION": instruction_list, "RESPONSE": response_list, "SOURCE": "wikipedia", "METADATA": metadata_list}
    )
    final_df.to_parquet("oa_wiki_qa_bart_10000row.parquet", row_group_size=100)
