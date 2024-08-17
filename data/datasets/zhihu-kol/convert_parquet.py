import json

import pandas as pd


def reformat_csv_to_openassistant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reformat the downloaded CSV into either Instruction or Text format
    so that it could be directly ingested into the training pipeline.

    Parameters
    ----------
    df: the downloaded panda dataframe

    Return
    ------
    DataFrame: reformatted dataframe
    """

    new_df = pd.DataFrame()
    new_df["INSTRUCTION"] = df["question_title"]
    new_df["RESPONSE"] = df["content"]
    new_df["SOURCE"] = "Zhihu"
    new_df["METADATA"] = df.apply(
        lambda x: json.dumps(
            {
                "question_id": x["question_id"],
                "answer_id": x["answer_id"],
                "author_id": x["author_id"],
                "upvotes": x["upvotes"],
                "answer_creation_time": x["answer_creation_time"],
            },
            ensure_ascii=False,
        ),
        axis=1,
    )
    # Remove empty response rows
    new_df = new_df[~(new_df["RESPONSE"] == " ") | (new_df["RESPONSE"].isna())]

    return new_df


if __name__ == "__main__":
    input_csv = "zhihu.csv"
    # Create a pandas dataframe from your dataset file(s)
    # Use the `dtype` parameter of `pd.read_csv`.
    df = pd.read_csv(input_csv)  # or any other way
    df = reformat_csv_to_openassistant(df)
    # Save the file in the Parquet format
    df.to_parquet("dataset.parquet", row_group_size=100, engine="pyarrow", index=False)
