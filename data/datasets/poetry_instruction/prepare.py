import json
import os
import random

import pandas as pd

# Read the CSV file into a pandas DataFrame. The CSV file was downloaded from Kaggle, and the source is linked in the README.

CURRENT_DIR = os.path.dirname(__file__)  # Gets directory path of the current python module
df = pd.read_csv(os.path.join(CURRENT_DIR, "PoetryFoundationData.csv"))

# The data in the CSV file is not formatted correctly, so we need to clean it up.
df["Title"] = df["Title"].replace("\n", "", regex=True).replace("\r", "", regex=True)
df["Title"] = df["Title"].str.strip()
df["Title"] = df["Title"].apply(lambda x: f'"{x}"')
df["Poem"] = df["Poem"].str.strip()
df["Poem"] = df["Poem"].str.replace("Translated from the French", "")

# "writing_prompts" are for tasks requesting the assistant to write a poem.
# "topic" or "notTopic" are used depending if the original dataset had a topic listed for the poem or not.
writing_prompts_topic = [
    "Write me a poem about $topic.",
    "I want a poem about $topic.",
    "Can you write a poem? Make it about $topic.",
    "Compose a poem, about $topic.",
    "Make a poem with themes of $topic." "Generate a poem with the following themes: $topic.",
]

writing_prompts_notTopic = [
    "Write me a poem.",
    "I want a poem.",
    "Can you write a poem?",
    "Compose a poem.",
    "Make a poem." "Generate a poem.",
]

# These are replies that the assistant can give to the user.
replies_topic = [
    "Here's a poem about $topic: \n$title\n$poem",
    "Sure, I can do that. Here's a poem about $topic. I call it $title: \n$poem",
    "Okay, a poem about $topic: \n$title\n$poem",
    "Of course! It's called $title: \n$poem",
    "It's called $title: \n$poem",
    "Here's your poem about $topic: \n$title\n$poem",
    "I've written a poem for you about $topic. The title is $title: \n$poem",
    "Here's a beautiful poem about $topic for you. It's called $title: \n$poem",
    "This is a poem about $topic that I just wrote. It's called $title: \n$poem",
    "Here's a poem I composed about $topic. It's called $title: \n$poem",
]

replies_notTopic = [
    "Here's a poem: \n$title\n$poem",
    "Sure, I can do that. Here's a poem. I call it $title: \n$poem",
    "Okay, a poem: \n$title\n$poem",
    "Of course! It's called $title: \n$poem",
    "It's called $title: \n$poem",
    "Here's your poem: \n$title\n$poem",
    "I've written a poem for you. The title is $title: \n$poem",
    "Here's a beautiful poem for you. It's called $title: \n$poem",
    "This is a poem that I just wrote. It's called $title: \n$poem",
    "Here's a poem I composed. It's called $title: \n$poem",
]

# "titling_prompts" are for tasks requesting that the assistant titles a poem. They make up 5% of the dataset.
titling_prompts = [
    "Title this poem: \n$poem",
    "Come up with a unique title for my poem: \n$poem",
    "What should I call this poem? \n$poem",
    "Name this poem: \n$poem",
    "What would be a good title for this poem? \n$poem",
    "I need help coming up with a title for my poem. \n$poem",
    "$poem\nWhat should I call this poem?",
]

titling_replies = [
    "Based on the poem, a good title could be $title.",
    "I suggest titling this poem $title.",
    "How about calling it $title?",
    "You could name this poem $title.",
    "The title that comes to mind is $title.",
    "Perhaps $title would be a fitting title for this poem.",
    "I think $title would be a great title for this poem.",
    "This poem seems like it could be called $title to me.",
    "$title is a good title for this poem.",
]

# Shuffling the dataset and delegating 5% to titling tasks.
# Calculating the number of titling tasks and writing tasks.
num_rows = len(df)
num_titling_tasks = int(num_rows * 0.05)
num_writing_tasks = num_rows - num_titling_tasks

# Shuffle the rows in the DataFrame.
df = df.sample(frac=1)

# Split the DataFrame into two DataFrames, one for titling tasks and one for writing tasks.
writing_tasks = df.iloc[:num_writing_tasks]
titling_tasks = df.iloc[num_writing_tasks:]

prepared_data = []

# Loop through the writing tasks and process them.
for index, row in writing_tasks.iterrows():
    # Get data from the entry
    poem = row["Poem"]
    topics = row["Tags"]
    title = row["Title"]
    author = row["Poet"]

    # Variables to store to instruction, reply, source, and metadata.
    instruction = ""
    reply = ""
    source = "PoetryFoundation.org" + " - " + author
    metadata = {"author": author, "title": title, "tags": str(topics), "task_type": "writing"}

    # If the entry has a topic, use the topic to generate a writing prompt and a reply.
    if topics != "nan":
        instruction = random.choice(writing_prompts_topic).replace("$topic", str(topics))
        reply = (
            random.choice(replies_topic).replace("$topic", str(topics)).replace("$title", title).replace("$poem", poem)
        )
    else:
        instruction = random.choice(writing_prompts_notTopic)
        reply = random.choice(replies_notTopic).replace("$title", title).replace("$poem", poem)

    entry = {"INSTRUCTION": instruction, "RESPONSE": reply, "SOURCE": source, "METADATA": json.dumps(metadata)}
    prepared_data.append(entry)

# Loop through the titling tasks and process them.
for index, row in titling_tasks.iterrows():
    # Get data from the entry
    poem = row["Poem"]
    topics = row["Tags"]
    title = row["Title"]
    author = row["Poet"]

    # Variables to store to instruction, reply, source, and metadata.
    instruction = random.choice(titling_prompts).replace("$poem", poem)
    reply = random.choice(titling_replies).replace("$title", title)
    source = "PoetryFoundation.org" + " - " + author
    metadata = {"author": author, "title": title, "tags": str(topics), "task_type": "titling"}

    # If the entry has a topic, use the topic to generate a writing prompt and a reply.

    entry = {"INSTRUCTION": instruction, "RESPONSE": reply, "SOURCE": source, "METADATA": json.dumps(metadata)}
    prepared_data.append(entry)

# Convert prepared_data to a DataFrame.
prepared_data = pd.DataFrame(prepared_data)

# Save the DataFrame to disk in the Parquet format
prepared_data.to_parquet("output.parquet", row_group_size=100, engine="pyarrow", index=False)

# Print the amount of entries in the final converted dataset
print(f"Prepared {len(df)} entries")
