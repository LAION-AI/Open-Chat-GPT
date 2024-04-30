---
dataset_info:
  features:
    - name: TEXT
      dtype: string
    - name: METADATA
      dtype: string
    - name: SOURCE
      dtype: string
  splits:
    - name: train
      num_bytes: 211728118
      num_examples: 2781
  download_size: 125187885
  dataset_size: 211728118
license: mit
task_categories:
  - conversational
  - text2text-generation
language:
  - en
tags:
  - OpenAssistant
  - transcripts
  - subtitles
  - television
pretty_name: TV and Movie dialogue and transcript corpus
size_categories:
  - 1K<n<10K
---

# Dataset Card for "tv_dialogue"

This dataset contains transcripts for famous movies and TV shows from multiple
sources.

An example dialogue would be:

```
[PERSON 1] Hello
[PERSON 2] Hello Person 2!
How's it going?

(they are both talking)

[PERSON 1] I like being an example
on Huggingface!

They are examples on Huggingface.
CUT OUT TO ANOTHER SCENE

We are somewhere else
[PERSON 1 (v.o)] I wonder where we are?
```

All dialogues were processed to follow this format. Each row is a single episode
/ movie (**2781** rows total) Following the
[OpenAssistant](https://open-assistant.io/) format The METADATA column contains
dditional information as a JSON string.

## Dialogue only, with some information on the scene

| Show                      | Number of scripts | Via                                                                                                     | Source               |
| ------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------- | -------------------- |
| Friends                   | 236 episodes      | https://github.com/emorynlp/character-mining                                                            | friends/emorynlp     |
| The Office                | 186 episodes      | https://www.kaggle.com/datasets/nasirkhalid24/the-office-us-complete-dialoguetranscript                 | office/nasirkhalid24 |
| Marvel Cinematic Universe | 18 movies         | https://www.kaggle.com/datasets/pdunton/marvel-cinematic-universe-dialogue                              | marvel/pdunton       |
| Doctor Who                | 306 episodes      | https://www.kaggle.com/datasets/jeanmidev/doctor-who                                                    | drwho/jeanmidev      |
| Star Trek                 | 708 episodes      | http://www.chakoteya.net/StarTrek/index.html based on https://github.com/GJBroughton/Star_Trek_Scripts/ | statrek/chakoteya    |

## Actual transcripts with detailed information on the scenes

| Show          | Number of scripts | Via                                 | Source              |
| ------------- | ----------------- | ----------------------------------- | ------------------- |
| Top Movies    | 919 movies        | https://imsdb.com/                  | imsdb               |
| Top Movies    | 171 movies        | https://www.dailyscript.com/        | dailyscript         |
| Stargate SG-1 | 18 episodes       | https://imsdb.com/                  | imsdb               |
| South Park    | 129 episodes      | https://imsdb.com/                  | imsdb               |
| Knight Rider  | 80 episodes       | http://www.knightriderarchives.com/ | knightriderarchives |
