---
dataset_info:
  features:
    - name: conversation
      dtype: string
  splits:
    - name: train
      num_bytes: 87758119
      num_examples: 1322
    - name: validation
      num_bytes: 7731418
      num_examples: 111
    - name: test
      num_bytes: 27041394
      num_examples: 331
  download_size: 63044464
  dataset_size: 122530931
---

# Dataset Card for "poetry-instructions"

A dataset of user-assistant dialogue instructions for guided poetry creation.
Poems used were taken from
[merve/poetry](https://huggingface.co/datasets/merve/poetry) and
[matthh/gutenberg-poetry-corpus](https://huggingface.co/datasets/matthh/gutenberg-poetry-corpus).

The dataset contains dialogues in the following formats:

- Poetry Completion:

```
User: Can you continue this poem for me? <poem_start>
Assistant: Sure, a continuation for this poem could be: <poem end>
```

- Create poem in style of (?):

```
User: Can you write a poem for me in the style of <author>?
Assistant: Sure, here's a poem in the style of <author>: <poem>
```

- Creat poem about (?):

```
User: Can you write me a poem about <keywords (extracted using keyphrase model)>?
Assistant: Sure, here's a poem about <keywords>: <poem>
```

- Create poem about (?) in the style of (?):

```
User: Can you write me a poem about <keywords> in the style of <author>?
Assistant: Sure, here's a poem about <keywords> in the style of <author>: <poem>
```
