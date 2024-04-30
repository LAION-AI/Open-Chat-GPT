# OA Pretokenizer Utility

The pretokenizer allows to tokenize datasets before training with the
[epfLLM/Megatron-LLM](https://github.com/epfLLM/Megatron-LLM) fork.

## Requirements

1. make sure the `model_training` module is installed:

```bash
pip install -e ..
```

2. Make sure the `oasst_data` module is installed:

```bash
python -m pip install ../../oasst-data/
```

### Configuration

The datamix to process can be configured with one or multiple sections in the
`configs/pretokenize.yaml` file.

### Example usage

```
python pretokenize.py --output_dir output--configs oasst_top1 llama2 --compress --write_json
```

### Help message

```
usage: pretokenize.py [-h] --configs CONFIGS [CONFIGS ...] [--output_dir OUTPUT_DIR] [--write_json] [--compress]

Tokenize datamixes for LLama2/Falcon fine-tuning with Megatron-LLM.

options:
  -h, --help            show this help message and exit

configuration:
  --configs CONFIGS [CONFIGS ...]
                        Configurations sections to apply (read from YAML, multiple can be specified).
  --output_dir OUTPUT_DIR
                        Path to output directory
  --write_json          Generate a JSONL file with the formatted dialogues (key='text').
  --compress            Generate a .tar.gz file of the output directory.
```
