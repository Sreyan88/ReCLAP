## Caption Augumentation

To run caption augmentation, first run `stage_1.py` and then run `stage_2.py` after updating the key and file paths in the files.

## Prompt Augmentation

After caption augmentation, run `llama_rephrase.py` for prompt augmentation after replacing `hf_key` with your huggingface key.

## Parallel Generation

We also provide `generate.sh` to aid parallel generation over multiple GPUSs.

## Format

To format the out of augmentations, use `formatter.py`.