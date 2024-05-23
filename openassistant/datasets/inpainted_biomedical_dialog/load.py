from datasets import load_dataset

if __name__ == "__main__":
    ds = load_dataset("ericyu3/openassistant_inpainted_dialogs_5k_biomedical")
    print(ds["train"])
