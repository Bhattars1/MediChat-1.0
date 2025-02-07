from datasets import load_dataset

# Load PubMedQA dataset

def load_plubmed():
    dataset = load_dataset("pubmed_qa", "pqa_labeled")
    return dataset