import os
import torch
import pandas as pd

from torch.nn import CosineSimilarity
from transformers import AutoTokenizer

model = (
    "bert-base-multilingual-cased"  # "bert-base-multilingual-cased", "xlm-roberta-base"
)
tokenizer = AutoTokenizer.from_pretrained(model)

cos_fct = CosineSimilarity(dim=0)

MODEL_NAME = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-RoBERTa",
}


def read_data(lang):

    if lang == "en":
        df = pd.read_csv(
            "../download/xnli/train-en.tsv",
            sep="\t",
            on_bad_lines='skip',
            names=["sent1", "sent2", "label"]
        )
    else:
        df = pd.read_csv(
            os.path.join(
                "../download/xtreme_translations/XNLI/translate-train",
                f"en-{lang}-translated.tsv"
            ),
            sep="\t",
            on_bad_lines='skip',
            names=["sent1_en", "sent2_en", f"sent1_{lang}", f"sent2_{lang}"]
        )
    # print(df.head())

    return df


def tokenize_text(claim, evidence):

    inputs = tokenizer.encode_plus(
        claim,
        evidence,
        max_length=128,
        padding="max_length",
        truncation=True,
        truncation_strategy="only_second",
    )

    return torch.Tensor(inputs["input_ids"])


def main():

    langs = ["zh", "vi"]

    # Write to file
    file_report = open(f"{MODEL_NAME[model]}/similarity_report.txt", "w")

    # Calculate the similarity between English and target lang
    for lang in langs:

        print(f"langs: en + {lang}", file=file_report)

        df = read_data(lang)

        file_sim_score = open(f"{MODEL_NAME[model]}/similarity_report_en+{lang}.txt", "w")

        score, amount = 0, 0

        for item in df.iterrows():

            try:
                input_lang1 = tokenize_text(item[1]["sent1_en"], item[1]["sent2_en"])
                input_lang2 = tokenize_text(item[1][f"sent1_{lang}"], item[1][f"sent2_{lang}"])
                
                cos_sim = cos_fct(input_lang1, input_lang2)
                print(f"{cos_sim}", file=file_sim_score)

                score += cos_sim.item()
                amount += 1
            except Exception as e:
                print(e)
                print(item[1]["sent1_en"])
                print(item[1]["sent2_en"])
                print(item[1][f"sent1_{lang}"])
                print(item[1][f"sent2_{lang}"])
                print()

        score /= amount
        print(f"\tCosine Similarity: {score}", file=file_report)

        file_sim_score.close()

    file_report.close()


if __name__ == "__main__":
    main()
