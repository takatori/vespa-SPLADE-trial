#!usr/bin/env python
import bz2
from vespa.application import Vespa
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("prithivida/Splade_PP_en_v1")
model = AutoModelForMaskedLM.from_pretrained("prithivida/Splade_PP_en_v1")


def generate_bulk_buffer():
    buf = []
    with bz2.open(
        # "dataset/simplewiki-202109-pages-with-pageviews-20211001.bz2", "rt"
        "dataset/small.csv.bz2",
        "rt",
    ) as bz2f:
        for line in bz2f:
            id, title, text, pageviews = line.rstrip().split("\t")
            splade_rep = convert_splade_rep(text)
            splade_rep = {k: int(v * 100) for k, v in splade_rep}
            data = {
                "id": id,
                "fields": {
                    "title": title,
                    "text": text,
                    "sparse_rep": splade_rep,
                    "pageviews": pageviews,
                },
            }
            buf.append(data)
            if 500 <= len(buf):
                print(f"Yielding {len(buf)} documents")
                yield buf
                buf.clear()
    if buf:
        yield buf


client = Vespa(url="http://vespa", port=8080)


def callback(response, id):
    if not response.is_successful():
        print(
            f"Failed to feed document {id} with status code {response.status_code}: Reason {response.get_json()}"
        )


def convert_splade_rep(sentence):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}
    model.to(device)

    inputs = tokenizer(
        sentence, return_tensors="pt", max_length=512, truncation=True, padding=True
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    input_ids = inputs["input_ids"]

    attention_mask = inputs["attention_mask"]

    outputs = model(**inputs)

    logits, attention_mask = outputs.logits, attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    max_val, _ = torch.max(weighted_log, dim=1)
    vector = max_val.squeeze()

    cols = vector.nonzero().squeeze().cpu().tolist()
    print("number of actual dimensions: ", len(cols))
    weights = vector[cols].cpu().tolist()

    d = {k: v for k, v in zip(cols, weights)}
    sorted_d = {
        k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)
    }
    bow_rep = []
    for k, v in sorted_d.items():
        bow_rep.append((reverse_voc[k], round(v, 2)))

    # print("SPLADE BOW rep:\n", bow_rep)
    return bow_rep


for buf in generate_bulk_buffer():
    client.feed_iterable(buf, schema="simplewiki", callback=callback)
