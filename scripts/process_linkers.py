from itertools import zip_longest
from modlamp.core import save_fasta

import pandas as pd
import numpy as np

import re
import yaml

with open("data/fps/all_names.txt") as f:
    names = [l.rstrip() for l in f.readlines()]

with open("data/fps/all_peptides.txt") as f:
    seqs = [l.rstrip() for l in f.readlines()]

linkers = []
peptides = []
for s, n in zip(seqs, names):
    hits = re.findall("(.*?)([OJK]$|[OJK]{2,})", s)
    if len(hits) == 1:
        peptide, linker = hits[0]
        if s == "YSIVGSYPROJKC":
            linker += "C"
    else:
        peptide, linker = s, ""
    peptides.append(peptide)
    linkers.append(linker)


max_linker_len = sorted([len(l) for l in linkers])[-1]
processed_linkers = []
for l in linkers:
    tmp = [x if x is not None else y for x, y in zip_longest(l, ["-"] * max_linker_len)]
    bin_ = "".join(["11" if i == "K" else
                    "01" if i == "J" else
                    "10" if i == "O" else
                    "CC" if i == "C" else
                    "00" for i in tmp])
    processed_linkers.append(bin_)

kd_values = pd.read_csv("data/fps/kd_values.csv")
res = {}
for i, (seq, name, linker, processed_linker) in enumerate(zip(peptides, names, linkers, processed_linkers), start=1):
    seq_name = f"Seq_{i}"
    if name in list(kd_values.Peptide):
        cl = 1
    else:
        cl = 0
    if seq in res:
        res[seq][name] = {"linker": linker, "processed_linker": processed_linker, "class": cl}
    else:
        res[seq] = {name: {"linker": linker, "processed_linker": processed_linker, "class": cl}, "seq_name": seq_name}

with open("data/fps/data.yaml", "w") as f:
    yaml.safe_dump(res, f)

with open("data/fps/seqs.fasta", "w") as f, \
        open("data/fps/classes.txt", "w") as f2:
    for k, v in res.items():
        f.write(f">{res[k]['seq_name']}\n")
        f.write(f"{k}\n")
        f2.write(f"{np.random.choice([0, 1])}\n")
