#### Overview

Accompanying code repository for the publication __Multivalent Binding Resolved by Fluorescence Proximity Sensing__ by Schulte 
*et al.* (2022). Manuscript submitted for publication.

#### Setup

1. Clone the PEPTIDE REACToR repo and follow the installation instructions:  
    `git clone git@github.com:spaenigs/peptidereactor.git`.
2. Create directory `data/fps/` and insert data (see supplementary data of manuscript).
3. Copy `nodes/fps` into the node directory.
4. Replace the original `main.py` with `./main.py` (recommended) or add the following rule after `aggregate_directories`:
   ```python
   w.add(fps.append_linker.rule(
        dir_in=f"data/temp/{TOKEN}/{{dataset}}/all/",
        linker_in="data/{dataset}/data.yaml",
        dir_out=all_encodings_dir, benchmark_dir=w.benchmark_dir))
   ``` 
#### Manual execution

1. Store peptide names in `data/fps/all_names.txt` and sequences in `data/fps/all_peptides.txt`.
2. Execute `scripts/process_linkers.py` to create the required `seqs.fasta` and `classes.txt` files.
3. Execute `main.py`. This will generate all possible encoded datasets, and append the linkers to the encoded sequences.
4. Execute `scripts/parse_and_add_kd.py` to replace the dummy classes with the actual K<sub>D</sub> values.
5. Execute `scripts/regression_final.py` to run the analysis. Replace `"data/fps/csv/all_kd/dist_freq_dn_100_dc_100.csv"` 
   with the path to your desired encoded dataset.

#### Automated analysis and graphics

1. Run `worklflows/process_data.smk` (requires Snakemake, see: https://snakemake.readthedocs.io/en/stable/).
2. Run `workflows/group_assoc.smk` to visualize the different iterations.
