import secrets


def _get_header(token):
    return f'''
rule dyn_biosens_append_linker_{token}:'''


def _get_benchmark(benchmark_out):
    return f'''
    benchmark:
        "{benchmark_out}"'''


def _get_main(dir_in, linker_in, dir_out):
    return f'''
    input:
         dir_in="{dir_in}",
         linker_in="{linker_in}",
    output:
         dir_out=directory("{dir_out}")
    threads:
         1000
    params:
         snakefile="nodes/dyn_biosens/append_linker/Snakefile",
         configfile="nodes/dyn_biosens/append_linker/config.yaml"
    run:
        with WorkflowExecuter(dict(input), dict(output), params.configfile, cores=CORES) as e:
            shell(f"""{{e.snakemake}} -s {{params.snakefile}} --configfile {{params.configfile}}""")
'''


def rule(dir_in, linker_in, dir_out, benchmark_dir=None):
    # """
    # Computes the maximum allowed dimension for ngram-based encodings.
    #
    # Category: utils. \n
    # Node: dim_size
    #
    # :param fasta_in: The path to the fasta file.
    # :param length_out: The path to the output file for a specific type and alphabet size.
    # :param benchmark_dir: The path to the directory to store the benchmark results. If None,
    #        benchmark will be not executed (default).
    #
    # :return: A string object representing a Snakemake rule.
    # """
    token = secrets.token_hex(4)
    rule = _get_header(token)
    if benchmark_dir is not None:
        benchmark_out = f"{benchmark_dir}utils_dim_size_{token}.txt"
        rule += _get_benchmark(benchmark_out)
    rule += _get_main(dir_in, linker_in, dir_out)
    return rule
