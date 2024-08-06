import numpy as np
import gzip
import streamlit as st
import pandas as pd

@st.cache_data
def read_gene_embeddings(file_path):
    gene_embeddings = {}
    with gzip.open(file_path, 'rt') as file:
        next(file)
        for line in file:
            parts = line.strip().split()
            gene = parts[0]
            embedding = np.array([float(x) for x in parts[1:]])
            gene_embeddings[gene] = embedding
    return gene_embeddings

@st.cache_data
def load_gene_id_mapping(file_path):
    mapping_df = pd.read_csv(file_path, sep='\t', header=None, names=['ensembl_id', 'gene_symbol'])
    ensembl_to_symbol = dict(zip(mapping_df['ensembl_id'], mapping_df['gene_symbol']))
    symbol_to_ensembl = {v: k for k, v in ensembl_to_symbol.items() if v != k}
    return ensembl_to_symbol, symbol_to_ensembl

# all dicts without 'm_' prefix
human_ensembl_to_symbol, human_symbol_to_ensembl = load_gene_id_mapping("data/gencode.v44.annotation.gff3_ID_Mapping.txt.clean") 
mouse_ensembl_to_symbol, mouse_symbol_to_ensembl = load_gene_id_mapping("data/gencode.vM33.annotation.gff3_ID_Mapping.txt.clean")

# This is always with 'm_' prefix
gene_embeddings = read_gene_embeddings("data/Human_mouse_gene_embeddings_from_coding_lncRNA_pseudogene_model.txt.gz")  # Load gene embeddings

# List of genes with same symbols in human and mouse
ambiguous_genes = set(['C2', 'C3', 'C6', 'C7', 'C9', 'C9orf72', 'F10', 'F11', 'F12', 'F2', 'F3', 'F5', 'F7', 'F8', 'F9', 'H19', 'LTO1'])

def is_ensembl_id(gene):
    if gene.startswith('m_'):
        gene = gene[2:]
    return gene.startswith('ENSG') or gene.startswith('ENSMUSG')

def is_mouse_gene(gene):
    if gene.startswith('m_'):
        gene = gene[2:]
    return (gene in mouse_symbol_to_ensembl or 
            gene in mouse_ensembl_to_symbol or 
            f"m_{gene}" in mouse_symbol_to_ensembl or 
            f"m_{gene}" in mouse_ensembl_to_symbol)

def check_ambiguous_gene(symbol):
    if symbol in ambiguous_genes:
        species = st.radio(f"{symbol} exists in both human and mouse. Please specify:", ("Human", "Mouse"))
        return f"m_{symbol}" if species == "Mouse" else symbol
    return symbol

def process_gene(gene, do_check_ambiguous_gene=False):
    """
    Always return mouse gene symbols and ensembl IDs with "m_" added
    """
    gene = gene.strip()
    if is_ensembl_id(gene):
        if is_mouse_gene(gene):
            symbol = mouse_ensembl_to_symbol.get(gene.replace('m_', ''), gene)
            ensembl = gene if gene.startswith('m_') else f"m_{gene}"
            return (f"m_{symbol}", ensembl)
        else:
            symbol = human_ensembl_to_symbol.get(gene, gene)
            return (symbol, gene)
    else:
        if do_check_ambiguous_gene:
            gene = check_ambiguous_gene(gene)
        if is_mouse_gene(gene):
            symbol = gene if gene.startswith('m_') else f"m_{gene}"
            ensembl = mouse_symbol_to_ensembl.get(gene.replace('m_', ''), gene.replace('m_', ''))
            return (symbol, f"m_{ensembl}")
        else:
            ensembl = human_symbol_to_ensembl.get(gene, gene)
            return (gene, ensembl)

def validate_gene(gene_tuple):
    symbol, ensembl = gene_tuple
    return symbol in gene_embeddings

def process_gene_list(input_genes, do_check_ambiguous_gene=True):
    """
    Processes a list of input genes, validates them, and separates them into valid and invalid gene lists.

    Args:
        input_genes (str): A string containing gene identifiers separated by spaces. These can be gene symbols or Ensembl IDs.

    Returns:
        tuple: A tuple containing two lists:
            - valid_genes (list): A list of tuples where each tuple contains a valid gene symbol and its corresponding Ensembl ID.
            - invalid_genes (list): A list of gene symbols that could not be validated against the gene embeddings.

    Notes:
        - The function handles both human and mouse genes. Mouse genes are prefixed with 'm_'.
        - Ambiguous genes (i.e., genes with the same symbol in both human and mouse) are resolved by user input via Streamlit radio buttons.
        - The function relies on pre-loaded dictionaries and gene embeddings for validation.
    """
    if isinstance(input_genes, str):
        gene_list = [gene.strip() for gene in input_genes.split(",") if gene.strip()]
    elif isinstance(input_genes, list):
        gene_list = [gene.strip() for gene in input_genes if gene.strip()]
    else:
        raise ValueError("Input genes should be either a string or a list.")
    processed_genes = [process_gene(gene, do_check_ambiguous_gene=do_check_ambiguous_gene) for gene in gene_list]
    valid_genes = [gene for gene in processed_genes if validate_gene(gene)]
    invalid_genes = [gene[0] for gene in processed_genes if not validate_gene(gene)]
    return valid_genes, invalid_genes
