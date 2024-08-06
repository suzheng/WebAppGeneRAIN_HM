import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

def find_closest_genes(gene_id, gene_embeddings, n=10, search_option="Human and Mouse genes"):
    target_embedding = gene_embeddings[gene_id].reshape(1, -1)  # Reshape to 2D array

    if search_option == "Human genes":
        gene_pool = [g for g in gene_embeddings if not g.startswith("m_")]
    elif search_option == "Mouse genes":
        gene_pool = [g for g in gene_embeddings if g.startswith("m_")]
    else:  # "Human and Mouse genes"
        gene_pool = list(gene_embeddings.keys())

    similarities = [(g, cosine_similarity(target_embedding, gene_embeddings[g].reshape(1, -1))[0][0]) 
                    for g in gene_pool if g != gene_id]

    return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]

def calculate_similarity(gene1, gene2, gene_embeddings):
    similarity = 1 - cosine(gene_embeddings[gene1], gene_embeddings[gene2])
    
    # Calculate rank and quantile for gene1
    similarities_1 = [1 - cosine(gene_embeddings[gene1], emb) for emb in gene_embeddings.values()]
    rank_1 = sum(s > similarity for s in similarities_1)
    quantile_1 = np.percentile([s for s in similarities_1 if s < 1], (1 - rank_1 / len(similarities_1)) * 100)
    
    # Calculate rank and quantile for gene2
    similarities_2 = [1 - cosine(gene_embeddings[gene2], emb) for emb in gene_embeddings.values()]
    rank_2 = sum(s > similarity for s in similarities_2)
    quantile_2 = np.percentile([s for s in similarities_2 if s < 1], (1 - rank_2 / len(similarities_2)) * 100)
    
    return similarity, rank_1, quantile_1, rank_2, quantile_2

def gene_calculation(gene_a, gene_b, gene_c, gene_embeddings):
    result_vector = (gene_embeddings[gene_b] - gene_embeddings[gene_a] + gene_embeddings[gene_c])
    similarities = [(gene, 1 - cosine(embedding, result_vector)) 
                    for gene, embedding in gene_embeddings.items()
                    if gene not in [gene_a, gene_b, gene_c]]  # Exclude input genes
    return sorted(similarities, key=lambda x: x[1], reverse=True)[0]
