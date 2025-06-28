from sklearn.metrics.pairwise import cosine_similarity
import torch

def find_duplicates(ticket_texts, ticket_ids, embeddings, threshold=0.8):
    """
    Find pairs of tickets with cosine similarity above a given threshold.
    Returns a list of (ticket_id1, ticket_id2, score).
    """
    duplicates = []
    sim_matrix = cosine_similarity(embeddings)

    num_tickets = len(ticket_texts)
    for i in range(num_tickets):
        for j in range(i + 1, num_tickets):
            score = sim_matrix[i][j]
            if score >= threshold:
                duplicates.append({
                    "ticket_1": ticket_ids[i],
                    "ticket_2": ticket_ids[j],
                    "similarity": round(score, 4),
                    "text_1": ticket_texts[i],
                    "text_2": ticket_texts[j]
                })
    return duplicates
