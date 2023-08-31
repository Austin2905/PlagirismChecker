from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

input_source = str(input("Enter the source:"))
comparison_source = str(input("Enter the text to be compared:"))
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
encoded_input_source = tokenizer(input_source, padding=True, truncation=True, return_tensors='pt')
encoded_comparison_source = tokenizer(comparison_source, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    model_output_input = model(**encoded_input_source)
    model_output_comparison = model(**encoded_comparison_source)

input_sentence_embedding = mean_pooling(model_output_input, encoded_input_source['attention_mask'])
comparison_sentence_embedding = mean_pooling(model_output_comparison, encoded_comparison_source['attention_mask'])
input_sentence_embedding = F.normalize(input_sentence_embedding, p=2, dim=1)
comparison_sentence_embedding = F.normalize(comparison_sentence_embedding, p=2, dim=1)
similarity_score = cosine_similarity(input_sentence_embedding, comparison_sentence_embedding)
print("Similarity:", *similarity_score*100,"%")
