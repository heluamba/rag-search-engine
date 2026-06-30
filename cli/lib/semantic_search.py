import os
import numpy as np
from utils import load_movies
from sentence_transformers import SentenceTransformer


EMBED_PATH = "cache/movie_embeddings.npy"

#VECTORS FUNC

def add_vectors(vec1, vec2):
	if len(vec1) != len(vec2):
		raise ValueError(f"Diferent length between vec1:{len(vec1)} and vec2:{len(vec2)}")
	
	new_vec = [a + b for a, b in zip(vec1, vec2)]
	return (new_vec)


def dot(vec1: list[float], vec2: list[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("vectors must be the same length")
    total = 0.0
    for i in range(len(vec1)):
        total += vec1[i] * vec2[i]
    return total


def	subtract_vectors(vec1, vec2):
	if len(vec1) != len(vec2):
		raise ValueError(f"Diferent length between vec1:{len(vec1)} and vec2:{len(vec2)}")

	new_vec = [a - b for a, b in zip(vec1, vec2)]
	return (new_vec)


def euclidean_norm(vec: list[float]) -> float:
    total = 0.0
    for x in vec:
        total += x**2

    return total**0.5


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

#CLASSE

class	SemanticSearch():
	def	__init__(self):
		self.embeddings = None
		self.documents = None
		self.document_map = {}
		self.model = SentenceTransformer('all-MiniLM-L6-v2')


	def generate_embedding(self, text: str):
		if not text.strip():
			raise ValueError("Text can't be empty")
		
		embeddings = self.model.encode([text])[0]
		return (embeddings)


	def	build_embeddings(self, documents):
		doc_list = []
		self.documents = documents

		for doc in documents:
			self.document_map[doc['id']] = doc
			doc_list.append(f"{doc['title']}: {doc['description']}")

		self.embeddings = self.model.encode(doc_list, show_progress_bar=True)

	
		np.save(EMBED_PATH, self.embeddings)
		return (self.embeddings)
	
	def load_or_create_embeddings(self, documents):

		if os.path.exists(EMBED_PATH):
			self.documents = documents

			for doc in documents:
				self.document_map[doc['id']] = doc

			self.embeddings = np.load(EMBED_PATH)
			if len(documents) == len(self.embeddings):
				return (self.embeddings)
		else:
			return (self.build_embeddings(documents))


	def	search(self, query, limit):
		if self.embeddings is None:
			raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
		query_embed = self.generate_embedding(query)

		similarity_score = []
		for i, embed in enumerate(self.embeddings):
			score = cosine_similarity(query_embed, embed)
			similarity_score.append((score, self.document_map[self.documents[i]['id']]))

		similarity_score.sort(key=lambda x: x[0], reverse=True)
		return (similarity_score[:limit])
		

#MODEL FUNC

def	verify_model():
	ss = SemanticSearch()

	print(f"Model loaded: {ss.model}")
	print(f"Max sequence length: {ss.model.max_seq_length}")

def	embed_text(text :str):
	ss = SemanticSearch()

	embedding = ss.generate_embedding(text)
	print(f"Text: {text}")
	print(f"First 3 dimensions: {embedding[:3]}")
	print(f"Dimensions: {embedding.shape[0]}")

def	verify_embeddings():
	cinema = load_movies()

	documents = cinema['movies']
	ss = SemanticSearch()
	
	embeddings = ss.load_or_create_embeddings(documents)
	print(f"Number of docs:   {len(documents)}")
	print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def	embed_query_text(query):
	ss = SemanticSearch()

	query_embed = ss.generate_embedding(query)
	print(f"Query: {query}")
	print(f"First 3 dimensions: {query_embed[:3]}")
	print(f"Shape: {query_embed.shape}")
	