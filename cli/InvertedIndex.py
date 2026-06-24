
import os
import math
import pickle
from collections import Counter
from utils import process_token
from utils import process_token_stemmer
from utils import load_movies
from utils import load_stopwords
from utils import stopwords

BM25_K1 = 1.5
BM25_B = 0.75
CACHE_DIR = "cache/"

class   InvertedIndex():
	def __init__(self):
		self.index = {}
		self.docmap = {}
		self.doc_lengths = {}
		self.term_frequencies = {}
		self.total_doc_count = 0
		self.lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")


	def __add_document(self, doc_id, text):
		tokens = process_token_stemmer(text, stopwords)
		
		if doc_id in self.doc_lengths:
			self.doc_lengths[doc_id] += len(tokens)
		else:
			self.doc_lengths[doc_id] = len(tokens)

		if doc_id not in self.term_frequencies:
			self.term_frequencies[doc_id] = Counter()
			self.total_doc_count += 1
		
		for token in tokens:
			if token not in self.index:
				self.index[token] = set()
			self.index[token].add(doc_id)
		
			self.term_frequencies[doc_id][token] += 1

	def	__get_avg_doc_length(self) -> float:
		if self.total_doc_count == 0:
			return (0)
		
		total_doc_length = sum(self.doc_lengths.values())
		
		return (total_doc_length / self.total_doc_count)


	def get_documents(self, term):
		docs_ids = self.index.get(term, set())
		return sorted(docs_ids)


	def	get_tf(self, doc_id, term):
		return (self.term_frequencies[doc_id][term])


	def	get_bm25_idf(self, term: str) ->float:
		docs_term_count = len(self.get_documents(term))

		docs_without_term = self.total_doc_count - docs_term_count + 0.5
		docs_with_term = docs_term_count + 0.5
		mb25 = math.log((docs_without_term / docs_with_term) + 1)
		return (mb25)


	def	get_bm25_tf_saturated(self, doc_id :int, term :str, k1=BM25_K1, b=BM25_B):
		tf = self.get_tf(doc_id, term)

		avg_doc_length = self.__get_avg_doc_length()
		length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length)
		
		tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
		return (tf_component)



	def	build(self):
		cinema = load_movies()

		for movie in cinema['movies']:
			doc_id = movie['id']
			
			self.docmap[doc_id] = movie

			text = f"{movie['title']} {movie['description']}"
			self.__add_document(doc_id, text)



	def	save(self):
		os.makedirs("cache", exist_ok=True)

		with open("cache/total_doc_count.pkl", "wb") as f:
			pickle.dump(self.total_doc_count, f)

		with open("cache/index.pkl", "wb") as f:
			pickle.dump(self.index, f)

		with open("cache/docmap.pkl", "wb") as f:
			pickle.dump(self.docmap, f)

		with open("cache/term_frequencies.pkl", "wb") as f:
			pickle.dump(self.term_frequencies, f)

		with open(self.lengths_path, "wb") as f:
			pickle.dump(self.doc_lengths, f)


	def	load(self):
		try:
			with open("cache/index.pkl", "rb") as f:
				self.index =  pickle.load(f)
		except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
			print("Error load file: cache/index.pkl", e)

		try:
			with open("cache/docmap.pkl", "rb") as f:
				self.docmap = pickle.load(f)
		except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
			print("Error load file: cache/index.pkl", e)

		try:
			with open("cache/term_frequencies.pkl", "rb") as f:
				self.term_frequencies = pickle.load(f)
		except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
			print("Error load file: cache/index.pkl", e)

		try:
			with open("cache/total_doc_count.pkl", "rb") as f:
				self.total_doc_count = pickle.load(f)
		except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
			print("Error load file: cache/total_doc_count.pkl", e)
		
		try:
			with open(self.lengths_path, "rb") as f:
				self.doc_lengths = pickle.load(f)
		except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
			print(f"Error load file: {self.lengths_path}", e)