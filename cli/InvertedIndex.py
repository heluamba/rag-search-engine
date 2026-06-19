
import os
import pickle
from collections import Counter
from utils import process_token
from utils import load_movies
from utils import load_stopwords
from utils import stopwords


class   InvertedIndex():
	def __init__(self):
		self.index = {}
		self.docmap = {}
		self.term_frequencies = {}

	def __add_document(self, doc_id, text):
		tokens = process_token(text, stopwords)

		if doc_id not in self.term_frequencies:
			self.term_frequencies[doc_id] = Counter()
		
		for token in tokens:
			if token not in self.index:
				self.index[token] = set()
			self.index[token].add(doc_id)
		
		self.term_frequencies[doc_id][token] += 1

	def get_documents(self, term):
		docs_ids = self.index.get(term, set())
		return sorted(docs_ids)

	def	get_tf(self, doc_id, term):
		return (self.term_frequencies.get(doc_id, 0).get(term, 0))

	def	build(self):
		cinema = load_movies()

		for movie in cinema['movies']:
			doc_id = movie['id']
			
			self.docmap[doc_id] = movie

			text = f"{movie['title']} {movie['description']}"
			self.__add_document(doc_id, text)


	def	save(self):
		os.makedirs("cache", exist_ok=True)

		with open("cache/index.pkl", "wb") as f:
			pickle.dump(self.index, f)

		with open("cache/docmap.pkl", "wb") as f:
			pickle.dump(self.docmap, f)

		with open("cache/term_frequencies.pkl", "wb") as f:
			pickle.dump(self.term_frequencies, f)


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