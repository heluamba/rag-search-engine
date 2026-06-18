This project is a **keyword search engine built with an inverted index**. 
It processes a collection of movie data by tokenizing, normalizing, and stemming text from titles and descriptions, 
then maps each token to the document IDs in which it appears. This enables fast and efficient full-text search without scanning all documents at query time.
The system also includes a command-line interface to build the index, search for keywords, and persist the index to disk using pickle for later reuse.
