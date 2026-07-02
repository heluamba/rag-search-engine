
import argparse
from lib.semantic_search import SemanticSearch
from lib.semantic_search import *


def	search(query: str, limit: int=5):
	cinema = load_movies()
	documents = cinema['movies']
	
	ss = SemanticSearch()
	ss.load_or_create_embeddings(documents)
	embed_score = ss.search(query, limit)

	for i, embed in enumerate(embed_score):
		if i >= 5:
			break
		print(f"{i + 1}. {embed[1]['title']}: {embed[1]['description']} (score: {embed[0]})")


def	chunk_text(text: str, n: int, overlap: int):
	words  = text.split()

	chunks = []
	step = n - overlap

	for i in range(0, len(words), step):
		chunks.append(" ".join(words[i:i + n]))

	print(f"Chunking {len(text)} characters")
	for i, chk in enumerate(chunks, 1):
		print(f"{i}. {chk}")

	return (chunks)


def main() -> None:
	parser = argparse.ArgumentParser(description="Semantic Search CLI")

	subparsers = parser.add_subparsers(dest="command", help="Available commands")
	
	verify_parse =  subparsers.add_parser("verify", help="Verify the model")
	embed_parse = subparsers.add_parser("embed_text", help="Generate Embedding text")
	verify_embed_parse = subparsers.add_parser("verify_embeddings", help="verify embeddings")
	embed_query_parse = subparsers.add_parser("embed_query", help="Process embed query")
	search_parse = subparsers.add_parser("search", help="Semantic Search")
	chunk_parse = subparsers.add_parser("chunk", help="chunking the text")
	semantic_chunck_parse = subparsers.add_parser("semantic_chunk", help="A Semantic Chunk")

	embed_parse.add_argument("text", type=str, help="Genarate Embeddings")
	embed_query_parse.add_argument("query", type=str, help="Query embedding")
	search_parse.add_argument("query", type=str, help="Semantic Search!")
	search_parse.add_argument("--limit", type=int, default=5, help="Maximum number of results")
	chunk_parse.add_argument("text", type=str, help="Text to chunk")
	chunk_parse.add_argument("--chunk-size", type=int, default=200, help="size of text to chunk")
	chunk_parse.add_argument("--overlap", type=int, default=0, help="chunck overlap amout")
	semantic_chunck_parse.add_argument("text", type=str, help="text for semantic chunk")
	semantic_chunck_parse.add_argument("--max-chunk-size", type=int, default=4)
	semantic_chunck_parse.add_argument("--overlap", type=int, default=0)


	args = parser.parse_args()
	
	match args.command:
		case "verify":
			verify_model()

		case "embed_text":
			embed_text(args.text)
                  
		case "verify_embeddings":
			verify_embeddings()
		
		case "embed_query":
			embed_query_text(args.query)

		case "search":
			search(args.query, args.limit)

		case "chunk":
			chunk_text(args.text, args.chunk_size, args.overlap)	

		case "semantic_chunk":
			semantic_chunk(args.text, args.max_chunk_size, args.overlap)

		case _:
			parser.print_help()


if __name__ == "__main__":
    main()