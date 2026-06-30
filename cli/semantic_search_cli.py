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



def main() -> None:
	parser = argparse.ArgumentParser(description="Semantic Search CLI")

	subparsers = parser.add_subparsers(dest="command", help="Available commands")
	
	verify_parse =  subparsers.add_parser("verify", help="Verify the model")
	embed_parse = subparsers.add_parser("embed_text", help="Generate Embedding text")
	verify_embed_parse = subparsers.add_parser("verify_embeddings", help="verify embeddings")
	embed_query_parse = subparsers.add_parser("embed_query", help="Process embed query")
	search_parse = subparsers.add_parser("search", help="Semantic Search")
      
	embed_parse.add_argument("text", type=str, help="Genarate Embeddings")
	embed_query_parse.add_argument("query", type=str, help="Query embedding")
	search_parse.add_argument("query", type=str, help="Semantic Search!")
	search_parse.add_argument("--limit", type=int, default=5, help="Maximum number of results")

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
	
		case _:
			parser.print_help()


if __name__ == "__main__":
    main()