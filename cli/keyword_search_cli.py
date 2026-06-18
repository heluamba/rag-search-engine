#!/usr/bin/env python3

import argparse
from utils import *
from pathlib import Path
from InvertedIndex import InvertedIndex

def search_movies(cinema, query):
    #count = 1
    move_list = []
    idx = InvertedIndex()
    idx.load()
    query_tokens = process_token(query, stopwords)

    for query_token in query_tokens:
        doc_ids = idx.get_documents(query_token)
        for doc_id in doc_ids:
            doc = idx.docmap[doc_id]
            print(doc['title'])
        #print(query_token + ": index of doc: ", doc_id)
        # match = any(q in t for q in query_tokens for t in title_tokens)
        # if match :
        #     move_list.append(title_tokens)
        #     print(f"{count}. {move['title']}")
        #     count += 1
    return (move_list)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    build_parse = subparsers.add_parser("build", help="Build inverted index")
    load_parse = subparsers.add_parser("load", help="Build loader indexs")

    search_parser.add_argument("query", type=str, help="Search query")
    #build_parse.add_argument("build", type=str, help="Build inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}\n")
            movies = load_movies()
            result = search_movies(movies, args.query.lower())
            #print(result)

        case "build":
            idx = InvertedIndex()
            
            idx.build()
            idx.save()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
