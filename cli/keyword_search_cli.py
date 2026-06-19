#!/usr/bin/env python3

import argparse
from pathlib import Path
from utils import stopwords
from utils import process_token
from InvertedIndex import InvertedIndex


def search_movies(query):
    count = 1
    move_list = []

    idx = InvertedIndex()
    idx.load()
    query_tokens = process_token(query, stopwords)

    for query_token in query_tokens:
        doc_ids = idx.get_documents(query_token)
        for doc_id in doc_ids:
            if count >= 5:
                break
            doc = idx.docmap[doc_id]
            move_list.append(doc)
            count += 1
    return (move_list)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    build_parse = subparsers.add_parser("build", help="Build inverted index")
    tf_parse = subparsers.add_parser("tf", help="geting frequencie")

    search_parser.add_argument("query", type=str, help="Search query")
    tf_parse.add_argument("id", type=int, help="Get Frequencie")
    tf_parse.add_argument("term", type=str)
    #build_parse.add_argument("build", type=str, help="Build inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}\n")
            result = search_movies(args.query.lower())

        case "build":
            idx = InvertedIndex()
            
            idx.build()
            idx.save()

        case "tf":
            idx = InvertedIndex()
            idx.load()

            print(f"Getting Frequencie: {args.id}\n")
            token = process_token(args.term, stopwords)

            print(doc['title'])
            if len(token) != 1:
                raise ValueError(f"Term must produce exactly 1 token, got: {token}")
            doc_id = idx.get_tf(doc['id'], token[0])
            #print(doc_id)

        case _:
            parser.print_help(args.query.lower())

if __name__ == "__main__":
    main()
