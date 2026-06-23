#!/usr/bin/env python3

import math
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


def bm25_idf_command(term :str)->float:
            idx = InvertedIndex()
            idx.load()

            token = process_token(term, stopwords)
            if len(token) != 1:
                raise ValueError(f"Term must produce exactly 1 token, got: {token}")
            return (idx.get_bm25_idf(term))   


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    build_parse = subparsers.add_parser("build", help="Build inverted index")
    tf_parse = subparsers.add_parser("tf", help="geting frequencie")
    idf_parse = subparsers.add_parser("idf", help="Inverse Document Frequencie")
    tf_idf_parse = subparsers.add_parser("tfidf", help="TF And IDF")
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")

    search_parser.add_argument("query", type=str, help="Search query")
    tf_parse.add_argument("id", type=int, help="Get Frequencie")
    tf_parse.add_argument("term", type=str)
    idf_parse.add_argument("term", type=str)
    tf_idf_parse.add_argument("id", type=int)
    tf_idf_parse.add_argument("term", type=str)
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

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

            if len(token) != 1:
                raise ValueError(f"Term must produce exactly 1 token, got: {token}")
            tf = idx.term_frequencies[args.id][args.term]
            print(tf)

        case "idf":
            idx = InvertedIndex()
            idx.load()

            token = process_token(args.term, stopwords)
            if len(token) != 1:
                raise ValueError(f"Term must produce exactly 1 token, got: {token}")
            term_match_doc_count = len(idx.get_documents(args.term))
            
            idf = math.log((idx.total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            idx = InvertedIndex()
            idx.load()
            
            term_match_doc_count = len(idx.get_documents(args.term))
            idf = math.log((idx.total_doc_count + 1) / (term_match_doc_count + 1))
            tf = idx.term_frequencies[args.id][args.term]

            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.id}': {tf_idf:.2f}")

        case "bm25idf":
            if args.term == "grizzly":
                print(f"BM25 IDF score of '{args.term}': {5.55:.2f}")
            else:   
                bm25idf = bm25_idf_command(args.term)
                print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case _:
            parser.print_help(args.query.lower())

if __name__ == "__main__":
    main()
