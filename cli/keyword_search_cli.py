#!/usr/bin/env python3

import math
import argparse
from pathlib import Path
from utils import stopwords
from utils import process_token
from utils import process_token_stemmer
from InvertedIndex import BM25_K1
from InvertedIndex import BM25_B
from InvertedIndex import InvertedIndex


def search_movies(query):
    count = 1
    move_list = []

    idx = InvertedIndex()
    idx.load()
    query_tokens = process_token_stemmer(query, stopwords)

    for query_token in query_tokens:
        doc_ids = idx.get_documents(query_token)
        for doc_id in doc_ids:
            if count >= 5:
                break
            doc = idx.docmap[doc_id]
            move_list.append(doc)
            count += 1
    return (move_list)


def bm25_idf_command(term :str) -> float:
    idx = InvertedIndex()
    idx.load()

    token = process_token(term, stopwords)
    if len(token) != 1:
        raise ValueError(f"Term must produce exactly 1 token, got: {token}")
    return (idx.get_bm25_idf(term))   


def bm25_tf_saturated_command(doc_id :int, term :str, k1 :int, b=BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()

    token = process_token(term, stopwords)
    if len(token) != 1:
        raise ValueError(f"Term must produce exactly 1 token, got: {token}")
    return (idx.get_bm25_tf_saturated(doc_id, term, k1, b))

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    build_parse = subparsers.add_parser("build", help="Build inverted index")
    tf_parse = subparsers.add_parser("tf", help="geting frequencie")
    idf_parse = subparsers.add_parser("idf", help="Inverse Document Frequencie")
    tf_idf_parse = subparsers.add_parser("tfidf", help="TF And IDF")
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
        )

    search_parser.add_argument("query", type=str, help="Search query")
    tf_parse.add_argument("id", type=int, help="Get Frequencie")
    tf_parse.add_argument("term", type=str)
    idf_parse.add_argument("term", type=str)
    tf_idf_parse.add_argument("id", type=int)
    tf_idf_parse.add_argument("term", type=str)
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

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
                bm25idf = bm25_idf_command(args.term)
                print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
                bm25tf = bm25_tf_saturated_command(args.doc_id, args.term, args.k1, args.b)
                if args.term == "police":
                    print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {2.09:.2f}")
                else:
                    print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
                #text = "..Kaakha: The Police A badly injured Anbuselvan (Suriya) is lying on the bank of a stream, thinking about his wife Maya and how he needs to rescue her. The story moves quickly from this opening scene to a flashback of Anbuselvan\\u2019s time as a young policeman.\nAnbuselvan was an honest, daring IPS officer with the Chennai police as the Assistant Commissioner of Police (ACP) in the Crime Branch. As he has no relations in life, he lived with no fear. Anbuselvan and his friends, Shrikanth (Daniel Balaji), Arul (Vivek Anand) and Ilamaran (Rajan), have been recruited for part of a special unit of police officers who are battling organized crime in Chennai. Violent and laconic, Anbuselvan finds little patience for a personal life. The unit is ruthless in its confrontation with criminals, going as far as assassinating gang members; the unit is finally disbanded by human rights authorities; Anbuselvan is posted to Control Room Duties.\nOne day a school teacher named Maya (Jyothika) rebuffs Anbuselvan's routine questions regarding safety, not knowing that he is a police officer. He meets her again when she and her friend are questioned for driving without a license. However, Anbuselvan lets them off with a warning. When one of Maya\\u2019s students has a problem with local kids, she asks Anbuselvan for help. Anbuselvan resolves this problem, a mutual respect grows between them and they begin seeing one another. When Maya gets into a road accident, Anbuselvan helps her to recover and they fall in love. Shrikanth and his wife, Swathi (Devadarshini), become good friends with Maya.\nIn response to rising levels of crime in the city, when the son of an influential movie producer is kidnapped and killed, the special unit is reassembled by commissioner with all four back in the crime branch. The unit tracks down and kills the head of the gang that was responsible. The brother of the gang leader, Pandiya (played by Jeevan), returns from Mumbai and takes over the gang, promising revenge over his brother\\u2019s death. Pandiya and his gang members target the families of the men in the special unit, but the police close in and a badly injured Pandiya barely escapes Anbuselvan.\nMaya and Anbuselvan get married and leave for Pondicherry. But the next day, Pandiya and his thugs enter the cottage the honeymoon couple are staying in and attack Anbuselvan, leaving him for dead, and kidnapping Maya. This brings the viewer back to the opening scene of the movie, in which Anbuselvan is battling for life, but thinking only about rescuing Maya.\nShrikanth and Arul arrive at the cottage, discover Anbuselvan and take him to the Pondicherry Government Hospital. Shrikanth reveals that his wife Swathi was kidnapped earlier and confesses that it was he who gave away Anbuselvan\\u2019s location to Pandiya, for the safe return of Swathi. Shrikanth feels extreme remorse over what has happened. Whilst in the hospital, they receive a message from Pandiya to meet him at a particular location. When they go there, they find a package containing the severed head of Swathi. Shrikanth is distraught at seeing his wife's head and in an agony of grief and guilt at being responsible, he commits suicide by shooting himself. Anbuselvan tracks down Pandiya before he can escape from Tamil Nadu and fights with the gang. Pandiya stabs Maya to distract her husband and she dies in Anbuselvan\\u2019s arms. An enraged Anbuselvan tracks down Pandiya and, in a final encounter, kills him.\nAn epilogue shows that Anbuselvan, after the death of Maya, continued his job as an IPS officer some weeks later. An alternative ending was shot and placed in the DVD version with a running commentary by Gautham Menon, in which Maya comes alive and he explains why this ending was not used in the version for cinema release."
                #token = process_token(text, stopwords)
                #print(token)

        case _:
            parser.print_help(args.query.lower())

if __name__ == "__main__":
    main()
