"""CLI entry point for RAG system."""

import argparse
import json
import sys
from pathlib import Path

from src.rag.config import settings
from src.rag.ingest import ingest_all
from src.rag.retrieve import retrieve_with_debug


def cmd_ingest(args: argparse.Namespace) -> int:
    """Run document ingestion pipeline."""
    docs_dir = Path(args.docs_dir) if args.docs_dir else None

    print("=" * 60)
    print("Starting document ingestion...")
    print("=" * 60)

    try:
        stats = ingest_all(docs_dir)
        print("\n" + "=" * 60)
        print("Ingestion Summary:")
        print(f"  Documents: {stats['documents']}")
        print(f"  Chunks: {stats['chunks']}")
        print(f"  Index: {stats.get('index_path', 'N/A')}")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\nError during ingestion: {e}", file=sys.stderr)
        return 1


def cmd_search(args: argparse.Namespace) -> int:
    """Search for documents matching a query."""
    query = args.query
    k = args.k

    print("=" * 60)
    print(f"Query: {query}")
    print(f"Top-k: {k}")
    print("=" * 60)

    try:
        results = retrieve_with_debug(query, k)

        print(f"\nFound {results['num_results']} results:\n")

        for r in results["results"]:
            print(f"[{r['rank']}] Score: {r['score']:.4f}")
            print(f"    Chunk: {r['chunk_id']}")
            print(f"    Doc: {r['doc_id']} ({r['classification']})")
            print(f"    Text: {r['text_preview']}")
            print()

        if args.json:
            print("\n--- JSON Output ---")
            print(json.dumps(results, indent=2, ensure_ascii=False))

        return 0
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("Run 'python -m src.app ingest' first to create the index.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nError during search: {e}", file=sys.stderr)
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Show index information."""
    print("=" * 60)
    print("RAG System Information")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Embedding model: {settings.embedding_model}")
    print(f"  Chunk size: {settings.chunk_size} chars")
    print(f"  Chunk overlap: {settings.chunk_overlap} chars")
    print(f"  Default top-k: {settings.default_top_k}")
    print(f"  Max top-k: {settings.max_top_k}")

    print(f"\nPaths:")
    print(f"  Docs directory: {settings.docs_dir}")
    print(f"  Index directory: {settings.index_dir}")

    # Check if index exists
    print(f"\nIndex Status:")
    if settings.faiss_index_path.exists():
        print(f"  FAISS index: {settings.faiss_index_path} (exists)")
    else:
        print(f"  FAISS index: Not created (run 'ingest' first)")

    if settings.docstore_path.exists():
        with open(settings.docstore_path) as f:
            docstore = json.load(f)
        print(f"  Docstore: {settings.docstore_path}")
        print(f"    Total chunks: {docstore['metadata']['total_chunks']}")
    else:
        print(f"  Docstore: Not created (run 'ingest' first)")

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enterprise RAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the index")
    ingest_parser.add_argument(
        "--docs-dir",
        type=str,
        help="Directory containing documents (default: data/docs)",
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search the document index")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "-k",
        type=int,
        default=settings.default_top_k,
        help=f"Number of results (default: {settings.default_top_k})",
    )
    search_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    # Info command
    subparsers.add_parser("info", help="Show system information")

    args = parser.parse_args()

    if args.command == "ingest":
        return cmd_ingest(args)
    elif args.command == "search":
        return cmd_search(args)
    elif args.command == "info":
        return cmd_info(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
