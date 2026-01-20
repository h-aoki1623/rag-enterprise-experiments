"""CLI entry point for RAG system."""

import argparse
import json
import sys
from pathlib import Path

from src.rag.config import settings
from src.rag.generate import generate_with_retrieval
from src.rag.ingest import ingest_all
from src.rag.retrieve import retrieve_hierarchical_with_debug, retrieve_with_debug


def cmd_ingest(args: argparse.Namespace) -> int:
    """Run document ingestion pipeline."""
    docs_dir = Path(args.docs_dir) if args.docs_dir else None
    use_hierarchy = not args.flat  # --flat disables hierarchy

    print("=" * 60)
    print("Starting document ingestion...")
    print(f"Mode: {'Flat' if args.flat else 'Hierarchical'} chunking")
    print("=" * 60)

    try:
        stats = ingest_all(docs_dir, use_hierarchy=use_hierarchy)
        print("\n" + "=" * 60)
        print("Ingestion Summary:")
        print(f"  Documents: {stats['documents']}")
        if stats.get("hierarchy_enabled"):
            print(f"  Parent chunks: {stats.get('parents', 0)}")
            print(f"  Child chunks: {stats.get('children', 0)}")
        else:
            print(f"  Chunks: {stats.get('chunks', 0)}")
        print(f"  Hierarchy: {'Enabled' if stats.get('hierarchy_enabled') else 'Disabled'}")
        print(f"  Index: {stats.get('index_path', 'N/A')}")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\nError during ingestion: {e}", file=sys.stderr)
        return 1


def cmd_search(args: argparse.Namespace) -> int:
    """Search for documents matching a query."""
    from src.rag.models import UserContext

    query = args.query
    k = args.k
    use_hierarchy = args.hierarchical

    # Build user context from CLI args
    user_context = None
    if args.roles:
        roles = args.roles.split(",")
        user_context = UserContext(user_roles=roles)

    print("=" * 60)
    print(f"Query: {query}")
    print(f"Top-k: {k}")
    print(f"Mode: {'Hierarchical' if use_hierarchy else 'Flat'}")
    if user_context:
        print(f"User Context: roles={user_context.user_roles}")
    else:
        print("User Context: NONE (no RBAC filtering)")
    print("=" * 60)

    try:
        if use_hierarchy:
            include_full = args.full if hasattr(args, "full") else False
            results = retrieve_hierarchical_with_debug(
                query, k * 3, return_parents=k, include_full_parent=include_full, user_context=user_context
            )
            print(f"\nFound {results['num_results']} parent chunks:\n")

            for r in results["results"]:
                print(f"[{r['rank']}] Aggregate Score: {r['aggregate_score']:.4f}")
                print("    Parent:")
                print(f"    Parent Chunk ID: {r['parent']['chunk_id']}")
                print(f"    Section Header: {r['parent']['section_header']}")
                print(f"    Doc: {r['parent']['doc_id']} ({r['parent']['classification']})")
                print(f"    Full text: {r['parent']['full_text_length']} chars")
                print()
                # Show preview (Stage 1)
                print("    [Preview]")
                for line in r["parent"]["preview"].split("\n")[:10]:
                    print(f"    {line}")
                print()
                # Show full text if requested (Stage 2)
                if include_full and r["parent"]["full_text"]:
                    print("    [Full Context]")
                    for line in r["parent"]["full_text"].split("\n"):
                        print(f"    {line}")
                    print()
                print("    Matched children:")
                for child in r["matched_children"]:
                    print(f"      - [{child['score']:.4f}] {child['section_header']}")
                    # Show child text (truncated for display)
                    print("        [Preview]")
                    child_preview = (
                        child["text"][:200] + "..." if len(child["text"]) > 200 else child["text"]
                    )
                    for line in child_preview.split("\n")[:5]:
                        print(f"        {line}")
                print()
        else:
            results = retrieve_with_debug(query, k, user_context=user_context)
            print(f"\nFound {results['num_results']} results:\n")

            for r in results["results"]:
                print(f"[{r['rank']}] Score: {r['score']:.4f}")
                print(f"    Chunk: {r['chunk_id']}")
                print(f"    Doc: {r['doc_id']} ({r['classification']})")
                print("    [Preview]")
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

    print("\nConfiguration:")
    print(f"  Embedding model: {settings.embedding_model}")
    print(f"  Hierarchy enabled: {settings.hierarchy_enabled}")
    print(f"  Flat chunk size: {settings.chunk_size} chars")
    print(f"  Flat chunk overlap: {settings.chunk_overlap} chars")
    print(f"  Parent chunk size: {settings.parent_chunk_size} chars")
    print(f"  Child chunk size: {settings.child_chunk_size} chars")
    print(f"  Child chunk overlap: {settings.child_chunk_overlap} chars")
    print(f"  Default top-k: {settings.default_top_k}")
    print(f"  Max top-k: {settings.max_top_k}")

    print("\nPaths:")
    print(f"  Docs directory: {settings.docs_dir}")
    print(f"  Index directory: {settings.index_dir}")

    # Check if index exists
    print("\nIndex Status:")
    if settings.faiss_index_path.exists():
        print(f"  FAISS index: {settings.faiss_index_path} (exists)")
    else:
        print("  FAISS index: Not created (run 'ingest' first)")

    if settings.docstore_path.exists():
        with open(settings.docstore_path) as f:
            docstore = json.load(f)
        version = docstore.get("version", "1.0")
        print(f"  Docstore: {settings.docstore_path}")
        print(f"    Version: {version}")
        if version == "2.0":
            print("    Mode: Hierarchical")
            print(f"    Total parents: {docstore['metadata'].get('total_parents', 0)}")
            print(f"    Total children: {docstore['metadata'].get('total_children', 0)}")
        else:
            print("    Mode: Flat")
            print(f"    Total chunks: {docstore['metadata'].get('total_chunks', 0)}")
    else:
        print("  Docstore: Not created (run 'ingest' first)")

    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Ask a question using RAG."""
    from src.rag.models import UserContext

    question = args.question
    k = args.k
    use_hierarchical = args.hierarchical

    # Build user context from CLI args
    user_context = None
    if args.roles:
        roles = args.roles.split(",")
        user_context = UserContext(user_roles=roles)

    print("=" * 60)
    print(f"Question: {question}")
    print(f"Retrieval mode: {'Hierarchical' if use_hierarchical else 'Flat'}")
    print(f"Top-k: {k}")
    if user_context:
        print(f"User Context: roles={user_context.user_roles}")
    else:
        print("User Context: NONE (no RBAC filtering)")
    print("=" * 60)

    try:
        result = generate_with_retrieval(
            query=question,
            k=k,
            use_hierarchical=use_hierarchical,
            user_context=user_context,
        )

        if args.json:
            print("\n--- JSON Output ---")
            output = {
                "answer": result.answer,
                "citations": [
                    {
                        "doc_id": c.doc_id,
                        "chunk_id": c.chunk_id,
                        "text_snippet": c.text_snippet,
                    }
                    for c in result.citations
                ],
                "confidence": result.confidence,
                "policy_flags": [f.value for f in result.policy_flags],
            }
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            print("\n--- Answer ---")
            print(result.answer)
            print(f"\nConfidence: {result.confidence:.2f}")

            if result.citations:
                print("\n--- Citations ---")
                for i, c in enumerate(result.citations, 1):
                    print(f"[{i}] {c.doc_id} / {c.chunk_id}")
                    if len(c.text_snippet) > 100:
                        snippet = c.text_snippet[:100] + "..."
                    else:
                        snippet = c.text_snippet
                    print(f"    \"{snippet}\"")

            if result.policy_flags:
                print("\n--- Policy Flags ---")
                for flag in result.policy_flags:
                    print(f"  - {flag.value}")

        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("Run 'python -m src.app ingest' first to create the index.", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"\nConfiguration Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nError during generation: {e}", file=sys.stderr)
        return 1


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
    ingest_parser.add_argument(
        "--flat",
        action="store_true",
        help="Use flat chunking instead of hierarchical chunking",
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
    search_parser.add_argument(
        "--hierarchical",
        "-H",
        action="store_true",
        help="Use hierarchical retrieval (returns parent chunks with matched children)",
    )
    search_parser.add_argument(
        "--full",
        "-F",
        action="store_true",
        help="Include full parent context (Stage 2) in hierarchical results",
    )

    # Info command
    subparsers.add_parser("info", help="Show system information")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question using RAG")
    ask_parser.add_argument("question", type=str, help="Question to ask")
    ask_parser.add_argument(
        "-k",
        type=int,
        default=settings.default_top_k,
        help=f"Number of chunks to retrieve (default: {settings.default_top_k})",
    )
    ask_parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )
    ask_parser.add_argument(
        "--hierarchical",
        "-H",
        action="store_true",
        help="Use hierarchical retrieval",
    )
    ask_parser.add_argument(
        "--roles",
        type=str,
        help="Comma-separated list of user roles (e.g., 'employee,contractor')",
    )

    # Add RBAC arguments to search parser
    search_parser.add_argument(
        "--roles",
        type=str,
        help="Comma-separated list of user roles (e.g., 'employee,contractor')",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        return cmd_ingest(args)
    elif args.command == "search":
        return cmd_search(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "ask":
        return cmd_ask(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
