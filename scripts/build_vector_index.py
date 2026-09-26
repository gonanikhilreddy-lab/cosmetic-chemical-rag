import argparse

from src.retrieval.vector_store import build_chemical_index


parser = argparse.ArgumentParser(description="Build the local Qdrant chemical-name vector index using Ollama embeddings.")
parser.add_argument("--force", action="store_true", help="Rebuild the vector index from dataset chemical names")
args = parser.parse_args()

count = build_chemical_index(force=args.force)
print(f"Indexed {count} unique chemical names in the local Qdrant store.")