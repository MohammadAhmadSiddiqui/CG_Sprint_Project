import os
import re
import struct
import argparse
import pickle
import math

# 1.  PKL  (LangChain InMemoryDocstore)

def _extract_strings_from_pkl(path: str):
    """
    Fallback: if langchain is not installed, pull readable text directly
    from the raw bytes with a regex — works for InMemoryDocstore pickles.
    """
    with open(path, "rb") as f:
        raw = f.read()
    # grab all printable ASCII runs ≥ 8 chars
    return re.findall(rb"[\x20-\x7e]{8,}", raw)


def read_pkl(path: str) -> list[dict]:
    """
    Return a list of document dicts: {id, page_content, metadata}.
    Tries real unpickling first; falls back to regex extraction.
    """
    docs = []

    # ── Attempt 1: proper unpickle (needs langchain installed) ────────────────
    try:
        with open(path, "rb") as f:
            store = pickle.load(f)          
        doc_dict = store._dict             
        for uid, doc in doc_dict.items():
            docs.append({
                "id":           uid,
                "page_content": doc.page_content,
                "metadata":     doc.metadata if doc.metadata else {},
            })
        return docs
    except Exception:
        pass  # langchain not available — fall through

    # ── Attempt 2: regex string extraction ───────────────────────────────────
    strings   = _extract_strings_from_pkl(path)
    text_blob = b"\n".join(strings).decode("ascii", errors="replace")

    # Split on UUID-like boundaries (36-char hex UUIDs stored in the pickle)
    uuid_pat = re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        re.I,
    )
    uuids = uuid_pat.findall(text_blob)

    # Collect meaningful text blocks (skip internal class names)
    _skip = {
        "langchain_community", "langchain_core", "InMemoryDocstore",
        "page_content", "__dict__", "__pydantic", "documents.base", "Document",
    }
    blocks = [
        s for s in text_blob.splitlines()
        if len(s) > 30 and not any(sk in s for sk in _skip)
    ]

    # Pair UUIDs with the text blocks that follow them
    for i, uid in enumerate(uuids):
        start = text_blob.find(uid)
        end   = text_blob.find(uuids[i + 1]) if i + 1 < len(uuids) else len(text_blob)
        chunk = text_blob[start + len(uid): end].strip()
        # Remove internal pickle noise lines
        clean_lines = [
            ln for ln in chunk.splitlines()
            if len(ln) > 20 and not any(sk in ln for sk in _skip)
        ]
        docs.append({
            "id":           uid,
            "page_content": "\n".join(clean_lines).strip(),
            "metadata":     {},
        })

    return docs


# 2.  FAISS  (IndexFlatL2 binary format)
# FAISS IndexFlatL2 binary layout (little-endian):
#   bytes  0– 3   magic  (0x494658 + index-type byte)
#   bytes  4– 7   version / reserved
#   bytes  8–11   ntotal  (lower 32 bits of int64)
#   bytes 12–15   ntotal  (upper 32 bits — usually 0)
#   bytes 16–19   d  (lower 32 bits of int64)
#   bytes 20–23   d  (upper 32 bits — usually 0)
#   bytes 24–44   remaining header (metric / is_trained flags …)
#   bytes 45–end  float32 vectors, row-major (ntotal × d × 4 bytes)

FAISS_HEADER_SIZE = 45
FLOAT_SIZE        = 4  # bytes per float32


def _infer_params(file_size: int) -> tuple[int, int] | None:
    """
    Given the file size, infer (ntotal, d) by trying common embedding dims.
    Returns (ntotal, d) or None.
    """
    data_bytes = file_size - FAISS_HEADER_SIZE
    if data_bytes <= 0 or data_bytes % FLOAT_SIZE != 0:
        return None
    total_floats = data_bytes // FLOAT_SIZE
    for d in [384, 512, 768, 1024, 1536, 3072]:
        if total_floats % d == 0:
            return total_floats // d, d
    return None


def read_faiss(path: str, show_vectors: bool = False) -> dict:
    """
    Parse a FAISS IndexFlatL2 file and return a human-readable dict.
    """
    with open(path, "rb") as f:
        raw = f.read()

    file_size = len(raw)
    magic     = raw[:4].hex()
    ntotal    = struct.unpack("<q", raw[8:16])[0]   # int64 stored at offset 8
    d         = struct.unpack("<q", raw[16:24])[0]  # int64 stored at offset 16

    # If header values are non-sensical, infer from file size
    if not (0 < ntotal < 100_000 and 0 < d < 10_000):
        inferred = _infer_params(file_size)
        if inferred:
            ntotal, d = inferred

    vectors = []
    for i in range(ntotal):
        offset = FAISS_HEADER_SIZE + i * d * FLOAT_SIZE
        vec    = struct.unpack(f"<{d}f", raw[offset: offset + d * FLOAT_SIZE])
        magnitude = math.sqrt(sum(x * x for x in vec))
        entry = {
            "index":     i,
            "magnitude": round(magnitude, 6),
        }
        if show_vectors:
            entry["vector_preview"] = list(vec[:10])   # first 10 dims
            entry["vector_full"]    = list(vec)
        vectors.append(entry)

    return {
        "magic":           magic,
        "num_vectors":     ntotal,
        "embedding_dim":   d,
        "file_size_bytes": file_size,
        "vectors":         vectors,
    }


# 3.  Pretty printing

def _divider(char="─", width=70):
    print(char * width)


def print_pkl_report(docs: list[dict]):
    print()
    _divider("═")
    print("  index.pkl  —  Document Chunks (LangChain InMemoryDocstore)")
    _divider("═")
    print(f"  Total chunks stored : {len(docs)}")
    _divider()

    for i, doc in enumerate(docs, 1):
        print(f"\n  Chunk #{i}")
        print(f"  UUID     : {doc['id']}")
        if doc["metadata"]:
            print(f"  Metadata : {doc['metadata']}")
        print(f"  Content  :")
        for line in doc["page_content"].splitlines():
            print(f"    {line}")
        _divider()


def print_faiss_report(info: dict, show_vectors: bool):
    print()
    _divider("═")
    print("   index.faiss  —  FAISS IndexFlatL2 Embedding Index")
    _divider("═")
    print(f"  Magic (hex)      : {info['magic']}")
    print(f"  Number of vectors: {info['num_vectors']}")
    print(f"  Embedding dim    : {info['embedding_dim']}")
    print(f"  File size        : {info['file_size_bytes']:,} bytes")
    _divider()

    for vec in info["vectors"]:
        print(f"\n  Vector #{vec['index']}")
        print(f"    L2 magnitude : {vec['magnitude']:.6f}  "
              f"({'≈ 1.0 — unit-normalised ✓' if abs(vec['magnitude'] - 1.0) < 0.05 else 'not unit-normalised'})")
        if show_vectors:
            preview = ", ".join(f"{v:.6f}" for v in vec["vector_preview"])
            print(f"    First 10 dims: [{preview}, …]")
        _divider()

    if not show_vectors:
        print("\n  Tip: run with --show-vectors to print actual embedding values.\n")


# 4.  Main

def main():
    parser = argparse.ArgumentParser(
        description="Human-readable viewer for FAISS index + LangChain PKL docstore"
    )
    parser.add_argument(
        "--store-dir",
        default="vector_store",
        help="Path to the folder containing index.faiss and index.pkl "
             "(default: ./vector_store)",
    )
    parser.add_argument(
        "--show-vectors",
        action="store_true",
        help="Also print the first 10 dimensions of each embedding vector",
    )
    args = parser.parse_args()

    pkl_path   = os.path.join(args.store_dir, "index.pkl")
    faiss_path = os.path.join(args.store_dir, "index.faiss")

    if os.path.exists(pkl_path):
        docs = read_pkl(pkl_path)
        print_pkl_report(docs)
    else:
        print(f"[WARN] PKL file not found: {pkl_path}")

    if os.path.exists(faiss_path):
        info = read_faiss(faiss_path, show_vectors=args.show_vectors)
        print_faiss_report(info, show_vectors=args.show_vectors)
    else:
        print(f"[WARN] FAISS file not found: {faiss_path}")


if __name__ == "__main__":
    main()
