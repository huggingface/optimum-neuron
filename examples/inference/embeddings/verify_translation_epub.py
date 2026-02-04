#!/usr/bin/env python3
"""
verify_epub_translation_ebooklib.py  EPUB_A  EPUB_B  [options]

Use ebooklib to extract chapter text from two EPUBs, obtain paragraph embeddings
from a cloud OpenAI-compatible service, and verify pairwise similarity
inside the chosen chapter(s).
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ebooklib
import requests
import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup
from ebooklib import epub
from openai import OpenAI
from torch import Tensor


# ---------- utils -------------------------------------------------
def die(msg: str, code: int = 1) -> None:
    sys.stderr.write(f"Error: {msg}\n")
    sys.exit(code)


def similarity(a: Tensor, b: Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    return (a @ b).item()


def fetch_file(url: str, tmpdir: Path) -> Path:
    """Download url into tmpdir and return local Path."""
    try:
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        fname = "downloaded.epub"
        local_path = tmpdir / fname
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_path
    except Exception as e:
        die(f"download failed: {e}")


# ---------- chapter extractor ---------------------------


def extract_chapters(book: epub.EpubBook) -> Dict[str, list[str]]:
    chapters = {}
    chapters_titles = [item.title for item in book.toc]
    documents = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    # Iterate over documents, extract text and group by chapter titles
    # Note that a document might contain multiple chapters or parts of chapters
    current_title: Optional[str] = None
    current_chapter: List[str] = []
    for doc in documents:
        soup = BeautifulSoup(doc.get_body_content(), "html.parser")
        for node in soup.find_all(["h2", "p"]):
            if node.name == "h2":
                heading = node.get_text(" ", strip=True)
                if heading in chapters_titles:
                    if len(current_chapter) > 0 and current_title:
                        chapters[current_title] = current_chapter
                    current_title = heading
                    current_chapter = []
                continue
            if node.name == "p" and current_title:
                text = node.get_text(" ", strip=True)
                if text and any(c.isalnum() for c in text):
                    current_chapter.append(text)
    return chapters


# ---------- embedding utility ---------------------------
def embed_texts(client: OpenAI, texts: List[str], model: str) -> Tensor:
    """
    Embed texts using OpenAI-compatible API with task instruction.
    Task: retrieve translation from provided documents.
    Returns normalized embeddings as a single torch tensor (N, D).
    """
    task = "Given a text, retrieve its translation from the provided documents."
    instructed_texts = [f"Instruct: {task}\nQuery: {text}" for text in texts]

    resp = client.embeddings.create(input=instructed_texts, model=model)
    embeddings = [torch.tensor(d.embedding, dtype=torch.float32) for d in resp.data]
    embeddings = torch.stack(embeddings)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


def similarity_matrix(emb_a: Tensor, emb_b: Tensor) -> Tensor:
    """Compute similarity matrix between two sets of embeddings.

    Args:
        emb_a: Tensor of shape (N, D)
        emb_b: Tensor of shape (M, D)

    Returns:
        Tensor of shape (N, M) with similarity scores
    """
    return emb_a @ emb_b.T


# ---------- similarity logic -------------------------------------
def find_first_match(emb_a: Tensor, emb_b: Tensor, threshold: float) -> Tuple[Optional[int], Optional[int]]:
    scores = similarity_matrix(emb_a, emb_b)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            if scores[i, j] >= threshold:
                return i, j
    return None, None


def check_chapter(para_a: List[str], para_b: List[str], emb_a: Tensor, emb_b: Tensor, threshold: float) -> None:
    """Check paragraph correspondence between two chapters using similarity matrix."""

    # Compute similarity matrix for all paragraph pairs
    scores = similarity_matrix(emb_a, emb_b)

    # Build correspondence: for each paragraph in A, find best match in B
    correspondence = {}  # maps para_a_idx -> (para_b_idx, similarity, status)
    matched_b = set()  # track which B paragraphs are matched

    for i in range(emb_a.shape[0]):
        best_idx = torch.argmax(scores[i]).item()
        best_sim = scores[i, best_idx].item()

        if best_sim >= threshold:
            status = "matched"
            matched_b.add(best_idx)
        else:
            status = "unmatched"
            best_idx = None

        correspondence[i] = (best_idx, best_sim, status)

    # Find unmatched paragraphs in B
    unmatched_b = [j for j in range(emb_b.shape[0]) if j not in matched_b]

    # Display results
    matched = sum(1 for _, _, status in correspondence.values() if status == "matched")
    unmatched_a = len(correspondence) - matched

    print("  Paragraph correspondence:")
    print(f"    A: {len(para_a)} paragraphs, {matched} matched, {unmatched_a} unmatched")
    print(f"    B: {len(para_b)} paragraphs, {matched} matched, {len(unmatched_b)} unmatched")
    print()

    # Show detailed matches
    print("  Matched paragraphs:")
    for i, (j, sim, status) in correspondence.items():
        if status == "matched":
            para_a_preview = para_a[i][:60] + "..." if len(para_a[i]) > 60 else para_a[i]
            para_b_preview = para_b[j][:60] + "..." if len(para_b[j]) > 60 else para_b[j]
            print(f"    A[{i:3d}] ↔ B[{j:3d}] sim={sim:.3f}")
            print(f"      A: {para_a_preview}")
            print(f"      B: {para_b_preview}")

    if unmatched_a > 0:
        print("\n  Unmatched paragraphs in A:")
        for i, (_, sim, status) in correspondence.items():
            if status == "unmatched":
                para_preview = para_a[i][:80] + "..." if len(para_a[i]) > 80 else para_a[i]
                print(f"    A[{i:3d}] (best sim={sim:.3f}): {para_preview}")

    if unmatched_b:
        print("\n  Unmatched paragraphs in B:")
        for j in unmatched_b:
            para_preview = para_b[j][:80] + "..." if len(para_b[j]) > 80 else para_b[j]
            print(f"    B[{j:3d}]: {para_preview}")


# ---------- main ------------------------------------------------
def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    ap = argparse.ArgumentParser(description="Verify translation similarity inside EPUB chapters (ebooklib).")
    ap.add_argument("epub_a")
    ap.add_argument("epub_b")
    ap.add_argument("--api-key", default="EMPTY", help="API key for embedding service")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="OpenAI-compatible endpoint")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args(argv)

    tmpdir = Path(tempfile.mkdtemp(prefix="epub_"))
    try:
        # obtain local files
        if os.path.exists(args.epub_a):
            path_a = Path(args.epub_a)
        else:
            dir_a = os.path.join(tmpdir, "a")
            os.makedirs(dir_a, exist_ok=True)
            path_a = fetch_file(args.epub_a, Path(dir_a))
        if os.path.exists(args.epub_b):
            path_b = Path(args.epub_b)
        else:
            dir_b = os.path.join(tmpdir, "b")
            os.makedirs(dir_b, exist_ok=True)
            path_b = fetch_file(args.epub_b, Path(dir_b))

        book_a = epub.read_epub(str(path_a))
        chapters_a = extract_chapters(book_a)
        book_b = epub.read_epub(str(path_b))
        chapters_b = extract_chapters(book_b)

        if not chapters_a or not chapters_b:
            die("no chapters found")

        # Get chapter titles and their embeddings for each book
        titles_a = list(chapters_a.keys())
        titles_b = list(chapters_b.keys())

        print(f"Book A: {len(titles_a)} chapters")
        print(f"Book B: {len(titles_b)} chapters")

        client = OpenAI(base_url=args.base_url, api_key=args.api_key)

        emb_titles_a = embed_texts(client, titles_a, args.model)
        emb_titles_b = embed_texts(client, titles_b, args.model)

        # Build correspondence table: find best matching chapters
        correspondence = {}  # maps title_a -> (title_b, similarity)
        scores = similarity_matrix(emb_titles_a, emb_titles_b)
        for i, title_a in enumerate(titles_a):
            best_idx = torch.argmax(scores[i]).item()
            best_match = titles_b[best_idx]
            best_sim = scores[i, best_idx].item()
            correspondence[title_a] = (best_match, best_sim)

        # Print correspondence table
        print("\nChapter Correspondence:")
        for title_a, (title_b, sim) in correspondence.items():
            print(f"  {title_a:40s} <-> {title_b:40s} (sim={sim:.3f})")

        # Process all chapters
        for title_a in titles_a:
            if title_a not in correspondence:
                continue
            title_b, sim = correspondence[title_a]
            if title_b not in chapters_b:
                continue

            para_a = chapters_a[title_a]
            para_b = chapters_b[title_b]
            print(f"\nChapter: {title_a} <-> {title_b}  ({len(para_a)} vs {len(para_b)} paragraphs, sim={sim:.3f})")

            emb_a = embed_texts(client, para_a, args.model)
            emb_b = embed_texts(client, para_b, args.model)
            check_chapter(para_a, para_b, emb_a, emb_b, args.threshold)

    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
