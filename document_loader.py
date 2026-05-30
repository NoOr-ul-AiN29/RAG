# document_loader.py
# ---------------------------------------------------------------------------
# PURPOSE:
#   1. Extract raw text from an uploaded PDF file using pypdf.
#   2. Split that text into overlapping chunks so FAISS can index them.
#
# WHY CHUNKS?
#   Large documents don't fit in a single embedding. Splitting into smaller
#   pieces lets us find the *most relevant* section for any given question.
# ---------------------------------------------------------------------------

from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    """
    Read every page of the PDF at `file_path` and return all text as one
    big string.

    Args:
        file_path: Absolute or relative path to the PDF on disk.

    Returns:
        A single string containing the text of the entire document.

    Raises:
        FileNotFoundError: If the PDF does not exist at the given path.
        Exception:         For any pypdf reading errors.
    """
    reader = PdfReader(file_path)
    all_text = []

    for page_number, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text:                        # some pages may be images
                all_text.append(page_text)
        except Exception as e:
            # Skip pages that can't be read instead of crashing entirely
            print(f"[document_loader] Warning: Could not read page "
                  f"{page_number + 1} — {e}")

    if not all_text:
        raise ValueError(
            "No readable text found in the PDF. "
            "The file may contain only scanned images."
        )

    return "\n".join(all_text)


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[str]:
    """
    Split `text` into chunks of `chunk_size` characters with `overlap`
    characters repeated between consecutive chunks.

    Overlap ensures that a sentence split across two chunks is still
    fully captured in at least one of them.

    Args:
        text:       The full document text to split.
        chunk_size: Maximum number of characters per chunk.
        overlap:    Number of characters to repeat from the previous chunk.

    Returns:
        A list of text chunk strings.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        # Don't cut in the middle of a word — walk back to the last space
        if end < text_length:
            last_space = text.rfind(" ", start, end)
            if last_space != -1:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:                                # ignore empty chunks
            chunks.append(chunk)

        # Move forward but keep `overlap` characters for context continuity
        start = end - overlap

    print(f"[document_loader] Split document into {len(chunks)} chunks "
          f"(chunk_size={chunk_size}, overlap={overlap})")
    return chunks