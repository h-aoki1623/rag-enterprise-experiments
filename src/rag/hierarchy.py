"""Hierarchical chunking implementation for improved context preservation."""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .models import ChunkLevel, Document, HierarchicalChunk


def extract_sections(content: str) -> list[tuple[str, str, int, int]]:
    """
    Extract sections based on markdown level-2 headers (##).

    Args:
        content: Full document content.

    Returns:
        List of (header, section_content, start_pos, end_pos) tuples.
    """
    # Pattern to match ## headers and capture content until next ## or end
    pattern = r"(^##\s+.+)(?:\n)([\s\S]*?)(?=\n##\s+|\Z)"

    sections = []
    for match in re.finditer(pattern, content, re.MULTILINE):
        header_line = match.group(1).strip()
        # Extract just the header text without ##
        header = re.sub(r"^##\s+", "", header_line)
        section_content = match.group(0)  # Full match including header
        sections.append((header, section_content, match.start(), match.end()))

    # If no ## headers found, check for # headers
    if not sections:
        pattern = r"(^#\s+.+)(?:\n)([\s\S]*?)(?=\n#\s+|\Z)"
        for match in re.finditer(pattern, content, re.MULTILINE):
            header_line = match.group(1).strip()
            header = re.sub(r"^#\s+", "", header_line)
            section_content = match.group(0)
            sections.append((header, section_content, match.start(), match.end()))

    # If still no sections found, treat whole document as one section
    if not sections:
        sections = [("Document", content, 0, len(content))]

    return sections


def extract_h3_subsections(content: str) -> list[tuple[str, str]]:
    """
    Extract H3 subsections from content.

    Args:
        content: Section content (may contain ### headers).

    Returns:
        List of (subsection_header, subsection_content) tuples.
        If no H3 headers found, returns the entire content as one subsection.
    """
    # Pattern to match ### headers and capture content until next ### or ## or end
    pattern = r"(^###\s+.+)(?:\n)([\s\S]*?)(?=\n###\s+|\n##\s+|\Z)"

    subsections = []
    last_end = 0

    for match in re.finditer(pattern, content, re.MULTILINE):
        # Capture any content before the first ### (preamble)
        if not subsections and match.start() > 0:
            preamble = content[:match.start()].strip()
            if preamble:
                subsections.append(("Preamble", preamble))

        header_line = match.group(1).strip()
        header = re.sub(r"^###\s+", "", header_line)
        subsection_content = match.group(0)
        subsections.append((header, subsection_content))
        last_end = match.end()

    # If no ### headers found, return the entire content
    if not subsections:
        return [("Full Section", content)]

    # Capture any trailing content after the last ###
    if last_end < len(content):
        trailing = content[last_end:].strip()
        if trailing:
            subsections.append(("Trailing", trailing))

    return subsections


def create_parent_chunks(document: Document) -> list[HierarchicalChunk]:
    """
    Create parent chunks from document sections.

    Parent chunks keep the full section content without splitting.

    Args:
        document: Document to process.

    Returns:
        List of parent HierarchicalChunk objects.
    """
    sections = extract_sections(document.content)

    parents = []
    for parent_idx, (header, content, start, end) in enumerate(sections):
        # Keep full section content - no splitting for parent chunks
        parent_id = f"{document.metadata.doc_id}-parent-{parent_idx:03d}"
        parents.append(
            HierarchicalChunk(
                chunk_id=parent_id,
                text=content,
                doc_id=document.metadata.doc_id,
                metadata=document.metadata,
                level=ChunkLevel.PARENT,
                parent_id=None,
                children_ids=[],
                section_header=header,
            )
        )

    return parents


def create_child_chunks(parent: HierarchicalChunk) -> list[HierarchicalChunk]:
    """
    Create child chunks from a parent chunk based on H3 sections.

    Child chunks are created per H3 subsection. If a subsection exceeds
    the child_chunk_size, it will be further split.

    Args:
        parent: Parent chunk to split into children.

    Returns:
        List of child HierarchicalChunk objects.
    """
    # Extract H3 subsections
    subsections = extract_h3_subsections(parent.text)

    # Splitter for subsections that exceed child_chunk_size
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", ". ", " ", ""],
    )

    children = []
    child_idx = 0

    for subsection_header, subsection_content in subsections:
        # If subsection exceeds limit, split it
        if len(subsection_content) > settings.child_chunk_size:
            texts = child_splitter.split_text(subsection_content)
            for sub_idx, text in enumerate(texts):
                child_id = f"{parent.chunk_id.replace('-parent-', '-child-')}-{child_idx:03d}"
                children.append(
                    HierarchicalChunk(
                        chunk_id=child_id,
                        text=text,
                        doc_id=parent.doc_id,
                        metadata=parent.metadata,
                        level=ChunkLevel.CHILD,
                        parent_id=parent.chunk_id,
                        children_ids=[],
                        section_header=f"{parent.section_header} > {subsection_header}"
                        if subsection_header != "Full Section"
                        else parent.section_header,
                    )
                )
                child_idx += 1
        else:
            # Keep subsection as-is
            child_id = f"{parent.chunk_id.replace('-parent-', '-child-')}-{child_idx:03d}"
            children.append(
                HierarchicalChunk(
                    chunk_id=child_id,
                    text=subsection_content,
                    doc_id=parent.doc_id,
                    metadata=parent.metadata,
                    level=ChunkLevel.CHILD,
                    parent_id=parent.chunk_id,
                    children_ids=[],
                    section_header=f"{parent.section_header} > {subsection_header}"
                    if subsection_header != "Full Section"
                    else parent.section_header,
                )
            )
            child_idx += 1

    return children


def chunk_document_hierarchical(
    document: Document,
) -> tuple[list[HierarchicalChunk], list[HierarchicalChunk]]:
    """
    Create hierarchical chunks from a document.

    This function creates a two-level hierarchy:
    - Parent chunks: Full H2 sections (not split)
    - Child chunks: H3 subsections (split only if exceeding size limit)

    Args:
        document: Document to chunk.

    Returns:
        Tuple of (parent_chunks, child_chunks).
    """
    parents = create_parent_chunks(document)
    all_children = []

    for parent in parents:
        children = create_child_chunks(parent)
        # Update parent with children IDs
        parent.children_ids = [c.chunk_id for c in children]
        all_children.extend(children)

    return parents, all_children


def chunk_documents_hierarchical(
    documents: list[Document],
) -> tuple[list[HierarchicalChunk], list[HierarchicalChunk]]:
    """
    Split multiple documents into hierarchical chunks.

    Args:
        documents: List of documents to chunk.

    Returns:
        Tuple of (all_parents, all_children).
    """
    all_parents = []
    all_children = []

    for doc in documents:
        parents, children = chunk_document_hierarchical(doc)
        all_parents.extend(parents)
        all_children.extend(children)

    print(f"Created {len(all_parents)} parent chunks and {len(all_children)} child chunks")
    return all_parents, all_children
