"""Document Q&A agent (RAG): upload, parse, chunk, embed, retrieve, answer with citations.
Plan: plannings/02-doc-qa-rag-agent.md.

Modules (stage in brackets):
    store       - DocumentStore: documents + chunks + vectors in Postgres (pgvector), schema `app`  [R1, R5, R6]
    models      - our own types (Element, Chunk, ...) so nothing outside parsing.py touches Docling  [R2-R7]
    parsing     - Docling: file -> headings, paragraphs, tables, pictures                           [R2]
    vision      - which pictures to describe, and the Groq vision call that describes them          [R3]
    chunking    - structure-aware chunks with a token budget and a heading-path prefix              [R4]
    embeddings  - the embedding model and the cross-encoder reranker, loaded once                   [R5]
    ingest      - the pipeline for one document, stage by stage, with status updates                [R5]
    worker      - background loop that takes queued documents from Postgres                          [R5]
    retrieval   - Reciprocal Rank Fusion of vector + keyword results                                [R6]
    prompts     - prompt templates (query rewrite, answer, vision)                                 [R3, R7]
    graph       - DocAgent: the LangGraph question-answering workflow                               [R7]
"""
