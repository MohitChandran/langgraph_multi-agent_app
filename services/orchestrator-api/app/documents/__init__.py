"""HTTP side of documents: upload, list, download, delete (plan 02, section 7).

Modules:
    files   - file-type detection by magic bytes, and where a document's files live on disk
    routes  - FastAPI endpoints under /api/documents and /api/sessions/{id}/documents
"""
