/**
 * The row of document "chips" above the composer: one per document in this chat, with its
 * live processing status and a remove (x) button (plan 02, section 8).
 *
 * The status comes from the server's status/stage/progress fields, polled by useDocuments
 * while anything is still processing.
 */
import { CloseIcon, FileIcon } from "./icons.jsx";

const STAGE_LABELS = {
  parsing: "Reading the document",
  describing_pictures: "Describing pictures",
  chunking: "Splitting into chunks",
  embedding: "Indexing",
};

/**
 * Human-readable status of one document, e.g. "Describing pictures 2/3" or "Ready · 5 pages".
 *
 * @param {object} doc - a Document from the API.
 * @returns {string}
 */
export function statusLabel(doc) {
  if (doc.status === "ready") return `Ready · ${doc.page_count} page${doc.page_count === 1 ? "" : "s"}`;
  if (doc.status === "failed") return `Failed: ${doc.error ?? "unknown error"}`;
  if (doc.status === "queued") return "Queued";
  const stage = STAGE_LABELS[doc.stage] ?? "Processing";
  if (doc.stage === "describing_pictures" && doc.progress) return `${stage} ${doc.progress.replace("pictures ", "")}`;
  return `${stage}…`;
}

/**
 * @param {{documents: Array<object>, uploading: number, onDelete: (doc: object) => void}} props
 *   uploading = files still being sent to the server (shown as one extra chip).
 */
export default function DocumentChips({ documents, uploading, onDelete }) {
  if (documents.length === 0 && uploading === 0) return null;
  return (
    <div className="doc-chips" aria-label="Documents in this chat">
      {documents.map((doc) => (
        <div key={doc.id} className={`doc-chip ${doc.status}`} title={`${doc.filename}: ${statusLabel(doc)}`}>
          {doc.status === "queued" || doc.status === "processing" ? <span className="mini-spinner" /> : <FileIcon />}
          <span className="doc-name">{doc.filename}</span>
          <span className="doc-status">{statusLabel(doc)}</span>
          <button type="button" className="doc-remove" onClick={() => onDelete(doc)} aria-label={`Remove ${doc.filename}`}>
            <CloseIcon />
          </button>
        </div>
      ))}
      {uploading > 0 && (
        <div className="doc-chip processing">
          <span className="mini-spinner" />
          <span className="doc-status">
            Uploading {uploading} file{uploading === 1 ? "" : "s"}…
          </span>
        </div>
      )}
    </div>
  );
}
