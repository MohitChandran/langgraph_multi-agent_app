/**
 * How an answer from the Doc QA agent is shown (plan 02, section 8):
 *   - the answer, with every [n] turned into a small clickable badge;
 *   - a Sources list: file, page, section, a snippet, a thumbnail for figures, "Open" at the page;
 *   - a "Show retrieval" panel: how every candidate chunk ranked in each search step.
 *
 * Everything comes from the saved message: content = the answer text, payload.result = the
 * DocAgentResult (citations, retrieval trace, steps).
 */
import { useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import AgentFlow from "./AgentFlow.jsx";
import { ChevronIcon, ExternalLinkIcon } from "./icons.jsx";

// "[2]" -> "[2](#cite-2)": a markdown link the custom <a> renderer below turns into a badge.
// (?!\() skips real markdown links like [text](url).
const CITATION = /\[(\d+)\](?!\()/g;

/**
 * @param {{message: object}} props - an assistant message whose payload.agent is "doc".
 */
export default function DocAnswer({ message }) {
  const result = message.payload.result;
  const [highlighted, setHighlighted] = useState(null);

  /** Scroll to source n of THIS message and flash it, so the reader sees where a claim came from. */
  function focusSource(n) {
    document.getElementById(`src-${message.id}-${n}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    setHighlighted(n);
    setTimeout(() => setHighlighted(null), 1600);
  }

  /** Markdown links: citation badges become buttons; any other link opens in a new tab. */
  function Link({ href, children }) {
    if (href?.startsWith("#cite-")) {
      const n = Number(href.slice(6));
      return (
        <button type="button" className="cite" onClick={() => focusSource(n)} aria-label={`Source ${n}`}>
          {n}
        </button>
      );
    }
    return (
      <a href={href} target="_blank" rel="noreferrer">
        {children}
      </a>
    );
  }

  return (
    <>
      <div className="markdown">
        <Markdown remarkPlugins={[remarkGfm]} components={{ a: Link }}>
          {message.content.replace(CITATION, "[$1](#cite-$1)")}
        </Markdown>
      </div>
      <div className="stats">
        <span className={result.grounded ? "chip ok" : "chip warn"}>{result.grounded ? "From documents" : "Not found"}</span>
        {result.citations.length > 0 && (
          <span className="chip">
            {result.citations.length} source{result.citations.length === 1 ? "" : "s"}
          </span>
        )}
        {message.latency_ms != null && <span className="chip">{(message.latency_ms / 1000).toFixed(1)} s</span>}
      </div>
      {result.citations.length > 0 && (
        <Sources citations={result.citations} messageId={message.id} highlighted={highlighted} />
      )}
      {(result.retrieval.length > 0 || result.reasoning?.length > 0) && (
        <RetrievalDetails result={result} payload={message.payload} />
      )}
    </>
  );
}

/**
 * The cited sources, numbered like the badges in the answer.
 *
 * @param {{citations: Array<object>, messageId: number, highlighted: number|null}} props
 */
function Sources({ citations, messageId, highlighted }) {
  return (
    <ol className="sources">
      {citations.map((c) => {
        const pages = c.page_end !== c.page_start ? `pp. ${c.page_start}–${c.page_end}` : `p. ${c.page_start}`;
        return (
          <li key={c.n} id={`src-${messageId}-${c.n}`} className={highlighted === c.n ? "source flash" : "source"}>
            <span className="cite static">{c.n}</span>
            <div className="source-body">
              <div className="source-head">
                <span className="source-file">{c.filename}</span>
                <span className="source-meta">
                  {pages}
                  {c.heading_path && ` · ${c.heading_path}`}
                  {c.chunk_type !== "text" && ` · ${c.chunk_type}`}
                </span>
                <a className="source-open" href={`/api/documents/${c.document_id}/file#page=${c.page_start}`} target="_blank" rel="noreferrer">
                  Open {pages} <ExternalLinkIcon />
                </a>
              </div>
              <div className="source-snippet">{c.snippet}</div>
            </div>
            {c.picture_url && <img className="source-thumb" src={c.picture_url} alt={`Figure on page ${c.page_start}`} loading="lazy" />}
          </li>
        );
      })}
    </ol>
  );
}

/**
 * Collapsible "Show retrieval" panel: how the question was routed and why (AgentFlow), the
 * workflow steps, the search query, and how every candidate chunk ranked in dense search,
 * keyword search, fusion (RRF) and reranking.
 *
 * @param {{result: object, payload: object}} props - the DocAgentResult and the message payload (route).
 */
function RetrievalDetails({ result, payload }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="details">
      <button type="button" className="details-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? "Hide" : "Show"} retrieval
        <ChevronIcon open={open} />
      </button>
      {open && (
        <div className="details-panel">
          <AgentFlow payload={payload} result={result} />
          <div className="details-label">Workflow</div>
          <ol className="steps">
            {result.steps.map((step, index) => (
              <li key={index} className="step">
                {step}
              </li>
            ))}
          </ol>
          <div className="details-label">Search query</div>
          <p className="details-explanation">{result.standalone_query}</p>
          {result.retrieval.length > 0 && (
            <>
              <div className="details-label">Candidates ({result.retrieval.length})</div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Chunk</th>
                      <th>Type</th>
                      <th>Page</th>
                      <th>Dense rank</th>
                      <th>Keyword rank</th>
                      <th>RRF score</th>
                      <th>Rerank score</th>
                      <th>Used</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.retrieval.map((row) => (
                      <tr key={row.chunk_id} className={row.used ? "used" : undefined}>
                        <td className="num">#{row.chunk_id}</td>
                        <td>{row.chunk_type}</td>
                        <td className="num">{row.page_start}</td>
                        <td className="num">{row.dense_rank ?? "—"}</td>
                        <td className="num">{row.keyword_rank ?? "—"}</td>
                        <td className="num">{row.rrf_score.toFixed(4)}</td>
                        <td className="num">{row.rerank_score != null ? row.rerank_score.toFixed(2) : "—"}</td>
                        <td>{row.used ? "✓" : ""}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
