/**
 * How an orchestrator answer is shown (plan 03, section 13). Since plan 03 every chat answer comes from
 * the orchestrator, which may have asked several agents:
 *
 *   - the answer, with citation badges: [D1] (a document passage) and [W1] (a web page) when several
 *     agents answered, or the single agent's own [1], [2] when its answer was passed through (D9);
 *   - the email card: approve / edit / reject while the answer waits, the outcome afterwards;
 *   - chips: one per agent call with its status, why the loop stopped (if it was a limit), the time;
 *   - Sources: the document passages and web pages behind the badges (a badge scrolls to its source);
 *   - "Show agent calls": the flow (AgentFlow), then one collapsed section per call, reusing the
 *     existing views: SQL & rows for the database, the retrieval table for documents.
 *
 * Everything comes from the saved message: content = the answer, payload.result = the OrchestratorResult.
 */
import { useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AGENT_LABELS, STOP_LABELS, callLabel, callStatus } from "../calls.js";
import AgentFlow from "./AgentFlow.jsx";
import { RetrievalBody, SourceItem } from "./DocAnswer.jsx";
import EmailApprovalCard, { EmailOutcome } from "./EmailApprovalCard.jsx";
import { ChevronIcon, ExternalLinkIcon } from "./icons.jsx";
import { DbResultBody } from "./ResultDetails.jsx";

// "[D1]" / "[W2]" / "[3]" -> a markdown link the <a> renderer turns into a badge. (?!\() skips real links.
const CITATION = /\[([DW]?\d+)\](?!\()/g;

/**
 * @param {{message: object, onEmailDecided: (message: object|null) => void}} props
 *   message = an assistant message whose payload.agent is "orchestrator".
 *   onEmailDecided = passed to the approval card (the page swaps in the updated message).
 */
export default function OrchestratorAnswer({ message, onEmailDecided }) {
  const result = message.payload.result;
  const [highlighted, setHighlighted] = useState(null);
  const sources = collectSources(result);
  const labels = new Set(sources.map((source) => source.label));

  /** Scroll to the source with this label in THIS message and flash it. */
  function focusSource(label) {
    document.getElementById(sourceId(message.id, label))?.scrollIntoView({ behavior: "smooth", block: "center" });
    setHighlighted(label);
    setTimeout(() => setHighlighted(null), 1600);
  }

  /** Markdown links: citation badges become buttons; any other link opens in a new tab. */
  function Link({ href, children }) {
    if (href?.startsWith("#cite-")) {
      const label = href.slice(6);
      return (
        <button type="button" className="cite" onClick={() => focusSource(label)} aria-label={`Source ${label}`}>
          {label}
        </button>
      );
    }
    return (
      <a href={href} target="_blank" rel="noreferrer">
        {children}
      </a>
    );
  }

  // Only labels that lead to a real source become badges; anything else stays as written.
  const text = message.content.replace(CITATION, (match, label) => (labels.has(label) ? `[${label}](#cite-${label})` : match));

  return (
    <>
      <div className="markdown">
        <Markdown remarkPlugins={[remarkGfm]} components={{ a: Link }}>
          {text}
        </Markdown>
      </div>
      {result.email &&
        (message.status === "awaiting_approval" ? (
          <EmailApprovalCard messageId={message.id} email={result.email} onDecided={onEmailDecided} />
        ) : (
          <EmailOutcome email={result.email} />
        ))}
      <CallChips result={result} latencyMs={message.latency_ms} />
      {sources.length > 0 && (
        <ol className="sources">
          {sources.map((source) =>
            source.kind === "doc" ? (
              <SourceItem
                key={source.label}
                citation={source.item}
                label={source.label}
                id={sourceId(message.id, source.label)}
                highlighted={highlighted === source.label}
              />
            ) : (
              <WebSourceItem
                key={source.label}
                source={source.item}
                label={source.label}
                id={sourceId(message.id, source.label)}
                highlighted={highlighted === source.label}
              />
            ),
          )}
        </ol>
      )}
      <CallsPanel result={result} payload={message.payload} />
    </>
  );
}

/** The element id of a source in a message (what a badge scrolls to). */
function sourceId(messageId, label) {
  return `src-${messageId}-${label}`;
}

/**
 * The sources behind the answer's badges, in call order.
 *
 * Passthrough (one agent answered, D9): that agent's own citations, labelled with its own numbers.
 * Otherwise: every doc/web call's sources under the labels the orchestrator gave them
 * (call.citation_labels maps "D1" -> the agent's citation number).
 *
 * @param {object} result - an OrchestratorResult.
 * @returns {Array<{kind: "doc"|"web", label: string, item: object}>}
 */
function collectSources(result) {
  const ran = result.calls.filter((call) => call.status === "ok" && call.result);
  if (result.passthrough) {
    const call = ran.find((c) => c.agent === "doc" || c.agent === "web");
    return call ? sourcesOf(call, (n) => String(n)) : [];
  }
  return ran.flatMap((call) => {
    const labelOf = Object.fromEntries(Object.entries(call.citation_labels).map(([label, n]) => [n, label]));
    return sourcesOf(call, (n) => labelOf[n]).filter((source) => source.label);
  });
}

/** A doc or web call's sources with the given labelling; other agents have none. */
function sourcesOf(call, label) {
  if (call.agent === "doc") return call.result.citations.map((c) => ({ kind: "doc", label: label(c.n), item: c }));
  if (call.agent === "web") return call.result.sources.map((s) => ({ kind: "web", label: label(s.n), item: s }));
  return [];
}

/**
 * One web page the answer used: title, site, snippet, and a link that opens it in a new tab.
 *
 * @param {{source: {title: string, url: string, snippet: string}, label: string, id: string, highlighted: boolean}} props
 */
function WebSourceItem({ source, label, id, highlighted }) {
  return (
    <li id={id} className={highlighted ? "source flash" : "source"}>
      <span className="cite static">{label}</span>
      <div className="source-body">
        <div className="source-head">
          <span className="source-file">{source.title}</span>
          <a className="source-open" href={source.url} target="_blank" rel="noreferrer">
            {hostname(source.url)} <ExternalLinkIcon />
          </a>
        </div>
        <div className="source-snippet">{source.snippet}</div>
      </div>
    </li>
  );
}

/** "https://example.com/a/b" -> "example.com" (falls back to the URL itself). */
function hostname(url) {
  try {
    return new URL(url).hostname;
  } catch {
    return url;
  }
}

/**
 * One chip per agent call (in order, with its status), why the loop stopped if a limit or a failure
 * ended it, and the total time.
 *
 * @param {{result: object, latencyMs: number|null}} props
 */
function CallChips({ result, latencyMs }) {
  const stop = STOP_LABELS[result.stop_reason];
  return (
    <div className="stats">
      {result.calls.length === 0 && <span className="chip">Answered directly</span>}
      {result.calls.map((call) => (
        <span key={call.call_id} className={`chip ${callStatus(call).tone}`} title={call.error ?? undefined}>
          {callLabel(call)}
        </span>
      ))}
      {stop && <span className="chip warn">{stop}</span>}
      {latencyMs != null && <span className="chip">{(latencyMs / 1000).toFixed(1)} s</span>}
    </div>
  );
}

/**
 * Collapsible "Show agent calls" panel: the flow and reasoning (AgentFlow), then every call.
 *
 * @param {{result: object, payload: object}} props
 */
function CallsPanel({ result, payload }) {
  const [open, setOpen] = useState(false);
  const count = result.calls.length;
  return (
    <div className="details">
      <button type="button" className="details-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? "Hide" : "Show"} {count ? `agent calls (${count})` : "reasoning"}
        <ChevronIcon open={open} />
      </button>
      {open && (
        <div className="details-panel">
          <AgentFlow payload={payload} result={result} />
          {count > 0 && (
            <>
              <div className="details-label">Agent calls, in order</div>
              <ol className="calls">
                {result.calls.map((call, index) => (
                  <CallDetails key={call.call_id} call={call} index={index} />
                ))}
              </ol>
            </>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * One call: a one-line header (agent, status, what it was asked, how long it took) that opens what
 * the agent did. Collapsed by default: rows and retrieval tables can be long.
 *
 * @param {{call: object, index: number}} props
 */
function CallDetails({ call, index }) {
  const [open, setOpen] = useState(false);
  const asked = call.input.question ?? call.input.query ?? (call.input.to ? `to ${call.input.to.join(", ")}` : "");
  return (
    <li className="call">
      <button type="button" className="call-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className={`chip ${callStatus(call).tone}`}>
          {index + 1}. {callLabel(call)}
        </span>
        <span className="call-input">{asked}</span>
        {call.duration_ms > 0 && <span className="call-time">{(call.duration_ms / 1000).toFixed(1)} s</span>}
        <ChevronIcon open={open} />
      </button>
      {open && (
        <div className="call-body">
          <CallBody call={call} />
        </div>
      )}
    </li>
  );
}

/** What one agent did: its own detail view, or why the call failed or was refused. */
function CallBody({ call }) {
  if (call.agent === "email") return <EmailCallBody call={call} />;
  if (call.status !== "ok" || !call.result) {
    return (
      <p className="details-error">
        {call.status === "refused" ? "Refused by a safety rule" : "Failed"}: {call.error}
      </p>
    );
  }
  if (call.agent === "db") return <DbResultBody result={call.result} />;
  if (call.agent === "doc") return <RetrievalBody result={call.result} />;
  return <WebCallBody result={call.result} />;
}

/** A web search call: the query that was sent, and the pages that came back. */
function WebCallBody({ result }) {
  return (
    <>
      <div className="details-label">Search query</div>
      <p className="details-explanation">{result.query}</p>
      <div className="details-label">Pages ({result.sources.length})</div>
      <ol className="web-links">
        {result.sources.map((source) => (
          <li key={source.n}>
            <a href={source.url} target="_blank" rel="noreferrer">
              {source.title}
            </a>{" "}
            <span className="source-meta">{hostname(source.url)}</span>
          </li>
        ))}
      </ol>
      {result.reasoning?.length > 0 && <p className="details-explanation">{result.reasoning.join(" ")}</p>}
    </>
  );
}

/** A draft_email call: what the orchestrator asked the email agent to write (the card above shows the result). */
function EmailCallBody({ call }) {
  return (
    <>
      <dl className="email-fields">
        <dt>To</dt>
        <dd>{(call.input.to ?? []).join(", ")}</dd>
        <dt>Subject</dt>
        <dd>{call.input.subject}</dd>
      </dl>
      <div className="details-label">Content requested</div>
      <pre className="email-body-text">{call.input.content}</pre>
      {call.error && (
        <p className="details-error">
          {call.status === "refused" ? "Refused by a safety rule" : "Not sent"}: {call.error}
        </p>
      )}
      <p className="details-explanation">Agent: {AGENT_LABELS.email}. Status: {callStatus(call).text}.</p>
    </>
  );
}
