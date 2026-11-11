/**
 * The three kinds of chat rows:
 *   UserMessage      - the question, right-aligned in a rounded bubble
 *   AssistantMessage - sparkle avatar + Markdown answer + stats + "Show SQL & data"
 *   ThinkingMessage  - spinning sparkle + shimmering lines while the agent works
 */
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { SparkleIcon } from "./icons.jsx";
import ResultDetails from "./ResultDetails.jsx";

/**
 * @param {{text: string}} props
 */
export function UserMessage({ text }) {
  return (
    <div className="message user">
      <div className="bubble">{text}</div>
    </div>
  );
}

/**
 * An answer from the agent, or an error if the request failed.
 *
 * react-markdown turns the LLM's Markdown (bullets, **bold**, tables via remark-gfm)
 * into React elements. It never renders raw HTML, so text that came from the
 * database can't inject scripts into the page.
 *
 * @param {{message: {result?: object, error?: string, durationMs?: number}}} props
 */
export function AssistantMessage({ message }) {
  const { result, error, durationMs } = message;
  return (
    <div className="message assistant">
      <div className="avatar">
        <SparkleIcon />
      </div>
      <div className="content">
        {error ? (
          <p className="error-text">{error}</p>
        ) : (
          <>
            <div className="markdown">
              <Markdown remarkPlugins={[remarkGfm]}>{result.answer}</Markdown>
            </div>
            <Stats result={result} durationMs={durationMs} />
            {result.sql && <ResultDetails result={result} />}
          </>
        )}
      </div>
    </div>
  );
}

/** Placeholder row shown while waiting for the API: Gemini-style shimmer lines. */
export function ThinkingMessage() {
  return (
    <div className="message assistant" aria-live="polite">
      <div className="avatar">
        <SparkleIcon spinning />
      </div>
      <div className="content">
        <div className="thinking-label">Querying the database…</div>
        <div className="shimmer" />
        <div className="shimmer" />
        <div className="shimmer short" />
      </div>
    </div>
  );
}

/**
 * One line of small "chips" summarising how the answer was produced.
 *
 * @param {{result: object, durationMs: number}} props
 */
function Stats({ result, durationMs }) {
  const attempts = `${result.attempts} attempt${result.attempts === 1 ? "" : "s"}`;
  return (
    <div className="stats">
      <span className={result.success ? "chip ok" : "chip warn"}>
        {result.success ? "From database" : "Not answered"}
      </span>
      <span className={result.attempts > 1 ? "chip warn" : "chip"}>{attempts}</span>
      {result.success && <span className="chip">{result.rows.length} rows</span>}
      <span className="chip">{(durationMs / 1000).toFixed(1)} s</span>
    </div>
  );
}
