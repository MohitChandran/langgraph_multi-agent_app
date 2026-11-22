/**
 * Chat rows, rendered from messages in the API's shape (as stored in Postgres):
 *   { id, role: "user"|"assistant", content, status: "pending"|"complete"|"error"|"awaiting_approval", payload,
 *     latency_ms }
 *
 *   UserMessage      - the question, right-aligned in a rounded bubble
 *   AssistantMessage - picks its look from `status` (thinking / error / answer) and, for answers,
 *                      from `payload.agent`: "doc" answers get citations and sources (DocAnswer.jsx),
 *                      "orchestrator" answers (plan 03) get agent-call chips, sources, the email
 *                      approval card and per-call details (OrchestratorAnswer.jsx); "guard" answers
 *                      (plan 04: the input guard blocked the message) show the refusal and why
 *   ThinkingMessage  - spinning sparkle + shimmering lines while the agent works
 */
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import AgentFlow from "./AgentFlow.jsx";
import DocAnswer from "./DocAnswer.jsx";
import { SparkleIcon } from "./icons.jsx";
import OrchestratorAnswer from "./OrchestratorAnswer.jsx";
import ResultDetails from "./ResultDetails.jsx";

/**
 * @param {{message: {content: string}}} props
 */
export function UserMessage({ message }) {
  return (
    <div className="message user">
      <div className="bubble">{message.content}</div>
    </div>
  );
}

/**
 * An answer from the agent.
 *   pending  -> the thinking row (the reply is still being generated, maybe in another tab)
 *   error    -> the saved error text, in red
 *   complete -> markdown answer + stats + "Show SQL & data"
 *   awaiting_approval -> (orchestrator answers) the answer + the email approval card
 *
 * The answer text is `content`; the agent's details (SQL, rows, steps) are in
 * payload.result, which ResultDetails already understands. react-markdown never renders
 * raw HTML, so text that came from the database can't inject scripts into the page.
 *
 * @param {{message: object, thinkingLabel?: string, onEmailDecided?: (message: object|null) => void}} props
 *   onEmailDecided = the page's handler for an approved/rejected email (swaps in the updated message).
 */
export function AssistantMessage({ message, thinkingLabel, onEmailDecided }) {
  if (message.status === "pending") return <ThinkingMessage label={thinkingLabel} />;

  const result = message.payload?.result;
  return (
    <div className="message assistant">
      <div className="avatar">
        <SparkleIcon />
      </div>
      <div className="content">
        {message.status === "error" ? (
          <p className="error-text">{message.content}</p>
        ) : message.payload?.agent === "doc" && result ? (
          <DocAnswer message={message} />
        ) : message.payload?.agent === "orchestrator" && result ? (
          <OrchestratorAnswer message={message} onEmailDecided={onEmailDecided} />
        ) : message.payload?.agent === "guard" ? (
          // Blocked by the input guard (plan 04): the refusal, then where it stopped and why. Its result is a
          // GuardResult, not a DB result, so the DB panels below would crash on it.
          <>
            <p className="guard-refusal">{message.content}</p>
            <div className="details-panel guard-panel">
              <AgentFlow payload={message.payload} result={result} />
            </div>
          </>
        ) : (
          <>
            <div className="markdown">
              <Markdown remarkPlugins={[remarkGfm]}>{message.content}</Markdown>
            </div>
            {result && <Stats result={result} latencyMs={message.latency_ms} />}
            {result && <ResultDetails result={result} payload={message.payload} />}
          </>
        )}
      </div>
    </div>
  );
}

/**
 * Placeholder row shown while a reply is being generated: Gemini-style shimmer lines.
 *
 * @param {{label?: string}} props - what the agent is doing ("Searching your documents…").
 */
export function ThinkingMessage({ label = "Querying the database…" }) {
  return (
    <div className="message assistant" aria-live="polite">
      <div className="avatar">
        <SparkleIcon spinning />
      </div>
      <div className="content">
        <div className="thinking-label">{label}</div>
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
 * @param {{result: object, latencyMs: number|null}} props - latencyMs = server-side time for the reply.
 */
function Stats({ result, latencyMs }) {
  const attempts = `${result.attempts} attempt${result.attempts === 1 ? "" : "s"}`;
  return (
    <div className="stats">
      <span className={result.success ? "chip ok" : "chip warn"}>
        {result.success ? "From database" : "Not answered"}
      </span>
      <span className={result.attempts > 1 ? "chip warn" : "chip"}>{attempts}</span>
      {result.success && <span className="chip">{result.rows.length} rows</span>}
      {latencyMs != null && <span className="chip">{(latencyMs / 1000).toFixed(1)} s</span>}
    </div>
  );
}
