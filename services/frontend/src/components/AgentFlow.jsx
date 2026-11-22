/**
 * "How this was answered": the path a question took and the reasoning at each hop,
 * shown at the top of the "Show retrieval" and "Show SQL & data" panels.
 *
 *   Your question -> Router -> Document agent -> Answer
 *   Router (rule): why this agent was chosen
 *   Document agent: what it did and why (rewrite, search, best match, outcome)
 *
 * Both parts are saved with the answer (payload.route and payload.result.reasoning), so a
 * reload shows exactly what happened then. Older answers were routed by a fixed rule ("Router (rule)");
 * since plan 03 every answer comes from the orchestrator, which can call several agents:
 *
 *   Your question -> Orchestrator -> Database ✓ -> Web search ✓ -> Answer
 *   Orchestrator: one line per call ("Asked the database agent: ... → 8 rows."), and why it stopped
 *
 * Since plan 04 an input guard checks every message first: a passed message shows "Guardrail ✓" before the
 * orchestrator; a blocked one (payload.agent "guard") shows "Guardrail ⛔" and nothing after it.
 *
 * Older messages saved before this existed simply show less.
 */
import { callLabel, callStatus } from "../calls.js";

const AGENT_NAMES = { db: "Database agent", doc: "Document agent" };

/**
 * @param {{payload: object, result: object}} props - the assistant message's payload and its result.
 */
export default function AgentFlow({ payload, result }) {
  if (payload?.agent === "orchestrator") return <OrchestratorFlow payload={payload} result={result} />;
  if (payload?.agent === "guard") return <GuardFlow result={result} />;
  const agentName = AGENT_NAMES[payload?.agent] ?? "Agent";
  const route = payload?.route;
  const routerName = route?.router === "llm" ? "Orchestrator" : "Router";
  const reasoning = result?.reasoning ?? [];

  return (
    <div className="flow">
      <div className="details-label">How this was answered</div>
      <ol className="flow-steps" aria-label="Route of the question">
        <li>
          <span className="flow-step">Your question</span>
        </li>
        <li>
          <span className="flow-step">{routerName}</span>
        </li>
        <li>
          <span className="flow-step active">{agentName}</span>
        </li>
        <li>
          <span className="flow-step">Answer</span>
        </li>
      </ol>
      {route?.reason && (
        <p className="flow-note">
          <span className="flow-who">
            {routerName}
            {route.router === "rule" ? " (rule)" : ""}:
          </span>{" "}
          {route.reason}
        </p>
      )}
      {reasoning.length > 0 && (
        <div className="flow-note">
          <span className="flow-who">{agentName}:</span>
          <ul>
            {reasoning.map((line, index) => (
              <li key={index}>{line}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/**
 * The orchestrator's version: "Guardrail ✓" (when the input guard checked the message), then one pill per
 * agent call, in the order they ran, each with its status (✓ done, ✗ failed/refused, ⏸ awaiting approval,
 * ✉ sent, ⨯ rejected), then the orchestrator's reasoning lines (written by code from the calls, so they match
 * what really happened).
 *
 * @param {{payload: object, result: object}} props - the message payload (payload.guard) and an OrchestratorResult.
 */
function OrchestratorFlow({ payload, result }) {
  const calls = result?.calls ?? [];
  const reasoning = result?.reasoning ?? [];
  const guard = payload?.guard;
  return (
    <div className="flow">
      <div className="details-label">How this was answered</div>
      <ol className="flow-steps" aria-label="Route of the question">
        <li>
          <span className="flow-step">Your question</span>
        </li>
        {guard?.status === "passed" && (
          <li>
            <span className="flow-step ok" title={`Checked in ${guard.duration_ms} ms`}>
              Guardrail ✓
            </span>
          </li>
        )}
        <li>
          <span className="flow-step active">Orchestrator</span>
        </li>
        {calls.map((call) => (
          <li key={call.call_id}>
            <span className={`flow-step ${callStatus(call).tone}`} title={call.error ?? callStatus(call).text}>
              {callLabel(call)}
            </span>
          </li>
        ))}
        <li>
          <span className="flow-step">Answer</span>
        </li>
      </ol>
      {guard?.jailbreak_flagged && (
        <p className="flow-note">
          <span className="flow-who">Guardrail:</span> the jailbreak detector flagged this message (score{" "}
          {guard.jailbreak_score}), but the safety check judged it a normal request.
        </p>
      )}
      {reasoning.length > 0 && (
        <div className="flow-note">
          <span className="flow-who">Orchestrator:</span>
          <ul>
            {reasoning.map((line, index) => (
              <li key={index}>{line}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/**
 * A message the input guard blocked: it stopped at the guardrail, and nothing behind it ran (no orchestrator,
 * no agent, no tool). Shows what was decided and why (category and the safety model's short reason).
 *
 * @param {{result: object}} props - a GuardResult (status "blocked").
 */
function GuardFlow({ result }) {
  const category = result?.category ?? result?.rail ?? "blocked";
  return (
    <div className="flow">
      <div className="details-label">How this was handled</div>
      <ol className="flow-steps" aria-label="Route of the question">
        <li>
          <span className="flow-step">Your question</span>
        </li>
        <li>
          <span className="flow-step warn">Guardrail ⛔ {category}</span>
        </li>
        <li>
          <span className="flow-step">Not processed</span>
        </li>
      </ol>
      <p className="flow-note">
        <span className="flow-who">Guardrail:</span>{" "}
        {result?.category === "unavailable"
          ? "the safety check couldn't run, so the message was not processed (it never lets an unchecked message through)."
          : `blocked by the ${result?.rail ?? "safety check"}${result?.reason ? `: ${sentence(result.reason)}` : "."}`}{" "}
        No agent, tool or database saw this message.
      </p>
    </div>
  );
}

/** End a model-written reason with a full stop, so the next sentence doesn't run into it. */
function sentence(text) {
  const trimmed = text.trim();
  return /[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`;
}
