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
 * Older messages saved before this existed simply show less.
 */
import { callLabel, callStatus } from "../calls.js";

const AGENT_NAMES = { db: "Database agent", doc: "Document agent" };

/**
 * @param {{payload: object, result: object}} props - the assistant message's payload and its result.
 */
export default function AgentFlow({ payload, result }) {
  if (payload?.agent === "orchestrator") return <OrchestratorFlow result={result} />;
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
 * The orchestrator's version: one pill per agent call, in the order they ran, each with its status
 * (✓ done, ✗ failed/refused, ⏸ awaiting approval, ✉ sent, ⨯ rejected), then the orchestrator's
 * reasoning lines (written by code from the calls, so they match what really happened).
 *
 * @param {{result: object}} props - an OrchestratorResult.
 */
function OrchestratorFlow({ result }) {
  const calls = result?.calls ?? [];
  const reasoning = result?.reasoning ?? [];
  return (
    <div className="flow">
      <div className="details-label">How this was answered</div>
      <ol className="flow-steps" aria-label="Route of the question">
        <li>
          <span className="flow-step">Your question</span>
        </li>
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
