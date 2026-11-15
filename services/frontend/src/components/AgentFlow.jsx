/**
 * "How this was answered": the path a question took and the reasoning at each hop,
 * shown at the top of the "Show retrieval" and "Show SQL & data" panels.
 *
 *   Your question -> Router -> Document agent -> Answer
 *   Router (rule): why this agent was chosen
 *   Document agent: what it did and why (rewrite, search, best match, outcome)
 *
 * Both parts are saved with the answer (payload.route and payload.result.reasoning), so a
 * reload shows exactly what happened then. Today the router is a fixed rule (a chat with
 * documents -> Document agent); the Phase 2 orchestrator will make it an LLM decision
 * ("router": "llm"), shown as "Orchestrator" with its own reasoning.
 * Older messages saved before this existed simply show less.
 */

const AGENT_NAMES = { db: "Database agent", doc: "Document agent" };

/**
 * @param {{payload: object, result: object}} props - the assistant message's payload and its result.
 */
export default function AgentFlow({ payload, result }) {
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
