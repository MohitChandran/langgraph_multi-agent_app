/**
 * Words and icons for the orchestrator's agent calls (plan 03, section 13), shared by the answer
 * chips (OrchestratorAnswer.jsx) and the "How this was answered" flow (AgentFlow.jsx).
 *
 * A call is one AgentCall from payload.result.calls: {call_id, tool, agent, input, status, result,
 * citation_labels, error, duration_ms}. Its status is one of:
 *   ok                 the agent ran (it may still have found nothing: result.success === false)
 *   error              the agent failed or timed out
 *   refused            a safety rule stopped the call before it ran (error says which)
 *   awaiting_approval  an email draft waits for the user
 *   sent / rejected    the user approved (and it was sent) / rejected the draft
 */

/** Short agent names for chips. */
export const AGENT_LABELS = { db: "Database", doc: "Documents", web: "Web search", email: "Email" };

const STATUS = {
  ok: { icon: "✓", text: "done", tone: "ok" },
  error: { icon: "✗", text: "failed", tone: "warn" },
  refused: { icon: "✗", text: "refused", tone: "warn" },
  awaiting_approval: { icon: "⏸", text: "awaiting approval", tone: "" },
  sent: { icon: "✉", text: "sent", tone: "ok" },
  rejected: { icon: "⨯", text: "rejected", tone: "warn" },
};

/** Why the loop stopped, when it wasn't the supervisor's own answer (result.stop_reason). */
export const STOP_LABELS = {
  tool_budget: "Stopped: tool budget used up",
  supervisor_limit: "Stopped: step limit reached",
  llm_error_fallback: "Orchestrator unavailable: fixed rule used",
  llm_error_forced: "Orchestrator failed mid-way",
};

/**
 * How to show one call's status. An agent that ran but couldn't answer is "no answer" (a warning),
 * not a green tick.
 *
 * @param {object} call - an AgentCall.
 * @returns {{icon: string, text: string, tone: string}} tone = "ok" | "warn" | "" (a CSS class).
 */
export function callStatus(call) {
  if (call.status === "ok" && call.result?.success === false) return { icon: "!", text: "no answer", tone: "warn" };
  return STATUS[call.status] ?? { icon: "", text: call.status, tone: "" };
}

/**
 * The chip text for a call, e.g. "Database ✓" or "Email ✉ sent".
 *
 * @param {object} call - an AgentCall.
 * @returns {string}
 */
export function callLabel(call) {
  const status = callStatus(call);
  const name = AGENT_LABELS[call.agent] ?? (call.tool || "Unknown tool");
  return call.status === "ok" && status.tone === "ok" ? `${name} ${status.icon}` : `${name} ${status.icon} ${status.text}`;
}
