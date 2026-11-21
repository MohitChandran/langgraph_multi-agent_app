/**
 * Email cards for orchestrator answers (plan 03, section 13):
 *
 *   EmailApprovalCard - the answer is "awaiting_approval": the orchestrator drafted an email and is
 *                       PAUSED until you decide. Shows To and Subject, an editable Body, and
 *                       "Approve & send" / "Reject". Nothing is sent before you click.
 *   EmailOutcome      - after the decision: sent / rejected / not sent, with the final body folded away.
 *
 * Approving calls POST /api/messages/{id}/email-approval, which resumes the paused orchestrator run
 * on the server and returns the SAME message with its final answer; the page swaps it in place.
 * The email's recipients can't be edited here on purpose: they must be addresses you typed (rule S11).
 */
import { useId, useState } from "react";
import { approveEmail } from "../api.js";

/**
 * @param {{messageId: number, email: object, onDecided: (message: object|null) => void}} props
 *   email = payload.result.email: {to, subject, body, status, ...}.
 *   onDecided = called with the updated assistant message, or null when the page should reload
 *   (e.g. the email was already decided in another tab: 409).
 */
export default function EmailApprovalCard({ messageId, email, onDecided }) {
  const [body, setBody] = useState(email.body ?? "");
  const [pending, setPending] = useState(null); // "approve" | "reject" while the request runs
  const [error, setError] = useState(null);
  const bodyId = useId();
  const edited = body !== (email.body ?? "");

  /** Send the decision. The server then sends (or discards) the email and writes the final answer. */
  async function decide(approved) {
    setPending(approved ? "approve" : "reject");
    setError(null);
    try {
      const response = await approveEmail(messageId, { approved, edited_body: approved && edited ? body : null });
      onDecided(response.assistant_message);
    } catch (problem) {
      setPending(null);
      setError(problem.message);
      if (problem.status === 404 || problem.status === 409) onDecided(null); // decided elsewhere, or gone: reload
    }
  }

  return (
    <section className="email-card awaiting" aria-label="Email waiting for your approval">
      <div className="email-card-head">✉ Email draft · waiting for your approval</div>
      <dl className="email-fields">
        <dt>To</dt>
        <dd>{email.to.join(", ")}</dd>
        <dt>Subject</dt>
        <dd>{email.subject}</dd>
      </dl>
      <label className="email-label" htmlFor={bodyId}>
        Body {edited && <span className="email-edited">(edited)</span>}
      </label>
      <textarea
        id={bodyId}
        className="email-body"
        value={body}
        onChange={(event) => setBody(event.target.value)}
        rows={8}
        maxLength={5000}
        disabled={pending !== null}
      />
      {error && <p className="error-text">{error}</p>}
      <div className="email-actions">
        <button
          type="button"
          className="action-button primary"
          onClick={() => decide(true)}
          disabled={pending !== null || !body.trim()}
        >
          {pending === "approve" ? "Sending…" : "Approve & send"}
        </button>
        <button type="button" className="action-button" onClick={() => decide(false)} disabled={pending !== null}>
          {pending === "reject" ? "Rejecting…" : "Reject"}
        </button>
        <span className="email-note">Nothing is sent until you approve.</span>
      </div>
    </section>
  );
}

const OUTCOMES = {
  sent: { tone: "ok", title: "✉ Email sent" },
  rejected: { tone: "warn", title: "⨯ Email rejected: nothing was sent" },
  error: { tone: "warn", title: "✗ Email not sent" },
};

/**
 * What happened to the email of a finished answer. The body is folded away (click to open).
 *
 * @param {{email: object}} props - payload.result.email with status "sent" | "rejected" | "error".
 */
export function EmailOutcome({ email }) {
  const outcome = OUTCOMES[email.status] ?? { tone: "", title: `Email: ${email.status}` };
  return (
    <details className={`email-card ${outcome.tone}`}>
      <summary className="email-card-head">
        {outcome.title}
        {email.to?.length > 0 && <span className="email-to"> · to {email.to.join(", ")}</span>}
      </summary>
      <dl className="email-fields">
        <dt>Subject</dt>
        <dd>{email.subject}</dd>
      </dl>
      <pre className="email-body-text">{email.body}</pre>
      {email.error && <p className="details-error">{email.error}</p>}
    </details>
  );
}
