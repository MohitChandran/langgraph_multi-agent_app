/**
 * API client: the ONLY place the frontend talks to the backend.
 *
 * Components call askDatabase(question) and get either a result object or a thrown
 * Error with a human-readable message. They never deal with fetch, status codes or
 * JSON parsing. When the API changes (e.g. streaming in Phase 2), only this file changes.
 */

/**
 * Ask the DB agent a question.
 *
 * @param {string} question - the user's question in plain English.
 * @param {AbortSignal} [signal] - lets the caller cancel the request (e.g. on "New chat").
 * @returns {Promise<object>} the DBAgentResult from the backend:
 *   { answer, success, sql, explanation, columns, rows, hit_row_limit, attempts, steps, error }
 * @throws {Error} with a message that is safe to show in the UI.
 */
export async function askDatabase(question, signal) {
  let response;
  try {
    response = await fetch("/api/db/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal,
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new Error("Can't reach the API. Is the backend running on port 8000?");
  }

  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(readErrorDetail(body) ?? `The API returned an error (HTTP ${response.status}).`);
  }
  return body;
}

/**
 * Pull a readable message out of a FastAPI error body.
 *
 * FastAPI sends {"detail": "text"} for errors we raise, and
 * {"detail": [{"msg": "..."}, ...]} for request validation errors (HTTP 422).
 *
 * @param {object|null} body - parsed JSON error body, or null if it wasn't JSON.
 * @returns {string|null} the message, or null if none was found.
 */
function readErrorDetail(body) {
  const detail = body?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((item) => item.msg).join("; ");
  return null;
}
