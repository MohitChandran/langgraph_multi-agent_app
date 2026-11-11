/**
 * API client: the ONLY place the frontend talks to the backend.
 *
 * Components call these functions and get data back, or a thrown ApiError with a message
 * that is safe to show in the UI and the HTTP status (so a page can tell "chat not found"
 * (404) from "chat busy" (409)). They never deal with fetch or JSON parsing.
 * When /api/chat becomes a stream (Phase 2), only this file changes.
 */

/** An error from the API (or from failing to reach it). status is 0 when the API was unreachable. */
export class ApiError extends Error {
  /**
   * @param {string} message - readable message for the UI.
   * @param {number} status - HTTP status code, or 0 for network errors.
   */
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Send one request and return the parsed JSON body (or null for "204 No Content").
 *
 * @param {string} path - e.g. "/api/sessions".
 * @param {{method?: string, body?: object, signal?: AbortSignal}} [options]
 * @returns {Promise<any>}
 * @throws {ApiError}
 */
async function request(path, { method = "GET", body, signal } = {}) {
  let response;
  try {
    response = await fetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal,
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError("Can't reach the API. Is the backend running on port 8000?", 0);
  }

  if (response.status === 204) return null;
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(readErrorDetail(data) ?? `The API returned an error (HTTP ${response.status}).`, response.status);
  }
  return data;
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

/**
 * All chats for the sidebar, most recently active first.
 * @returns {Promise<Array<{id, title, message_count, created_at, updated_at, is_running}>>}
 */
export function listSessions() {
  return request("/api/sessions");
}

/**
 * A chat's messages, oldest to newest.
 *
 * @param {string} sessionId
 * @param {{before?: number}} [options] - before = only messages older than this message id ("load earlier").
 * @returns {Promise<{items: Array<object>, has_more: boolean}>}
 */
export function getMessages(sessionId, { before } = {}) {
  const query = before ? `?before=${before}` : "";
  return request(`/api/sessions/${sessionId}/messages${query}`);
}

/**
 * Send a message. sessionId null starts a new chat. Resolves when the answer is saved.
 *
 * @param {{sessionId: string|null, content: string}} message
 * @returns {Promise<{session: object, user_message: object, assistant_message: object}>}
 */
export function sendMessage({ sessionId, content }) {
  return request("/api/chat", { method: "POST", body: { session_id: sessionId, content } });
}

/**
 * Rename a chat.
 * @param {string} sessionId
 * @param {string} title
 * @returns {Promise<object>} the updated session.
 */
export function renameSession(sessionId, title) {
  return request(`/api/sessions/${sessionId}`, { method: "PATCH", body: { title } });
}

/**
 * Delete a chat and all its messages.
 * @param {string} sessionId
 * @returns {Promise<null>}
 */
export function deleteSession(sessionId) {
  return request(`/api/sessions/${sessionId}`, { method: "DELETE" });
}
