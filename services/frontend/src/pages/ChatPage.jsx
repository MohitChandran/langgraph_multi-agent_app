/**
 * One chat: its messages, the composer, sending (plan 01, section 8).
 *
 * Routes: "/" = a new, empty chat; "/c/:sessionId" = an existing chat.
 * Documents (plan 02, R8): attach with the paperclip or drag-and-drop; chips above the composer
 * show their processing status (useDocuments polls it). The orchestrator on the server (plan 03)
 * decides per question whether to ask the database, these documents, the web, or draft an email.
 * Email approval: while an answer is "awaiting_approval", the composer is locked (the server would
 * answer 409) until you approve or reject the draft on its card; the card then swaps in the final answer.
 * The SERVER is the source of truth: messages are loaded from GET .../messages, and after
 * a send we reload them instead of patching a local copy. The browser only adds two
 * temporary "optimistic" rows (your question + a thinking row) while it waits.
 *
 * The hard part is time. An answer takes seconds, and meanwhile you may switch chats,
 * reload, or start a new chat. So:
 *   - Switching chats does NOT cancel the request: the server finishes and saves the
 *     answer anyway. When the request returns, we only update the screen if you're
 *     still on that chat (checked with refs, see below).
 *   - If a chat's last message is "pending" (you came back mid-answer, or reloaded), we
 *     poll every 2 s until it's done, giving up after 2 minutes.
 *   - A first message on "/" creates the chat; when the answer arrives we jump to
 *     /c/<new id> (replace: true, so Back doesn't return to an empty "/").
 *
 * Why refs? An async function that resumes after `await` still sees the state and props
 * from the render in which it STARTED. Refs are one mutable box shared by all renders,
 * so `sessionIdRef.current` always answers "which chat is on screen right now?".
 */
import { useEffect, useRef, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import { deleteDocument, getMessages, sendMessage, uploadDocument } from "../api.js";
import Composer from "../components/Composer.jsx";
import DocumentChips from "../components/DocumentChips.jsx";
import Greeting from "../components/Greeting.jsx";
import { AssistantMessage, UserMessage } from "../components/Messages.jsx";
import { useDocuments } from "../hooks/useDocuments.js";

const POLL_INTERVAL_MS = 2_000;
const POLL_GIVE_UP_MS = 120_000;
const SIDEBAR_REFRESH_DELAY_MS = 400;

let localCounter = 0;
/** A temporary message shown before the server has saved it. Its id is a string, so it never clashes with a real id. */
function localMessage(role, content, status = "complete") {
  localCounter += 1;
  return { id: `local-${localCounter}`, role, content, status, payload: {}, latency_ms: null };
}

/**
 * @param {{onSessionsChanged: () => void}} props - asks App to refresh the sidebar (titles, order, busy dots).
 */
export default function ChatPage({ onSessionsChanged }) {
  const { sessionId } = useParams(); // undefined on "/"
  const location = useLocation(); // location.key is new on every navigation, even "/" -> "/"
  const navigate = useNavigate();

  const [messages, setMessages] = useState([]);
  const [hasMore, setHasMore] = useState(false);
  const [view, setView] = useState("ready"); // "loading" | "ready" | "not-found" | "error"
  const [viewError, setViewError] = useState(null);
  const { documents, refresh: refreshDocuments } = useDocuments(sessionId);
  const [uploading, setUploading] = useState(0); // files still being sent
  const [notice, setNotice] = useState(null); // upload/delete problems, shown above the composer
  const [dragging, setDragging] = useState(false); // a file is being dragged over the chat

  const sessionIdRef = useRef(sessionId); // which chat is on screen right now
  const pageKeyRef = useRef(location.key); // which visit of "/" is on screen right now
  const loadSeqRef = useRef(0); // only the newest load may update the screen
  const pollRef = useRef({ timer: null, startedAt: 0 });
  const seedRef = useRef(null); // messages to show right after navigating to a chat we just created
  const bottomRef = useRef(null);

  /** Cancel any scheduled poll. */
  function stopPolling() {
    clearTimeout(pollRef.current.timer);
    pollRef.current = { timer: null, startedAt: 0 };
  }

  /**
   * Load a chat's latest messages and show them, if the user is still on that chat and no
   * newer load has started (loadSeq). Starts polling if the last message is still pending.
   */
  async function reload(id) {
    const seq = ++loadSeqRef.current;
    const isCurrent = () => seq === loadSeqRef.current && sessionIdRef.current === id;
    let page;
    try {
      page = await getMessages(id);
    } catch (error) {
      if (!isCurrent()) return;
      stopPolling();
      if (error.status === 404 || error.status === 422) {
        setView("not-found");
      } else {
        setView("error");
        setViewError(error.message);
      }
      return;
    }
    if (!isCurrent()) return;
    setMessages(page.items);
    setHasMore(page.has_more);
    setView("ready");

    if (page.items.at(-1)?.status === "pending") {
      schedulePoll(id);
    } else if (pollRef.current.startedAt) {
      stopPolling(); // the reply we were waiting for has arrived
      onSessionsChanged();
    }
  }

  /** Check again in 2 s; after 2 minutes, stop and show a message instead of waiting forever. */
  function schedulePoll(id) {
    const poll = pollRef.current;
    if (!poll.startedAt) poll.startedAt = Date.now();
    if (Date.now() - poll.startedAt > POLL_GIVE_UP_MS) {
      stopPolling();
      setMessages((previous) =>
        previous.map((message) =>
          message.status === "pending"
            ? { ...message, status: "error", content: "This reply is taking too long. Reload the page to check again." }
            : message,
        ),
      );
      return;
    }
    clearTimeout(poll.timer);
    poll.timer = setTimeout(() => reload(id), POLL_INTERVAL_MS);
  }

  // Runs whenever the URL changes: a different chat, "/" (new chat), or "New chat" clicked again on "/".
  useEffect(() => {
    sessionIdRef.current = sessionId;
    pageKeyRef.current = location.key;
    stopPolling();
    loadSeqRef.current += 1; // answers to loads started for the previous chat are now ignored
    setViewError(null);
    setHasMore(false);

    const seed = seedRef.current;
    seedRef.current = null;
    if (!sessionId) {
      setMessages([]);
      setView("ready");
    } else if (seed?.sessionId === sessionId) {
      setMessages(seed.messages); // we just created this chat and already have its messages
      setView("ready");
    } else {
      setMessages([]);
      setView("loading");
      reload(sessionId);
    }
    return stopPolling;
    // Only the URL is a dependency on purpose: reload/stopPolling use refs and setters, which never go stale.
  }, [sessionId, location.key]);

  // Scroll to the newest message when it changes (not when older messages are prepended).
  const last = messages.at(-1);
  const lastKey = last ? `${last.id}:${last.status}` : "";
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [lastKey]);

  /**
   * Send a question: show it right away with a thinking row, POST it, then show the saved
   * version (reload, or navigate to the new chat). Errors replace the thinking row.
   */
  async function send(content) {
    const targetId = sessionId ?? null;
    const pageKey = location.key;
    const thinking = localMessage("assistant", "", "pending");
    setMessages((previous) => [...previous, localMessage("user", content), thinking]);
    const stillHere = () => (targetId === null ? pageKeyRef.current === pageKey : sessionIdRef.current === targetId);
    // Shortly after sending, the server has claimed the chat: refresh so the sidebar shows its "busy" dot.
    setTimeout(onSessionsChanged, SIDEBAR_REFRESH_DELAY_MS);

    try {
      const turn = await sendMessage({ sessionId: targetId, content });
      onSessionsChanged();
      if (!stillHere()) return; // you moved on; the answer is saved and the sidebar shows the chat
      if (targetId === null) {
        seedRef.current = { sessionId: turn.session.id, messages: [turn.user_message, turn.assistant_message] };
        navigate(`/c/${turn.session.id}`, { replace: true });
      } else {
        await reload(targetId);
      }
    } catch (error) {
      onSessionsChanged();
      if (!stillHere()) return;
      if (error.status === 404) {
        setView("not-found");
        return;
      }
      setMessages((previous) =>
        previous.map((message) =>
          message.id === thinking.id ? { ...message, status: "error", content: error.message } : message,
        ),
      );
    }
  }

  /**
   * An email draft was approved or rejected: show the returned (final) version of that message. null means
   * "something changed elsewhere" (decided in another tab, chat deleted): reload this chat instead.
   */
  function handleEmailDecided(updated) {
    onSessionsChanged();
    if (!updated) {
      if (sessionIdRef.current) reload(sessionIdRef.current);
      return;
    }
    setMessages((previous) => previous.map((message) => (message.id === updated.id ? updated : message)));
  }

  /** Prepend the previous page of messages ("Load earlier messages"). */
  async function loadEarlier() {
    const id = sessionId;
    try {
      const page = await getMessages(id, { before: messages[0].id });
      if (sessionIdRef.current !== id) return;
      setMessages((previous) => [...page.items, ...previous]);
      setHasMore(page.has_more);
    } catch (error) {
      setViewError(error.message);
    }
  }

  /**
   * Upload files one by one. On "/" (a new chat) the first upload creates the chat; the rest go
   * into it, and we then open that chat. Errors (too big, unsupported type) are shown per file.
   */
  async function uploadFiles(files) {
    const pageKey = location.key;
    let targetId = sessionId ?? null;
    let createdId = null;
    setNotice(null);
    setUploading((count) => count + files.length);
    for (const file of files) {
      try {
        const { session } = await uploadDocument(file, targetId);
        if (!targetId) {
          targetId = createdId = session.id;
        }
      } catch (error) {
        setNotice(`${file.name}: ${error.message}`);
      } finally {
        setUploading((count) => count - 1);
      }
    }
    onSessionsChanged();
    if (createdId && pageKeyRef.current === pageKey) {
      navigate(`/c/${createdId}`, { replace: true });
    } else if (targetId && sessionIdRef.current === targetId) {
      refreshDocuments(); // shows the new chips and starts polling their status
    }
  }

  /** Remove a document from this chat after confirming (its chunks and files are deleted too). */
  async function removeDocument(doc) {
    if (!window.confirm(`Remove "${doc.filename}" from this chat?\nIt will no longer be used to answer questions.`)) return;
    try {
      await deleteDocument(doc.id);
    } catch (error) {
      setNotice(`${doc.filename}: ${error.message}`);
    }
    refreshDocuments();
  }

  /** Drag-and-drop: a file is over the chat. dataTransfer.types tells files apart from dragged text. */
  function handleDragOver(event) {
    if (view === "not-found" || !event.dataTransfer.types.includes("Files")) return;
    event.preventDefault(); // allow dropping
    setDragging(true);
  }

  /** The drag left the chat area itself (not just moved onto a child element): hide the overlay. */
  function handleDragLeave(event) {
    if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false); // left the chat area itself
  }

  /** Files were dropped: upload them into this chat (or a new one on "/"). */
  function handleDrop(event) {
    event.preventDefault(); // otherwise the browser opens the file in the tab
    setDragging(false);
    const files = [...event.dataTransfer.files];
    if (files.length) uploadFiles(files);
  }

  const busy = last?.status === "pending";
  const hasDocuments = documents.length > 0;
  const awaitingApproval = messages.some((message) => message.status === "awaiting_approval");
  let placeholder = hasDocuments ? "Ask about your documents, data or the web" : "Ask about invoices, vendors, contracts…";
  if (awaitingApproval) placeholder = "Approve or reject the email first";

  return (
    <div className="chat-page" onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
      {dragging && (
        <div className="drop-overlay">
          <div className="drop-box">Drop files to add them to this chat</div>
        </div>
      )}
      <main className="chat">
        <div className="thread">
          {view === "loading" && <p className="thread-note">Loading chat…</p>}
          {view === "not-found" && (
            <p className="thread-note">
              This chat doesn't exist (it may have been deleted). <Link to="/">Start a new chat</Link>
            </p>
          )}
          {viewError && <p className="thread-note error-text">{viewError}</p>}
          {view === "ready" && messages.length === 0 && <Greeting onPick={send} hasDocuments={hasDocuments} />}
          {hasMore && (
            <button type="button" className="load-earlier" onClick={loadEarlier}>
              Load earlier messages
            </button>
          )}
          {messages.map((message) =>
            message.role === "user" ? (
              <UserMessage key={message.id} message={message} />
            ) : (
              <AssistantMessage
                key={message.id}
                message={message}
                thinkingLabel="Asking the agents…"
                onEmailDecided={handleEmailDecided}
              />
            ),
          )}
          <div ref={bottomRef} />
        </div>
      </main>

      <footer className="composer-area">
        {notice && (
          <p className="upload-notice" role="alert">
            {notice}
            <button type="button" className="text-button" onClick={() => setNotice(null)}>
              Dismiss
            </button>
          </p>
        )}
        <DocumentChips documents={documents} uploading={uploading} onDelete={removeDocument} />
        <Composer
          onSend={send}
          onAttach={uploadFiles}
          busy={busy}
          disabled={view === "not-found" || awaitingApproval}
          placeholder={placeholder}
        />
        <p className="disclaimer">
          Answers come from the database, your documents and the web. Check the SQL and the cited sources for
          anything important. Emails are only sent after you approve them.
        </p>
      </footer>
    </div>
  );
}
