/**
 * The chat list on the left (plan 01, section 8).
 *
 * Shows every chat from GET /api/sessions, most recently active first (the server sorts).
 * Each item is a link to /c/<id>; the open chat is highlighted; a pulsing dot marks a
 * chat that's still generating a reply. Rename is inline (click the pencil, Enter saves,
 * Esc cancels); delete asks for confirmation because it can't be undone.
 *
 * The sidebar owns no data: the list and the rename/delete actions come from App,
 * which refreshes the list after every change.
 */
import { useState } from "react";
import { Link } from "react-router-dom";
import { PencilIcon, PlusIcon, TrashIcon } from "./icons.jsx";

/**
 * @param {{
 *   sessions: Array<{id: string, title: string, is_running: boolean}>,
 *   activeId: string|undefined,
 *   open: boolean,
 *   error: string|null,
 *   onNewChat: () => void,
 *   onNavigate: () => void,
 *   onClose: () => void,
 *   onRename: (id: string, title: string) => Promise<void>,
 *   onDelete: (id: string) => Promise<void>,
 * }} props - onNavigate runs after a chat is picked (closes the sidebar on phones).
 */
export default function Sidebar({ sessions, activeId, open, error, onNewChat, onNavigate, onClose, onRename, onDelete }) {
  const [editingId, setEditingId] = useState(null);
  const [draft, setDraft] = useState("");

  /** Switch an item into rename mode, pre-filled with its current title. */
  function startRename(session) {
    setEditingId(session.id);
    setDraft(session.title);
  }

  /** Save the new title if it changed and isn't empty, then leave rename mode. */
  async function commitRename(session) {
    const title = draft.trim();
    setEditingId(null);
    if (title && title !== session.title) await onRename(session.id, title.slice(0, 100));
  }

  /** Enter saves, Escape cancels. */
  function handleRenameKey(event, session) {
    if (event.key === "Enter") commitRename(session);
    if (event.key === "Escape") setEditingId(null);
  }

  /** Ask before deleting: there is no undo (the messages are deleted too). */
  function confirmDelete(session) {
    if (window.confirm(`Delete "${session.title}"?\nIts messages are deleted too. This can't be undone.`)) {
      onDelete(session.id);
    }
  }

  return (
    <>
      {open && <div className="backdrop" onClick={onClose} aria-hidden="true" />}
      <aside className={open ? "sidebar open" : "sidebar"} aria-label="Chats">
        <div className="sidebar-header">
          <button type="button" className="new-chat" onClick={onNewChat}>
            <PlusIcon /> New chat
          </button>
        </div>

        <div className="sidebar-label">Recent</div>
        {error && <p className="sidebar-error">{error}</p>}

        <nav className="session-list">
          {sessions.length === 0 && !error && <p className="sidebar-empty">No chats yet. Ask something to start one.</p>}
          {sessions.map((session) =>
            editingId === session.id ? (
              <div key={session.id} className="session-item editing">
                <input
                  className="rename-input"
                  value={draft}
                  maxLength={100}
                  autoFocus
                  aria-label="Chat title"
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={(event) => handleRenameKey(event, session)}
                  onBlur={() => commitRename(session)}
                />
              </div>
            ) : (
              <div key={session.id} className={session.id === activeId ? "session-item active" : "session-item"}>
                <Link to={`/c/${session.id}`} className="session-link" title={session.title} onClick={onNavigate}>
                  {session.is_running && <span className="running-dot" title="Answering…" />}
                  <span className="session-title">{session.title}</span>
                </Link>
                <div className="session-actions">
                  <button type="button" onClick={() => startRename(session)} aria-label={`Rename "${session.title}"`}>
                    <PencilIcon />
                  </button>
                  <button type="button" onClick={() => confirmDelete(session)} aria-label={`Delete "${session.title}"`}>
                    <TrashIcon />
                  </button>
                </div>
              </div>
            ),
          )}
        </nav>
      </aside>
    </>
  );
}
