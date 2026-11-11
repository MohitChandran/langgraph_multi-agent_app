/**
 * App shell: the sidebar (chat list) + the routed chat page.
 *
 *   /              -> ChatPage with no session (a new chat, created by its first message)
 *   /c/:sessionId  -> ChatPage for that chat (reloads and bookmarks work)
 *   anything else  -> redirect to /
 *
 * App owns the sidebar's list of sessions and refreshes it whenever something changes
 * (a message was sent, a chat renamed or deleted), so titles, order and "busy" dots
 * stay correct. The chat page tells App via onSessionsChanged.
 */
import { useCallback, useEffect, useState } from "react";
import { Navigate, Route, Routes, useMatch, useNavigate } from "react-router-dom";
import { deleteSession, listSessions, renameSession } from "./api.js";
import { MenuIcon } from "./components/icons.jsx";
import Sidebar from "./components/Sidebar.jsx";
import ChatPage from "./pages/ChatPage.jsx";

const PHONE_QUERY = "(max-width: 900px)";
const RUNNING_POLL_MS = 3_000;

/** True on narrow screens, where the sidebar overlays the chat instead of sitting beside it. */
const isPhone = () => window.matchMedia(PHONE_QUERY).matches;

export default function App() {
  const [sessions, setSessions] = useState([]);
  const [sessionsError, setSessionsError] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(() => !isPhone());
  const navigate = useNavigate();
  const activeId = useMatch("/c/:sessionId")?.params.sessionId; // the open chat, for highlighting

  /**
   * Reload the chat list from the server. useCallback keeps the same function between
   * renders, so ChatPage can call it from async code without it ever going stale.
   */
  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await listSessions());
      setSessionsError(null);
    } catch (error) {
      setSessionsError(error.message);
    }
  }, []);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  // While any chat is generating a reply (maybe in another tab), re-check every few seconds
  // so its "busy" dot disappears when it's done. No running chats = no polling at all.
  const anyRunning = sessions.some((session) => session.is_running);
  useEffect(() => {
    if (!anyRunning) return undefined;
    const timer = setInterval(refreshSessions, RUNNING_POLL_MS);
    return () => clearInterval(timer);
  }, [anyRunning, refreshSessions]);

  /** On phones, close the overlay sidebar after picking something. */
  function closeOnPhone() {
    if (isPhone()) setSidebarOpen(false);
  }

  /** "New chat": go to "/". Clicking it while already on "/" still starts fresh (new location.key). */
  function newChat() {
    navigate("/");
    closeOnPhone();
  }

  /** Rename a chat, then refresh the list. Errors show at the top of the sidebar. */
  async function rename(id, title) {
    try {
      await renameSession(id, title);
    } catch (error) {
      setSessionsError(error.message);
    }
    await refreshSessions();
  }

  /** Delete a chat; if it was the open one, go to a new chat. Then refresh the list. */
  async function remove(id) {
    try {
      await deleteSession(id);
      if (id === activeId) navigate("/", { replace: true });
    } catch (error) {
      setSessionsError(error.message);
    }
    await refreshSessions();
  }

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        activeId={activeId}
        open={sidebarOpen}
        error={sessionsError}
        onNewChat={newChat}
        onNavigate={closeOnPhone}
        onClose={() => setSidebarOpen(false)}
        onRename={rename}
        onDelete={remove}
      />
      <div className="main">
        <header className="topbar">
          <button
            type="button"
            className="icon-button ghost"
            onClick={() => setSidebarOpen((open) => !open)}
            aria-label={sidebarOpen ? "Hide chats" : "Show chats"}
          >
            <MenuIcon />
          </button>
          <div className="brand">
            Ops Assistant <span className="badge">DB agent</span>
          </div>
        </header>
        <Routes>
          <Route path="/" element={<ChatPage onSessionsChanged={refreshSessions} />} />
          <Route path="/c/:sessionId" element={<ChatPage onSessionsChanged={refreshSessions} />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </div>
  );
}
