/**
 * The whole chat app: header, message list (or greeting) and the composer.
 *
 * State lives here and flows down as props:
 *   messages - every row in the chat, in order
 *   pending  - true while waiting for the API (shows the thinking row and stop button)
 * Each question is answered independently: the DB agent has no conversation memory
 * yet (that comes with the orchestrator's checkpointer in Phase 2).
 */
import { useEffect, useRef, useState } from "react";
import { askDatabase } from "./api.js";
import Composer from "./components/Composer.jsx";
import Greeting from "./components/Greeting.jsx";
import { PlusIcon } from "./components/icons.jsx";
import { AssistantMessage, ThinkingMessage, UserMessage } from "./components/Messages.jsx";

/** Return a unique id for React's `key` prop. */
const newId = () => crypto.randomUUID();

export default function App() {
  const [messages, setMessages] = useState([]);
  const [pending, setPending] = useState(false);
  const abortRef = useRef(null); // the AbortController of the request in flight
  const bottomRef = useRef(null); // invisible marker at the end of the thread

  // Keep the newest message (or the thinking row) in view.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, pending]);

  /**
   * Add the question to the chat, call the API, then add the answer (or the error).
   * performance.now() measures how long the whole round trip took, as the user felt it.
   */
  async function send(question) {
    if (pending) return;
    setMessages((previous) => [...previous, { id: newId(), role: "user", text: question }]);
    setPending(true);

    const controller = new AbortController();
    abortRef.current = controller;
    const startedAt = performance.now();
    try {
      const result = await askDatabase(question, controller.signal);
      const durationMs = performance.now() - startedAt;
      setMessages((previous) => [...previous, { id: newId(), role: "assistant", result, durationMs }]);
    } catch (error) {
      if (error.name === "AbortError") return; // stopped by the user; stop() already handled it
      setMessages((previous) => [...previous, { id: newId(), role: "assistant", error: error.message }]);
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        setPending(false);
      }
    }
  }

  /** Cancel the running request (the stop button). The backend may still finish, but we ignore it. */
  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setPending(false);
    setMessages((previous) => [...previous, { id: newId(), role: "assistant", error: "Stopped." }]);
  }

  /** Clear the conversation, cancelling any request in flight. */
  function newChat() {
    abortRef.current?.abort();
    abortRef.current = null;
    setPending(false);
    setMessages([]);
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          Ops Assistant <span className="badge">DB agent</span>
        </div>
        <button type="button" className="new-chat" onClick={newChat} disabled={messages.length === 0}>
          <PlusIcon /> New chat
        </button>
      </header>

      <main className="chat">
        <div className="thread">
          {messages.length === 0 && !pending ? (
            <Greeting onPick={send} />
          ) : (
            messages.map((message) =>
              message.role === "user" ? (
                <UserMessage key={message.id} text={message.text} />
              ) : (
                <AssistantMessage key={message.id} message={message} />
              ),
            )
          )}
          {pending && <ThinkingMessage />}
          <div ref={bottomRef} />
        </div>
      </main>

      <footer className="composer-area">
        <Composer onSend={send} onStop={stop} pending={pending} />
        <p className="disclaimer">Answers come from the operations database. Check the SQL for anything important.</p>
      </footer>
    </div>
  );
}
