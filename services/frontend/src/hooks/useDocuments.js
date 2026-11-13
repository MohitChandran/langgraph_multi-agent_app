/**
 * useDocuments: the documents of one chat, kept fresh while they are being processed.
 *
 * A custom hook is a function whose name starts with "use" and that uses React's own hooks
 * (state, effects). It packages "data + how to keep it up to date" so ChatPage stays readable.
 *
 * Polling: processing happens in the background on the server, so we ask for the list every
 * 2 seconds WHILE any document is queued or processing, and stop as soon as all are ready or
 * failed. No document busy = no requests at all.
 */
import { useCallback, useEffect, useState } from "react";
import { listDocuments } from "../api.js";

const POLL_MS = 2_000;

/**
 * @param {string|undefined} sessionId - the open chat (undefined on "/", a new chat has no documents).
 * @returns {{documents: Array<object>, refresh: () => void}} refresh() reloads now (e.g. after an upload).
 */
export function useDocuments(sessionId) {
  const [documents, setDocuments] = useState([]);
  const [version, setVersion] = useState(0); // bumping it re-runs the effect below = reload + restart polling

  useEffect(() => {
    setDocuments([]);
    if (!sessionId) return undefined;
    let cancelled = false; // set when the chat changes or the component unmounts: ignore late answers
    let timer;

    async function load() {
      try {
        const list = await listDocuments(sessionId);
        if (cancelled) return;
        setDocuments(list);
        if (list.some((doc) => doc.status === "queued" || doc.status === "processing")) {
          timer = setTimeout(load, POLL_MS);
        }
      } catch {
        // chat deleted or API down: the chat page shows its own error; just stop polling
      }
    }

    load();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [sessionId, version]);

  const refresh = useCallback(() => setVersion((value) => value + 1), []);
  return { documents, refresh };
}
