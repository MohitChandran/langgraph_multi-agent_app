/**
 * The rounded input box at the bottom (Gemini calls it the "prompt bar").
 *
 * Enter sends, Shift+Enter adds a new line, and the text area grows with its
 * content up to a maximum height. While THIS chat is generating a reply, sending is
 * disabled (the server allows one reply at a time per chat and would answer 409);
 * other chats stay usable. There's no stop button: the server finishes and saves the
 * answer anyway, so you can switch away and come back to it.
 */
import { useEffect, useRef, useState } from "react";
import { SendIcon } from "./icons.jsx";

const MAX_HEIGHT_PX = 200;

/**
 * @param {{onSend: (question: string) => void, busy: boolean, disabled?: boolean}} props
 *   busy = this chat is generating a reply; disabled = the chat can't be used (e.g. not found).
 */
export default function Composer({ onSend, busy, disabled = false }) {
  const [text, setText] = useState("");
  const textareaRef = useRef(null);
  const blocked = busy || disabled;

  // Grow/shrink the textarea to fit its text: reset the height, then use the content height.
  useEffect(() => {
    const textarea = textareaRef.current;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, MAX_HEIGHT_PX)}px`;
  }, [text]);

  // Put the cursor back in the box when the chat becomes usable, so the next question can be typed right away.
  useEffect(() => {
    if (!blocked) textareaRef.current?.focus();
  }, [blocked]);

  /** Send the trimmed text (if any) and clear the box. */
  function submit(event) {
    event?.preventDefault();
    const question = text.trim();
    if (!question || blocked) return;
    onSend(question);
    setText("");
  }

  /**
   * Enter = send; Shift+Enter = new line. isComposing is true while typing with an
   * input method (e.g. Hindi/Japanese keyboards), where Enter confirms a character instead.
   */
  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <form className="composer" onSubmit={submit}>
      <textarea
        ref={textareaRef}
        rows={1}
        value={text}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={busy ? "Waiting for the answer…" : "Ask the operations database"}
        aria-label="Your question"
        maxLength={1000}
        disabled={disabled}
      />
      <button type="submit" className="icon-button send" disabled={blocked || !text.trim()} aria-label="Send">
        <SendIcon />
      </button>
    </form>
  );
}
