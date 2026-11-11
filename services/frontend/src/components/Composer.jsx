/**
 * The rounded input box at the bottom (Gemini calls it the "prompt bar").
 *
 * Enter sends, Shift+Enter adds a new line, and the text area grows with its
 * content up to a maximum height. While a request runs, the send button turns
 * into a stop button.
 */
import { useEffect, useRef, useState } from "react";
import { SendIcon, StopIcon } from "./icons.jsx";

const MAX_HEIGHT_PX = 200;

/**
 * @param {{onSend: (question: string) => void, onStop: () => void, pending: boolean}} props
 */
export default function Composer({ onSend, onStop, pending }) {
  const [text, setText] = useState("");
  const textareaRef = useRef(null);

  // Grow/shrink the textarea to fit its text: reset the height, then use the content height.
  useEffect(() => {
    const textarea = textareaRef.current;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, MAX_HEIGHT_PX)}px`;
  }, [text]);

  // Put the cursor back in the box when an answer arrives, so the next question can be typed right away.
  useEffect(() => {
    if (!pending) textareaRef.current?.focus();
  }, [pending]);

  /** Send the trimmed text (if any) and clear the box. */
  function submit(event) {
    event?.preventDefault();
    const question = text.trim();
    if (!question || pending) return;
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
        placeholder="Ask the operations database"
        aria-label="Your question"
        maxLength={1000}
      />
      {pending ? (
        <button type="button" className="icon-button stop" onClick={onStop} aria-label="Stop">
          <StopIcon />
        </button>
      ) : (
        <button type="submit" className="icon-button send" disabled={!text.trim()} aria-label="Send">
          <SendIcon />
        </button>
      )}
    </form>
  );
}
