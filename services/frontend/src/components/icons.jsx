/**
 * Small inline SVG icons. Inline SVG needs no icon library or network request,
 * and `currentColor` makes the simple icons follow the surrounding text colour
 * (so they work in light and dark mode automatically).
 */
import { useId } from "react";

/**
 * Four-point sparkle with a blue -> purple -> pink gradient: the assistant's avatar.
 *
 * @param {{size?: number, spinning?: boolean}} props - spinning=true while the agent is working.
 */
export function SparkleIcon({ size = 28, spinning = false }) {
  // Each icon needs its own gradient id; useId gives a unique one per instance.
  const gradientId = `sparkle-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={spinning ? "spin" : undefined} aria-hidden="true">
      <defs>
        <linearGradient id={gradientId} x1="2" y1="2" x2="22" y2="22" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#4285f4" />
          <stop offset="0.55" stopColor="#9b72cb" />
          <stop offset="1" stopColor="#d96570" />
        </linearGradient>
      </defs>
      <path
        d="M12 2C12.6 7.4 16.6 11.4 22 12C16.6 12.6 12.6 16.6 12 22C11.4 16.6 7.4 12.6 2 12C7.4 11.4 11.4 7.4 12 2Z"
        fill={`url(#${gradientId})`}
      />
    </svg>
  );
}

/** Paper-plane "send" icon for the composer button. */
export function SendIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M3.4 20.4 20.85 12.92a1 1 0 0 0 0-1.84L3.4 3.6a1 1 0 0 0-1.39.91L2 9.12c0 .5.37.93.87.99L17 12 2.87 13.88c-.5.07-.87.5-.87 1l.01 4.61c0 .71.73 1.2 1.39.91z" />
    </svg>
  );
}

/** Square "stop" icon, shown in place of send while a request is running. */
export function StopIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <rect x="5" y="5" width="14" height="14" rx="2" />
    </svg>
  );
}

/** Plus icon for the "New chat" button. */
export function PlusIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}

/**
 * Down-pointing chevron; rotated 180 degrees by CSS when its panel is open.
 *
 * @param {{open?: boolean}} props
 */
export function ChevronIcon({ open = false }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={open ? "chevron open" : "chevron"}
      aria-hidden="true"
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}
