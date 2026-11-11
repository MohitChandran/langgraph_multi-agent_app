/**
 * The collapsible "Show SQL & data" panel under an answer.
 *
 * This is what makes the UI useful for TESTING the agent: you see which workflow
 * steps ran (including retries), the exact SQL that was executed, and the raw rows
 * the answer was written from, so every answer can be checked against the data.
 */
import { useState } from "react";
import { ChevronIcon } from "./icons.jsx";

/**
 * @param {{result: object}} props - the DBAgentResult returned by the API.
 */
export default function ResultDetails({ result }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  /** Copy the SQL to the clipboard (e.g. to paste into DBeaver) and briefly show "Copied". */
  async function copySql() {
    await navigator.clipboard.writeText(result.sql);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="details">
      <button type="button" className="details-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? "Hide" : "Show"} SQL & data
        <ChevronIcon open={open} />
      </button>

      {open && (
        <div className="details-panel">
          <div className="details-label">Workflow</div>
          <ol className="steps">
            {result.steps.map((step, index) => (
              <li key={index} className={`step step-${step}`}>
                {step}
              </li>
            ))}
          </ol>

          {result.explanation && <p className="details-explanation">{result.explanation}</p>}

          <div className="details-label with-action">
            SQL
            <button type="button" className="text-button" onClick={copySql}>
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
          <pre className="sql">
            <code>{result.sql}</code>
          </pre>

          {result.error && <p className="details-error">Last error: {result.error}</p>}

          {result.columns.length > 0 && (
            <>
              <div className="details-label">
                Rows ({result.rows.length}
                {result.hit_row_limit ? ", row limit reached" : ""})
              </div>
              <ResultTable columns={result.columns} rows={result.rows} />
            </>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Scrollable table of query results with a sticky header.
 *
 * @param {{columns: string[], rows: object[]}} props
 */
function ResultTable({ columns, rows }) {
  if (rows.length === 0) return <p className="details-empty">The query returned no rows.</p>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column} className={typeof row[column] === "number" ? "num" : undefined}>
                  {formatCell(row[column])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Format one cell for display: thousands separators for numbers, a dash for NULL,
 * yes/no for booleans and JSON text for objects (e.g. audit_logs.details).
 *
 * @param {unknown} value
 * @returns {string}
 */
function formatCell(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
