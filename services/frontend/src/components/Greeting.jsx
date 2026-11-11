/**
 * Empty-state screen shown before the first question: a greeting and example
 * questions. The examples double as a quick smoke test of the DB agent: each one
 * exercises a different skill (date filters, joins, currencies, aggregation).
 */

const SUGGESTIONS = [
  "Which pending invoices are due this week?",
  "Which contracts expire in the next 60 days, and who owns them?",
  "What did we spend per vendor category last quarter?",
  "Which vendors have the most open high or critical tickets?",
];

/**
 * @param {{onPick: (question: string) => void}} props - called with the clicked example question.
 */
export default function Greeting({ onPick }) {
  return (
    <section className="greeting">
      <h1 className="greeting-title">Hello there</h1>
      <p className="greeting-subtitle">Ask me anything about vendors, contracts, invoices and tickets.</p>
      <div className="suggestions">
        {SUGGESTIONS.map((question) => (
          <button key={question} type="button" className="suggestion" onClick={() => onPick(question)}>
            {question}
          </button>
        ))}
      </div>
    </section>
  );
}
