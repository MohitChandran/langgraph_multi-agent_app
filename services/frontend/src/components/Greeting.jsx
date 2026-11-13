/**
 * Empty-state screen shown before the first question: a greeting and example questions,
 * about the database or, once documents are attached, about the documents. The database
 * examples double as a quick smoke test of the DB agent: each one exercises a different
 * skill (date filters, joins, currencies, aggregation).
 */

const DATABASE_SUGGESTIONS = [
  "Which pending invoices are due this week?",
  "Which contracts expire in the next 60 days, and who owns them?",
  "What did we spend per vendor category last quarter?",
  "Which vendors have the most open high or critical tickets?",
];

// In a chat with uploaded documents, the Doc agent answers: suggest document questions instead.
const DOCUMENT_SUGGESTIONS = [
  "Summarize the main points of this document.",
  "What do the tables say about service levels?",
  "What do the charts in the document show?",
  "What are the key findings or recommendations?",
];

/**
 * @param {{onPick: (question: string) => void, hasDocuments?: boolean}} props
 *   onPick = called with the clicked example question; hasDocuments = this chat has uploaded documents.
 */
export default function Greeting({ onPick, hasDocuments = false }) {
  const suggestions = hasDocuments ? DOCUMENT_SUGGESTIONS : DATABASE_SUGGESTIONS;
  return (
    <section className="greeting">
      <h1 className="greeting-title">Hello there</h1>
      <p className="greeting-subtitle">
        {hasDocuments
          ? "Ask me anything about your documents. I'll cite the page."
          : "Ask about vendors, contracts, invoices and tickets, or attach a document."}
      </p>
      <div className="suggestions">
        {suggestions.map((question) => (
          <button key={question} type="button" className="suggestion" onClick={() => onPick(question)}>
            {question}
          </button>
        ))}
      </div>
    </section>
  );
}
