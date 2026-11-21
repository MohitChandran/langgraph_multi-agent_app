/**
 * Empty-state screen shown before the first question: a greeting and example questions,
 * about the data or, once documents are attached, about the documents. The examples double
 * as a quick smoke test of the orchestrator: a database question with dates, one with joins,
 * one that drafts an email (approval card), and one that needs the database AND the web.
 */

const DATABASE_SUGGESTIONS = [
  "Which pending invoices are due this week?",
  "Which contracts expire in the next 60 days, and who owns them?",
  "Which vendors have overdue invoices? Email the list to accounts@company.example",
  "How many open tickets do we have for Example Cloud, and did it change its SLA recently?",
];

// In a chat with uploaded documents, ask_documents becomes available: suggest document questions.
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
          : "Ask about vendors, contracts, invoices and tickets, attach a document, or ask me to email a summary."}
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
