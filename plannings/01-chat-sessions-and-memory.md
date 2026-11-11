# Plan 01 — Chat Sessions (Conversations) & Agent Memory

| | |
|---|---|
| **Status** | Planned 2026-11-11. S0 + S1 done 2026-11-11 (dev session); next: S2. |
| **Written in** | a planning-only session. No code was changed. |
| **Implement in** | a development session, stage by stage (Section 10). |
| **Replaces** | an earlier draft with users and login. **Dropped by the user's decision:** no users, no login, no per-user history. |
| **Storage rule** | Postgres only. No Redis, no localStorage as source of truth. |

### How the development session should use this document
1. Read it fully first. Section 2 is the *why*; Sections 4–9 are *what to build*.
2. Build one stage at a time (Section 10). After each stage: run the tests, show the user, and commit only when asked.
3. Learning mode: explain every new file and important function (what and why) in chat, step by step, and give every function
   a `"""docstring"""`. No interview questions until development is finished.
4. Match the existing style: raw SQL with psycopg 3 (no ORM), small modules with module docstrings, Pydantic for API models,
   integration tests that skip when Postgres isn't running.

---

## 1. What we're building (user's words → features)

> "I should be able to create different sessions (conversations) and continue chats there — one session for invoices,
> another for documents — and switch between sessions and continue the chat."

| Feature | Meaning |
|---|---|
| **Create** a session | "New chat" opens an empty chat; the first message creates it |
| **List** sessions | A sidebar shows all chats, most recently active first, with a title |
| **Switch** sessions | Click a chat → its full history loads; the URL is `/c/<session_id>`, so reloads and bookmarks work |
| **Continue** a session | New messages are added to that chat, and the agent **remembers the earlier turns of that chat only** |
| Rename / delete | Small extras: rename a title, delete a chat |

**Not in scope:** users, login, sharing, search, editing old messages, per-user isolation.
(If users are ever needed, it's one `user_id` column on `chat_sessions` plus a `WHERE user_id = ...` in each query.
Nothing in this design blocks that.)

**Today's starting point:** `POST /api/db/ask` answers one question with no memory, and nothing is saved.
The React app keeps messages in `useState`, so a reload loses them.

---

## 2. How chat platforms do it, and why

### 2.1 The pattern (ChatGPT, Claude.ai, Gemini)
1. **Every conversation has an ID** that appears in the URL (`chatgpt.com/c/<id>`). Switching chats = loading a different ID.
2. **History is stored on the server**, one row per message, in order. The browser is just a view, so a reload or another
   tab shows the same chats.
3. **The browser sends only the new message plus the conversation ID.** The server looks up the history itself.
4. **A conversation is created by its first message**, not by clicking "New chat", so no empty chats pile up.
5. **The sidebar is sorted by last activity:** the chat you just used moves to the top.
6. **The model's "memory" is built from the stored history on every turn.** The server takes the *recent* messages of
   *that* conversation and puts them into the prompt. The model itself remembers nothing between requests: memory is
   *re-reading the conversation*. That's why two chats never mix: each request loads only its own history.
7. **The model sees a window, not everything.** You can scroll through 200 messages, but the model gets the last few
   turns (plus, in big products, a summary of older ones). **What you see ≠ what the model sees.**

### 2.2 Why it's done this way
- **One source of truth.** The same Postgres rows feed the UI *and* the agent's memory, so the two can never disagree.
- **Stateless server.** No conversation lives in the API process's memory. Restart the API and every chat is still there;
  later, several API processes could serve the same chat.
- **The browser can't fake history.** If the client sent the history, it could slip in a fake "assistant said..." turn
  (a prompt-injection trick). Loading history server-side prevents that.
- **Isolation for free.** Memory is loaded `WHERE session_id = <this chat>`, so the invoices chat can't leak into the
  documents chat.

### 2.3 Why it's efficient
- **Constant cost per turn.** The agent gets the last **3 exchanges**, not the whole chat. Turn 50 costs the same tokens
  as turn 4. Sending everything would make each turn more expensive than the last, so total cost grows quadratically.
  It would also hit **Groq's free-tier tokens-per-minute limits** within a few turns.
- **Every query is a short index scan.** Sidebar, open chat and load memory each read a few rows through an index,
  whether there are 10 chats or 10,000 (Section 9).
- **A turn is a handful of tiny writes** (a few milliseconds) next to an LLM call that takes seconds.
- **Bulky data stays out of the prompt.** The rows shown under an answer (up to 100) are saved for the UI in a `JSONB`
  column. Memory uses only *question + SQL + short answer*, never the rows.

### 2.4 Memory design: why not LangGraph's checkpointer (yet)?
LangGraph's usual memory is a **checkpointer** (`thread_id` → saved graph state). We **don't** use it for chat memory now:

| | Memory from our messages table (chosen) | LangGraph checkpointer as memory |
|---|---|---|
| Stores | **One:** `chat_messages` feeds both UI and memory | **Two:** UI table *and* checkpoint blobs, which can drift apart |
| Readable | Plain SQL rows, visible in DBeaver | Serialized binary blobs |
| Delete a chat | One `DELETE` (messages cascade) | `DELETE` + `checkpointer.adelete_thread()` |
| Complexity | A ~30-line `load_history()` | New package, pool settings, state reducers, summarize node |
| What it's really for | Conversation context | **Durable execution:** pausing a run and resuming it later |

The checkpointer **is** the right tool in **Phase 6 (human approval of emails)**, where a run must *pause* and *resume*.
It will use the same ID: `thread_id = chat_sessions.id`. Phase 2's orchestrator also uses `load_history()` for its routing
prompt, so this work isn't thrown away.

---

## 3. Architecture

```
 Browser (React)                        FastAPI  (orchestrator-api)                       Postgres  ops_platform
 ───────────────                        ───────────────────────────                       ──────────────────────
 Sidebar  ── GET /api/sessions ──────►  chat routes ──────────────────────────────────►  schema app  (role app_rw)
 /c/<id>  ── GET /api/sessions/<id>/messages ─►                                             chat_sessions
          ── POST /api/chat ─────────►  chat service (one turn):                            chat_messages
                                         1 claim session (one reply at a time) ─────────►
                                         2 load memory: last 3 exchanges ────────────────►
                                         3 save question + "pending" answer ─────────────►
                                         4 DBAgent.ask(question, history)
                                             └─ SQL via read-only pool ──────────────────►  schema public (role readonly_agent,
                                         5 save answer + SQL/rows, release session ──────►   NO access to schema app)
```

**Why a separate schema `app` with its own role `app_rw`** (about 10 lines of setup):
- The DB agent's role `readonly_agent` has automatic SELECT rights on every new table in `public`. Chat tables there would
  be readable by LLM-written SQL. In schema `app`, Postgres refuses: *permission denied for schema app*.
- The chat code then can't touch the business tables either.
- The admin login (`ops_admin`) stays reserved for seeding and schema changes, as `.env.example` says today.

---

## 4. Database

### 4.1 Role and schema: new init script `scripts/db/03_app_schema.sh`
Same style as `02_readonly_role.sh`; runs once on a fresh volume:
```sql
CREATE ROLE app_rw LOGIN PASSWORD :'app_password';      -- env APP_DB_PASSWORD
GRANT CONNECT ON DATABASE :"db_name" TO app_rw;
CREATE SCHEMA app AUTHORIZATION app_rw;                   -- app_rw owns schema app and can create its tables
ALTER ROLE app_rw SET search_path = app;                  -- unqualified table names mean app.<table>
ALTER ROLE app_rw SET statement_timeout = '10s';
-- Not granted on purpose: readonly_agent gets nothing on schema app; app_rw gets nothing on public.
```
Also: add `APP_DB_PASSWORD: ${APP_DB_PASSWORD}` to the postgres service env in `docker-compose.yml`.
Existing machines need `./scripts/reset_db.sh` once. The business data is reproducible, so nothing is lost.

### 4.2 Tables: created by the API at startup (`ChatStore.setup()`)
No migration framework: `CREATE TABLE IF NOT EXISTS`, run on every startup. It's idempotent, the same idea as LangGraph's
`checkpointer.setup()`. If a table changes later during development, run `reset_db.sh` (dev chats are disposable).

```sql
-- One row per conversation (sidebar item). Its id is used everywhere: URL, API, later LangGraph thread_id.
CREATE TABLE IF NOT EXISTS chat_sessions (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title          TEXT NOT NULL,                        -- first question, cut to 60 chars (stage S5: LLM title)
    message_count  INTEGER NOT NULL DEFAULT 0,
    active_run_at  TIMESTAMPTZ,                          -- set while a reply is being generated, else NULL
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()    -- last activity: the sidebar sort key
);
CREATE INDEX IF NOT EXISTS chat_sessions_recent_idx ON chat_sessions (updated_at DESC);

-- Append-only transcript. BIGINT identity = strict order (timestamps can tie).
CREATE TABLE IF NOT EXISTS chat_messages (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,   -- delete chat -> its messages go too
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL DEFAULT '',                -- the question, or the final answer (markdown)
    status      TEXT NOT NULL DEFAULT 'complete' CHECK (status IN ('pending', 'complete', 'error')),
    payload     JSONB NOT NULL DEFAULT '{}',             -- assistant extras for the UI and memory, see 4.3
    latency_ms  INTEGER,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chat_messages_session_idx ON chat_messages (session_id, id);
```

### 4.3 `payload` for assistant messages
```json
{ "agent": "db",
  "result": { "success": true, "sql": "SELECT ...", "explanation": "...", "columns": ["..."], "rows": [{"...": "..."}],
              "hit_row_limit": false, "attempts": 1, "steps": ["write_sql", "validate_sql", "execute_sql", "answer"],
              "error": null } }
```
`result` = today's `DBAgentResult` minus `question` and `answer` (the answer is `content`).
The frontend passes `payload.result` to the existing `result` prop, so `ResultDetails` keeps working unchanged.
The `agent` field matters later: a "documents" chat's messages will carry `"agent": "doc"` with their own result shape.
A session is **not** tied to one agent; in Phase 2 the orchestrator routes each message.

---

## 5. Agent memory

### 5.1 What gets remembered
```python
@dataclass(frozen=True)
class ConversationTurn:
    question: str  # user message content
    answer: str  # assistant message content (cut to ~300 chars for the prompt)
    sql: str | None  # payload.result.sql, so follow-ups can modify the previous query
```
`load_history(session_id, max_turns=3)`, in `app/chat/memory.py`:
```sql
SELECT role, content, status, payload->'result'->>'sql' AS sql
FROM chat_messages
WHERE session_id = %s AND status = 'complete'
ORDER BY id DESC
LIMIT 6;                     -- 3 exchanges; reverse in Python, then pair user -> assistant
```
Rules:
- Load it **before** inserting the new question, so the new question isn't part of its own history.
- Use only `complete` messages: failed or pending turns don't pollute memory.
- Never include rows.
- Make the window size a setting: `CHAT_MEMORY_TURNS=3`.

### 5.2 How the DB agent uses it
- `DBAgent.ask(question, history: list[ConversationTurn] | None = None)`. Default `None` keeps the CLI, `/api/db/ask`
  and the eval working unchanged.
- `DBAgentState` gets `history: str`: the formatted text, built once per turn by `format_history(turns)`.
- `prompts.py`: `WRITE_SQL_HUMAN` and `REPAIR_SQL_HUMAN` get a `{history}` block before `Question:`, and so does
  `ANSWER_HUMAN`, so "those vendors" reads naturally. When there's no history the block is empty, so the prompts behave
  exactly as today. Example of the block:
  ```
  Conversation so far (most recent last):
  Q1: Which vendors have overdue invoices?
  SQL1: SELECT v.name, ... WHERE i.status = 'overdue' ...
  A1: Amazon Web Services, Lenovo and DHL have overdue invoices ...
  ```
  Add to the SQL system prompt: *"The question may refer to earlier turns ('those', 'only India', 'same for last month'):
  resolve it using the conversation and the previous SQL. If it's a new topic, ignore the conversation."*
- Cost: about 400–600 extra prompt tokens, regardless of chat length.

### 5.3 Upgrade path (not now)
- **Rolling summary:** add `chat_sessions.summary`; every 5 turns, a fast model folds older turns into it, and the prompt
  gets summary + last 3 turns. Only add this if real chats show the agent forgetting context.
- **Phase 2:** the orchestrator gets the same `history` for routing ("is this follow-up about invoices or documents?").
- **Phase 6:** LangGraph checkpointer for pausing runs, with `thread_id = str(session_id)`.

---

## 6. One turn, step by step (`POST /api/chat`)
```
input: { session_id: uuid | null, content: str (1–1000 chars) }

1  if session_id is null:                                         -- "New chat": created by its first message
       INSERT INTO chat_sessions (title) VALUES (first 60 chars of content) RETURNING *
2  claim the session (one reply at a time per chat, single statement):
       UPDATE chat_sessions SET active_run_at = now()
       WHERE id = %s AND (active_run_at IS NULL OR active_run_at < now() - interval '5 minutes')
       RETURNING *
   no row -> session doesn't exist: 404 | otherwise it's busy: 409 "A reply is still being generated in this chat"
   (the 5-minute rule frees a chat whose previous run crashed)
3  history = load_history(session_id)                             -- BEFORE saving the new question
4  one transaction:
       INSERT user message      (status 'complete')
       INSERT assistant message (status 'pending', content '')     -- reload during the run shows "thinking"
       UPDATE chat_sessions SET message_count = message_count + 2, updated_at = now()
5  result = await db_agent.ask(content, history)
6  success: UPDATE assistant message SET content = answer, payload = {...}, status = 'complete', latency_ms
   failure: UPDATE ... status = 'error', content = short readable error (full error in the server log)
   always:  UPDATE chat_sessions SET active_run_at = NULL, updated_at = now()      -- in a finally block
7  return { session, user_message, assistant_message }
```
**Why save the question before running the agent:** if the tab closes or the server crashes mid-run, the question isn't lost,
and the `pending` row tells the UI the reply is still coming. On startup, mark leftover `pending` messages older than
10 minutes as `error` ("The reply was interrupted.").

**Switching chats mid-reply:** the request keeps running on the server and saves its answer.
Chatting in *another* session meanwhile is fine; the busy check is per session.

---

## 7. API contract

| Method & path | Body / params | Response |
|---|---|---|
| `POST /api/chat` | `{session_id: uuid \| null, content}` | 200 `{session: Session, user_message: Message, assistant_message: Message}`; 404 unknown session; 409 busy; 422 invalid; 502 agent crashed (still saved as an `error` message) |
| `GET /api/sessions` | – | 200 `[Session]`, newest activity first, max 100 |
| `GET /api/sessions/{id}/messages` | `?limit=100&before=<message_id>` (optional; `before` = load older) | 200 `{items: [Message] (oldest→newest), has_more: bool}`; 404 |
| `PATCH /api/sessions/{id}` | `{title}` (1–100 chars) | 200 `Session`; 404 |
| `DELETE /api/sessions/{id}` | – | 204 (messages deleted by CASCADE); 404 |

```
Session = { id, title, message_count, created_at, updated_at, is_running }   (is_running = active_run_at IS NOT NULL)
Message = { id, role, content, status, payload, latency_ms, created_at }
```
Keep `POST /api/db/ask` unchanged for the CLI, debugging and the eval. The UI stops using it.
**Phase 2 note:** `POST /api/chat` becomes an SSE stream (events: `session`, `step`, `token`, `done`, `error`).
The database writes in Section 6 stay exactly the same.

---

## 8. Frontend (`services/frontend`)

```
┌───────────────┬──────────────────────────────────────────┐
│ + New chat    │  Overdue invoices by vendor               │
│───────────────│                                           │
│ ▸ Overdue in… │   You: Which vendors have overdue …       │
│   Contracts … │   Agent: AWS, Lenovo … [Show SQL & data]  │
│   Spend by c… │   You: only the ones in India             │
│               │   Agent: …                                │
│               │  ┌─────────────────────────────────────┐  │
│               │  │ Ask about your data…           [➤]  │  │
└───────────────┴──┴─────────────────────────────────────┴──┘
```
- `react-router-dom`. Routes: `/` = new chat, `/c/:sessionId` = an existing chat.
- `components/Sidebar.jsx`: list from `GET /api/sessions`, New chat button, highlight the active chat, rename (inline),
  delete (with confirm). Refresh the list after every send.
- `pages/ChatPage.jsx` (most of today's `App.jsx`): on `sessionId` change, load messages. Send →
  `POST /api/chat`. If this was a new chat, `navigate('/c/' + session.id, { replace: true })`.
- Switching chats **doesn't** abort a running request. When it finishes, refresh the sidebar, and reload the messages if the
  user is still on that chat. If the last message is `pending` (e.g. after switching back or a reload), show the thinking
  row and poll `GET .../messages` every 2 seconds until it's done (give up after 2 minutes).
- `api.js`: `listSessions, getMessages, sendMessage, renameSession, deleteSession`, with the same error handling style as
  `askDatabase`.
- Map `message.payload.result` to the existing `result` prop of `AssistantMessage`. A user message maps to `{role, text: content}`.

---

## 9. Efficiency cheat-sheet

| Operation | Query | Index | Cost |
|---|---|---|---|
| Sidebar | `ORDER BY updated_at DESC LIMIT 100` | `chat_sessions_recent_idx` | Reads ≤100 index entries, no sort |
| Open a chat | `WHERE session_id = $1 [AND id < $before] ORDER BY id DESC LIMIT 101` | `chat_messages_session_idx` | ≤101 entries |
| Load memory | same index, `LIMIT 6`, `status = 'complete'` | `chat_messages_session_idx` | ~6 entries |
| Claim session | conditional `UPDATE ... RETURNING` on the PK | PK | 1 row, no long-held lock |
| Save a turn | 2 INSERT + 2 UPDATE | PKs | A few ms |

- `LIMIT n + 1` trick: fetch one extra row to know whether there's more, so no `COUNT(*)` is needed.
- Prompt size per turn is constant (5.2).
- Big row payloads are compressed automatically by Postgres (TOAST) and are never read by the sidebar or memory queries.

---

## 10. Implementation stages

Rough effort: S1 ½ day · S2 1 day · S3 ½ day · S4 1–1½ days · S5 1–2 hours.

### [x] S0 — Housekeeping  (already committed + pushed before this plan: 607b471, 14cef23, 9ca421c)
Commit the pending chat-UI work (frontend + `main.py`), which is uncommitted as of 2026-11-11, so this work starts from a clean tree.

### [x] S1 — Database foundations  (done 2026-11-11, not committed yet)
- `scripts/db/03_app_schema.sh` (4.1); `docker-compose.yml` env; `.env.example`:
  `APP_DB_PASSWORD=...`, `APP_DATABASE_URL=postgresql://app_rw:...@localhost:5432/ops_platform`, `CHAT_MEMORY_TURNS=3`
- `app/config.py`: `app_database_url` (required), `chat_memory_turns: int = 3`
- `app/chat/__init__.py`, `app/chat/store.py`: `ChatStore` class with its own `AsyncConnectionPool` on `APP_DATABASE_URL`
  (`kwargs={"autocommit": True, "row_factory": dict_row}`; multi-statement writes inside `async with conn.transaction()`),
  `open()/close()`/`async with`, and `setup()` running the DDL from 4.2
- `app/main.py` lifespan: open `ChatStore`, `await store.setup()`, mark stale `pending` messages as error; close on shutdown

Acceptance: after `./scripts/reset_db.sh` and starting the API, DBeaver shows `app.chat_sessions` and `app.chat_messages`.
Restarting the API changes nothing. As `readonly_agent`, `SELECT 1 FROM app.chat_messages` fails with
*permission denied for schema app*.

### [ ] S2 — Sessions and transcript API (no memory yet)
- `app/chat/models.py`: Pydantic `Session`, `Message`, `ChatRequest`, `ChatResponse`, `RenameRequest`
- `app/chat/store.py`: methods `create_session, list_sessions, get_session, rename_session, delete_session, claim_session,
  release_session, add_turn (user + pending assistant, one transaction), complete_message, fail_message, list_messages,
  fail_stale_pending`
- `app/chat/service.py`: `send_message(...)`, the Section 6 flow. Takes the agent as a parameter, so tests pass a fake.
- `app/chat/routes.py`: `APIRouter` with Section 7; `main.py` includes it. The agent comes from a dependency `get_db_agent`
  (reads `app.state`), so tests can override it.

Acceptance (in `/docs`): first message with `session_id: null` creates a chat; the second message with that ID appends to it.
Restart the API: `GET /messages` still shows both turns. Delete removes the chat and its messages.

### [ ] S3 — Agent memory
- `app/chat/memory.py`: `ConversationTurn`, `load_history(store, session_id, max_turns)`, `format_history(turns) -> str`
  (answer cut to ~300 chars; empty string for no turns)
- `app/agents/db/models.py`: `DBAgentState.history: str`
- `app/agents/db/graph.py`: `ask(question, history=None)` → formats and puts it into the initial state; `write_sql` and
  `answer` pass `history` to the prompts
- `app/agents/db/prompts.py`: `{history}` block plus the follow-up rule (5.2)
- `service.py`: call `load_history` before `add_turn`, then pass it to `ask`

Acceptance: in one chat, ask "Which vendors have overdue invoices?" then "only the ones in India". The second SQL filters
`country = 'India'` *and* overdue. In a **different** chat, "only the ones in India" gets no memory of the first chat.

### [ ] S4 — Frontend sessions
Section 8. Acceptance:
- Create two chats (invoices and contracts) and switch back and forth; each continues with its own memory.
- Reload on `/c/<id>`: the same chat comes back.
- Switch away during a reply and come back: the thinking indicator, then the answer.
- Rename and delete work; the sidebar order follows the latest activity.

### [ ] S5 — Optional polish: LLM titles
After the first turn, a background task (`fastapi.BackgroundTasks`) asks the fast model for a 3–6 word title
(`max_tokens=20`, because Groq's per-minute output-token limit rejects requests that could exceed it). It updates the title only if it's still
the auto-generated one; never overwrite a manual rename (track it with a boolean `title_is_custom` column or by comparing
to the truncated question). On error, keep the truncated question.

---

## 11. Tests (pytest; DB tests skip when Postgres is down, as today)

Dev dependencies: `httpx` (`AsyncClient(transport=ASGITransport(app=app))`) and `asgi-lifespan` (`LifespanManager`),
because `ASGITransport` alone does **not** run the FastAPI lifespan. Override `get_db_agent` with a fake agent that
records the `history` it received: no Groq calls in tests. Each test deletes the sessions it creates.

| Area | Test |
|---|---|
| Setup | `setup()` twice is a no-op; tables are in schema `app` |
| DB security | `readonly_agent` can't read `app.*`; `validate_sql("SELECT content FROM app.chat_messages")` is rejected |
| Create & continue | `session_id: null` → new session with 2 messages; second send with that ID → 4 messages, same session |
| Order & list | The most recently used session is first; `message_count` is correct |
| Busy guard | Second send while `active_run_at` is set → 409; a claim older than 5 min is taken over; the claim is released after success **and** after an agent exception |
| Failure | Fake agent raises → assistant `status='error'`, the next send works |
| Memory window | After 5 turns the fake agent receives exactly the last 3; `error`/`pending` turns excluded; the new question isn't in its own history |
| Memory isolation | Session B's first turn gets an empty history even though session A has turns |
| Prompts | `format_history([])` = `""`; with turns, the write-SQL prompt contains the previous SQL; the answer is cut to 300 chars |
| Pagination | 120 messages → latest 100 with `has_more=true`; `before` returns the rest |
| Delete | Deleting a session removes its messages; GET → 404 |
| Stale pending | Pending messages older than 10 min become `error` on startup |

---

## 12. Gotchas for the development session
1. **The two pools behave differently on purpose.** The read-only pool keeps autocommit **off** (read-only mode works only
   inside `BEGIN READ ONLY`; see `database.py`). The chat pool uses autocommit **on** plus explicit `conn.transaction()`
   for multi-statement writes.
2. **Init scripts only run on an empty volume.** `03_app_schema.sh` needs `./scripts/reset_db.sh` once.
3. **Two different "users" in the project:** `public.users` = mock employees the DB agent reports on. Nothing in this plan
   adds app users.
4. **Load history *before* inserting the new question,** or the question appears in its own history.
5. **Never put rows into memory.** Only question, SQL and a short answer. Rows bloat prompts and hit Groq's token limits.
6. **Release the session claim in a `finally` block,** or one crash leaves the chat stuck "busy" for 5 minutes.
7. **Bump `updated_at`** on every send, or the sidebar order goes stale.
8. **`ASGITransport` doesn't run the lifespan:** use `asgi-lifespan` or build `app.state` in a fixture.
9. **Don't abort the fetch when switching chats** (today's `newChat()` does). The server finishes and saves anyway; the
   UI should reload instead.

## 13. Decisions taken (user can change)
| # | Decision | Default |
|---|---|---|
| 1 | Memory window | Last **3** exchanges (setting `CHAT_MEMORY_TURNS`) |
| 2 | Memory source | Our `chat_messages` table, not the LangGraph checkpointer (2.4); checkpointer later for Phase 6 |
| 3 | Titles | First question cut to 60 chars; LLM title is optional stage S5 |
| 4 | Delete | Hard delete with a confirm dialog (no trash / undo) |
| 5 | A session's agent | Not fixed; each message is routed (DB agent only until Phase 2) |
