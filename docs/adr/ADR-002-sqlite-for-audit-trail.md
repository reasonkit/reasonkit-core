# ADR-002: SQLite for Audit Trail

## Status

**Accepted** - 2024-12-28

## Context

ReasonKit's core value proposition is **auditable AI reasoning**. Every reasoning step, decision, and module invocation must be logged for:

1. **Debugging**: Understanding why a reasoning chain produced specific output
2. **Compliance**: Providing audit trails for regulated industries (finance, healthcare)
3. **Improvement**: Analyzing patterns to improve reasoning quality
4. **Reproducibility**: Replaying reasoning chains with identical inputs

We evaluated several storage options:

| Option                       | Pros                                                       | Cons                                                          |
| ---------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------- |
| **PostgreSQL**               | Full SQL, excellent querying, ACID, proven at scale        | Requires server, complex setup, connection management         |
| **File-based (JSON/JSONL)**  | Simple, portable, human-readable                           | No indexing, slow queries, no ACID, append-only complexity    |
| **Redis**                    | Fast, good for recent data                                 | In-memory (data loss risk), limited querying, requires server |
| **SQLite**                   | Zero setup, single file, full SQL, ACID, excellent tooling | Single-writer limitation, no replication                      |
| **Embedded RocksDB/LevelDB** | Very fast writes, LSM-tree efficiency                      | No SQL, complex queries require custom code                   |

### Key Requirements

1. **Zero Configuration**: Developers should get audit logging by default without setup
2. **Portability**: Audit logs should be easily shared, backed up, or analyzed offline
3. **Query Capability**: Need to filter by time, module, confidence, outcome
4. **Reliability**: Cannot lose audit data; ACID guarantees required
5. **Performance**: Logging should not slow down reasoning (<1ms overhead per step)

### Simon Willison's LLM CLI Precedent

The `llm` CLI tool (by Simon Willison) uses SQLite for all conversation logging and demonstrates the pattern's viability:

- Automatic logging to `~/.llm/logs.db`
- Full queryability via SQL
- Integration with Datasette for visualization
- Zero user configuration required

## Decision

**We will use SQLite as the primary storage for audit trails and reasoning logs.**

Specifically:

- Default database location: `~/.reasonkit/audit.db`
- Write-ahead logging (WAL) mode for concurrent reads during writes
- Schema versioned with migrations
- Optional Datasette integration for web UI exploration
- Export to JSON/CSV for external analysis

### Schema Design (Core Tables)

```sql
-- Reasoning sessions
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    profile TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    status TEXT NOT NULL
);

-- Individual reasoning steps
CREATE TABLE steps (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    module TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    input_summary TEXT,
    output_summary TEXT,
    confidence REAL,
    tokens_used INTEGER,
    error TEXT
);

-- Module decisions and branching
CREATE TABLE decisions (
    id TEXT PRIMARY KEY,
    step_id TEXT NOT NULL REFERENCES steps(id),
    decision_type TEXT NOT NULL,
    options_considered TEXT,
    chosen_option TEXT,
    rationale TEXT
);

-- Indexes for common queries
CREATE INDEX idx_sessions_started ON sessions(started_at);
CREATE INDEX idx_steps_module ON steps(module);
CREATE INDEX idx_steps_confidence ON steps(confidence);
```

## Consequences

### Positive

1. **Zero Setup**: Works immediately after `cargo install reasonkit`
2. **Portable**: Single `audit.db` file can be copied, shared, backed up
3. **Full SQL**: Complex queries for analysis without custom code
4. **ACID Guarantees**: Reasoning logs survive crashes; no corruption
5. **Excellent Tooling**: sqlite3 CLI, Datasette, DB Browser, language bindings
6. **Low Overhead**: Sub-millisecond writes with WAL mode
7. **Proven Pattern**: Matches `llm` CLI approach; developers familiar with it

### Negative

1. **Single-Writer Limitation**: Only one process can write at a time
2. **No Built-in Replication**: No native distributed support
3. **File Locking**: Network filesystems (NFS) can cause issues
4. **Scale Ceiling**: Not suitable for extremely high-throughput scenarios

### Mitigations

| Negative       | Mitigation                                                                 |
| -------------- | -------------------------------------------------------------------------- |
| Single writer  | WAL mode allows concurrent reads; queued writes for multi-process          |
| No replication | Export functionality for backup; enterprise version can use PostgreSQL     |
| File locking   | Document limitation; recommend local filesystem for database               |
| Scale ceiling  | Per-user databases; enterprise tier with PostgreSQL for shared deployments |

### Query Examples

```sql
-- Find all sessions with low confidence steps
SELECT DISTINCT s.id, s.started_at, MIN(st.confidence) as min_conf
FROM sessions s
JOIN steps st ON st.session_id = s.id
GROUP BY s.id
HAVING MIN(st.confidence) < 0.7;

-- Module usage statistics
SELECT module, COUNT(*) as uses, AVG(confidence) as avg_conf
FROM steps
WHERE started_at > datetime('now', '-7 days')
GROUP BY module
ORDER BY uses DESC;

-- Reasoning chain replay data
SELECT * FROM steps
WHERE session_id = 'sess_abc123'
ORDER BY started_at;
```

## Related Documents

- `src/telemetry/storage.rs` - Storage implementation

## References

- [SQLite When to Use](https://www.sqlite.org/whentouse.html)
- [Simon Willison's LLM Tool](https://llm.datasette.io/)
- [Datasette](https://datasette.io/)
- [SQLite WAL Mode](https://www.sqlite.org/wal.html)
