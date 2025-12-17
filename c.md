```mermaid
sequenceDiagram
    participant User
    participant API
    participant Supervisor
    participant DB
    participant Memory

    User->>API: Query with JWT
    API->>Memory: Load history
    API->>Supervisor: Process query
    Supervisor->>DB: Execute tenant query
    Supervisor->>Memory: Store context
    Supervisor-->>User: Response
```
