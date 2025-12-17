```mermaid
sequenceDiagram
    participant Agent
    participant LLM
    participant Fallback

    Agent->>LLM: Request
    LLM-->>Agent: Error
    Agent->>LLM: Retry
    LLM-->>Agent: Error
    Agent->>Fallback: Rule-based response
```
