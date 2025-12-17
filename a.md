```md
```mermaid
graph TD
    Query --> Router

    Router -->|Greeting| SmallTalk
    Router -->|Explain| Knowledge
    Router -->|SQL Query| SQLPipeline
    Router -->|Forecast| Insight

    SQLPipeline --> EntityExtraction
    EntityExtraction --> Text2SQL
    Text2SQL --> ExecuteSQL
    ExecuteSQL --> Response
