graph TD
    Start[User Query] --> Router{Intent Router}
    
    Router -->|greeting, casual| SmallTalk[Small Talk Agent]
    Router -->|what is, explain, definition| Knowledge[Knowledge Agent]
    Router -->|show, list, get, fetch| SQL[SQL Pipeline]
    Router -->|plot, chart, visualize| Viz[Visualizer Agent]
    Router -->|forecast, predict, anomaly| Insight[Insight Agent]
    
    SQL --> Entity[Entity Extraction]
    Entity --> Text2SQL[Text-to-SQL]
    Text2SQL --> Validate{SQL Valid?}
    Validate -->|No| Retry[Retry with Error Context]
    Retry --> Text2SQL
    Validate -->|Yes| Execute[Execute Query]
    Execute --> DataFetch[Data Fetcher]
    
    DataFetch --> CheckViz{Visualization Needed?}
    CheckViz -->|Yes| Viz
    CheckViz -->|No| Response[Generate Response]
    
    Insight --> Analytics[Run Analytics]
    Analytics --> Viz
    
    Viz --> Response
    Knowledge --> Response
    SmallTalk --> Response
