graph TD
    subgraph Ingestion
        A[Raw PDF Document] --> B(pdfplumber / camelot);
        B --> C{Detect Content Type};
    end

    subgraph Analysis & Structured Extraction
        C --> D[Extract Characters & Metadata];
        C --> E[Extract Images / Vector Graphics];
        C --> F[Extract Tables];
    end

    subgraph Dynamic Context Engine
        D --> G[Calculate Mode Font Size];
        D --> H[Calculate StdDev Size];
        G & H --> I[Determine Heading Threshold];
        I --> J(Identify Parents: Headings);
        I --> K(Identify Children: Paragraphs);
        
        E --> L[Merge Image Fragments via Spatial Clustering];
        F --> M[Heal Cross-Page Tables];
    end

    subgraph Semantic Linking ERD
        J & K --> N[Assign Paragraphs to Headings via Coordinate Proximity];
        L & M --> O[Link Images / Tables to Nearby Paragraphs];
    end

    subgraph Binary Serialization .loom
        N & O --> P{Create Graph Structure};
        P --> Q[Convert to MessagePack / Binary];
        Q --> R(.loom File);
    end

    style I fill:#f96,stroke:#333,stroke-width:2px
    style J fill:#bbf,stroke:#333,stroke-width:1px
    style K fill:#bbf,stroke:#333,stroke-width:1px
    style L fill:#bbf,stroke:#333,stroke-width:1px
    style M fill:#bbf,stroke:#333,stroke-width:1px
    style R fill:#6c6,stroke:#333,stroke-width:2px