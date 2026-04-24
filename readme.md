flowchart TD
    %% Styling
    classDef process fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef decision fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef startend fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,rx:20px,ry:20px;
    classDef llm fill:#fff3e0,stroke:#f57c00,stroke-width:2px;

    %% Nodes
    Start([Student Submission: Abstract / Document Upload]) ::: startend
    Extract[Text Extraction & Cleaning] ::: process
    Vectorize[FastEmbed Vectorization: 384-dim Vector] ::: process
    SimCheck[Compute Cosine Similarity against DB] ::: process
    
    Decision{Similarity Score Threshold} ::: decision
    
    Flag[🚨 Flag as Plagiarized] ::: process
    Review[⚠️ Mark for Review] ::: process
    Novel[🌟 Mark as Novel Concept] ::: process
    
    DB[(Store Vectors & Flags in SQLite)] ::: process
    
    LLM[Groq LLM Analysis: LLaMA-3.3-70b] ::: llm
    Summary[Generate Executive Summary] ::: llm
    TechStack[Identify Technology Stack] ::: llm
    Viva[Formulate 5 Custom Viva Questions] ::: llm
    
    Faculty[Faculty Dashboard: Interactive Grading] ::: process
    Export([Export Final Evaluation CSV]) ::: startend

    %% Connections
    Start --> Extract
    Extract --> Vectorize
    Vectorize --> SimCheck
    SimCheck --> Decision
    
    Decision -- "> 75%" --> Flag
    Decision -- "25% - 75%" --> Review
    Decision -- "< 25%" --> Novel
    
    Flag --> DB
    Review --> DB
    Novel --> DB
    
    DB --> LLM
    
    LLM --> Summary
    LLM --> TechStack
    LLM --> Viva
    
    Summary --> Faculty
    TechStack --> Faculty
    Viva --> Faculty
    
    Faculty --> Export
