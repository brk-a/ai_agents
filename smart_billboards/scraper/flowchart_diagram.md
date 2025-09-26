flowchart TD
    UI["User Input / API Layer"]
    Orchestrator["Scraping Orchestrator"]
    ProxyMgr["Proxy Manager"]
    Scraper["AI-Powered Scraper<br/>(Headless Browser + AI/ML)"]
    DataProc["Data Processing & Analytics"]
    Decision["Decision Engine"]
    API["API Endpoint"]
    Billboard["Smart Billboard Controller"]
    Monitor["Monitoring & Feedback Loop"]

    UI --> Orchestrator
    Orchestrator --> ProxyMgr
    ProxyMgr --> Scraper
    Scraper --> DataProc
    DataProc --> Decision
    Decision --> API
    API --> Billboard
    Monitor --> Orchestrator
    Monitor --> Scraper
    Monitor --> Decision
