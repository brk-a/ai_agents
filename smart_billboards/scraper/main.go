package main

import (
	"context"
	"fmt"

	orchestrator "github.com/brk-a/scraper/orchestrator"
	web_scraper "github.com/brk-a/scraper/web_scraper"
)

func main() {
	// Create channels for jobs and results
	jobQueue := make(chan orchestrator.ScrapeJob, 10)
	results := make(chan web_scraper.Product, 10)

	workerCount := 3
	maxRetries := 2

	// Create a context with cancellation (optional timeout)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Instantiate the real scraper and solver implementations
	scraper := &web_scraper.RealScraper{}
	solver := &web_scraper.RealSolver{
		APIKey: "YOUR_2CAPTCHA_API_KEY", // Replace with your actual API key
	}

	// Create the orchestrator with injected dependencies and config
	orchestratorInstance := orchestrator.Orchestrator{
		Scraper:       scraper,
		Solver:        solver,
		AllowedDomains: []string{"example.com"},
		ProxyAddr:     "", // e.g., "http://your-proxy:port" or empty for none
		UserAgent:     "Mozilla/5.0 (compatible; MyScraperBot/1.0)",
	}

	// Start orchestrator workers
	go orchestratorInstance.Orchestrate(ctx, jobQueue, results, workerCount, maxRetries)

	// Enqueue scraping jobs
	jobQueue <- orchestrator.ScrapeJob{URL: "https://example.com/products", Params: map[string]string{"category": "tech"}}
	jobQueue <- orchestrator.ScrapeJob{URL: "https://example.com/deals", Params: map[string]string{"category": "deals"}}
	close(jobQueue) // no more jobs

	// Collect and print results as they come in
	for product := range results {
		fmt.Printf("Scraped product: %s | %s | %s\n", product.Name, product.Price, product.Link)
	}

	fmt.Println("All scraping jobs completed.")
}

