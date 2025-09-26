package orchestrator

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	web_scraper "github.com/brk-a/scraper/web_scraper"
)

// ScrapeJob defines a scraping task
type ScrapeJob struct {
	URL        string
	Params     map[string]string
	RetryCount int
}

// Orchestrator manages scraping jobs and workers
type Orchestrator struct {
	Scraper       web_scraper.Scraper
	Solver        web_scraper.CaptchaSolver
	AllowedDomains []string
	ProxyAddr     string
	UserAgent     string
}

// Orchestrate runs the job queue with workerCount concurrent workers
func (o *Orchestrator) Orchestrate(ctx context.Context, jobQueue chan ScrapeJob, results chan web_scraper.Product, workerCount int, maxRetries int) {
	var wg sync.WaitGroup

	for i := 0; i < workerCount; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					fmt.Printf("[Worker %d] Context cancelled, exiting\n", workerID)
					return
				case job, ok := <-jobQueue:
					if !ok {
						return
					}
					fmt.Printf("[Worker %d] Processing job: %s (Retry %d)\n", workerID, job.URL, job.RetryCount)

					jobCtx, cancel := context.WithTimeout(ctx, 2*time.Minute)
					products, err := o.Scraper.RunScraper(jobCtx, job.URL, job.Params, o.Solver, o.AllowedDomains, o.ProxyAddr, o.UserAgent)
					cancel()

					if err != nil {
						log.Printf("[Worker %d] Error scraping %s: %v\n", workerID, job.URL, err)
						if job.RetryCount < maxRetries {
							job.RetryCount++
							fmt.Printf("[Worker %d] Requeuing job: %s (Retry %d)\n", workerID, job.URL, job.RetryCount)
							select {
							case jobQueue <- job:
							case <-ctx.Done():
								return
							}
						} else {
							log.Printf("[Worker %d] Max retries reached for %s. Skipping.\n", workerID, job.URL)
						}
					} else {
						for _, p := range products {
							select {
							case results <- p:
							case <-ctx.Done():
								return
							}
						}
					}
				}
			}
		}(i + 1)
	}

	wg.Wait()
	close(results)
}
