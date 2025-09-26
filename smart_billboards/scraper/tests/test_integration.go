package tests

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	orchestrator "github.com/brk-a/scraper/orchestrator"
	web_scraper "github.com/brk-a/scraper/web_scraper"
)

type MockSolver struct {
	ShouldFail bool
	Timeout    bool
}

func TestIntegration(t *testing.T) {
	// Setup mock HTTP server serving simple product page
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `
		<html><body>
		<div class="product">
			<h2 class="product-name">Test Product</h2>
			<span class="price">$9.99</span>
			<a class="product-link" href="/product1">Link</a>
		</div>
		</body></html>`)
	})
	server := httptest.NewServer(handler)
	defer server.Close()

	jobQueue := make(chan orchestrator.ScrapeJob, 2)
	results := make(chan web_scraper.Product, 2)
	workerCount := 1
	maxRetries := 1

	mockSolver := &web_scraper.MockSolver{} // Use mock solver from scraper package

	orchestratorInstance := orchestrator.Orchestrator{
		Solver:         mockSolver,
		AllowedDomains: []string{"localhost"},
		ProxyAddr:      "",
		UserAgent:      "IntegrationTestAgent/1.0",
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	go orchestratorInstance.Orchestrate(ctx, jobQueue, results, workerCount, maxRetries)

	jobQueue <- orchestrator.ScrapeJob{URL: server.URL, Params: nil}
	close(jobQueue)

	collected := []web_scraper.Product{}
	timeout := time.After(15 * time.Second)

loop:
	for {
		select {
		case p, ok := <-results:
			if !ok {
				break loop
			}
			collected = append(collected, p)
		case <-timeout:
			t.Fatal("Integration test timed out")
		}
	}

	if len(collected) != 1 {
		t.Fatalf("Expected 1 product, got %d", len(collected))
	}
	if collected[0].Name != "Test Product" {
		t.Errorf("Unexpected product name: %s", collected[0].Name)
	}
}
