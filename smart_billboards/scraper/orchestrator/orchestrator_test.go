package orchestrator

import (
	"context"
	"errors"
	"testing"
	"time"

	web_scraper "github.com/brk-a/scraper/web_scraper"
)

type MockScraper struct{}

// mockRunScraper simulates web_scraper.RunScraper behavior for testing
var mockRunScraper = func(ctx context.Context, url string, params map[string]string, solver web_scraper.CaptchaSolver, allowedDomains []string, proxyAddr, userAgent string) ([]web_scraper.Product, error) {
	if url == "fail" {
		return nil, errors.New("simulated failure")
	}
	return []web_scraper.Product{
		{Name: "Test Product", Price: "$1", Link: url + "/prod"},
	}, nil
}

// Mock CAPTCHA solver for orchestrator tests
type MockSolver struct {
	ShouldFail bool
}

func (m *MockSolver) SubmitCaptcha(ctx context.Context, captchaType, siteKey, pageURL, imageBase64 string) (string, error) {
	if m.ShouldFail {
		return "", errors.New("mock solver failure")
	}
	return "mock-token", nil
}

func (m *MockScraper) RunScraper(ctx context.Context, url string, params map[string]string, solver web_scraper.CaptchaSolver, allowedDomains []string, proxyAddr, userAgent string) ([]web_scraper.Product, error) {
	if url == "fail" {
		return nil, errors.New("simulated failure")
	}
	return []web_scraper.Product{
		{Name: "Test Product", Price: "$1", Link: url + "/prod"},
	}, nil
}

func TestOrchestrate(t *testing.T) {
	jobQueue := make(chan ScrapeJob, 5)
	results := make(chan web_scraper.Product, 5)
	workerCount := 2
	maxRetries := 1

	mockSolver := &MockSolver{}
	mockScraper := &MockScraper{}

	orchestrator := Orchestrator{
		Scraper:       mockScraper,
		Solver:        mockSolver,
		AllowedDomains: []string{"example.com"},
		ProxyAddr:     "",
		UserAgent:     "test-agent",
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go orchestrator.Orchestrate(ctx, jobQueue, results, workerCount, maxRetries)

	jobQueue <- ScrapeJob{URL: "success", Params: nil}
	jobQueue <- ScrapeJob{URL: "fail", Params: nil}
	close(jobQueue)

	collected := []web_scraper.Product{}
	timeout := time.After(5 * time.Second)

loop:
	for {
		select {
		case p, ok := <-results:
			if !ok {
				break loop
			}
			collected = append(collected, p)
		case <-timeout:
			t.Fatal("Test timed out waiting for results")
		}
	}

	if len(collected) != 1 {
		t.Errorf("Expected 1 successful product, got %d", len(collected))
	}
	if collected[0].Name != "Test Product" {
		t.Errorf("Unexpected product name: %s", collected[0].Name)
	}
}

func TestOrchestrateWithCaptchaFailures(t *testing.T) {
	jobQueue := make(chan ScrapeJob, 5)
	results := make(chan web_scraper.Product, 5)
	workerCount := 2
	maxRetries := 1

	mockSolver := &MockSolver{ShouldFail: true}
	mockScraper := &MockScraper{}
	orchestrator := Orchestrator{
		Scraper: mockScraper,
		Solver:         mockSolver,
		AllowedDomains: []string{"example.com"},
		ProxyAddr:      "",
		UserAgent:      "test-agent",
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go orchestrator.Orchestrate(ctx, jobQueue, results, workerCount, maxRetries)

	jobQueue <- ScrapeJob{URL: "success", Params: nil}
	jobQueue <- ScrapeJob{URL: "fail", Params: nil}
	close(jobQueue)

	collected := []web_scraper.Product{}
	timeout := time.After(5 * time.Second)

loop:
	for {
		select {
		case p, ok := <-results:
			if !ok {
				break loop
			}
			collected = append(collected, p)
		case <-timeout:
			t.Fatal("Test timed out waiting for results")
		}
	}

	if len(collected) == 0 {
		t.Errorf("Expected some successful products, got none")
	}
}
