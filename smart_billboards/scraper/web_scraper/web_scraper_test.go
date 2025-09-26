package web_scraper

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/PuerkitoBio/goquery"
)

// --- CAPTCHA Detection Tests ---

func TestDetectCaptchaTypeWithGoquery(t *testing.T) {
	tests := []struct {
		name     string
		html     string
		expected string
	}{
		{"reCAPTCHA v2", `<div id="g-recaptcha" data-sitekey="sitekey"></div>`, "recaptcha_v2"},
		{"hCaptcha", `<div class="h-captcha" data-sitekey="sitekey"></div>`, "hcaptcha"},
		{"Image CAPTCHA", `<img class="captcha-image" src="captcha.jpg"/><input name="captcha" />`, "image_captcha"},
		{"No CAPTCHA", `<div>No captcha here</div>`, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			doc, err := goquery.NewDocumentFromReader(strings.NewReader(tt.html))
			if err != nil {
				t.Fatalf("Failed to parse HTML: %v", err)
			}
			captchaType := DetectCaptchaType(doc.Selection)
			if captchaType != tt.expected {
				t.Errorf("Expected %q, got %q", tt.expected, captchaType)
			}
		})
	}
}

// --- Mock CAPTCHA Solver for Testing ---

func (m *MockSolver) TestSubmitCaptcha(ctx context.Context, captchaType, siteKey, pageURL, imageBase64 string) (string, error) {
	if m.Timeout {
		// Simulate timeout by waiting and then returning error
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(2 * time.Second):
			return "", errors.New("CAPTCHA solving timed out")
		}
	}
	if m.ShouldFail {
		return "", errors.New("failed to solve CAPTCHA")
	}
	return "mocked-captcha-token", nil
}

// --- RunScraper Tests with Mock Solver ---

func TestRunScraperWithCaptcha(t *testing.T) {
	mockSolver := &MockSolver{}
	ctx := context.Background()

	// Test successful CAPTCHA solving
	products, err := RunScraper(ctx, "https://example.com/products", nil, mockSolver, []string{"example.com"}, "", "")
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if len(products) == 0 {
		t.Errorf("Expected products, got none")
	}

	// Test CAPTCHA solve failure
	mockSolver.ShouldFail = true
	_, err = RunScraper(ctx, "https://example.com/products", nil, mockSolver, []string{"example.com"}, "", "")
	if err == nil {
		t.Errorf("Expected error on CAPTCHA solve failure, got nil")
	}
	mockSolver.ShouldFail = false

	// Test CAPTCHA solve timeout
	mockSolver.Timeout = true
	_, err = RunScraper(ctx, "https://example.com/products", nil, mockSolver, []string{"example.com"}, "", "")
	if err == nil {
		t.Errorf("Expected error on CAPTCHA solve timeout, got nil")
	}
}
