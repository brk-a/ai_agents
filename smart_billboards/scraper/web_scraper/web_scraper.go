package web_scraper

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/PuerkitoBio/goquery"
	"github.com/gocolly/colly/v2"
	"github.com/gocolly/colly/v2/proxy"
)

type MockSolver struct {
	ShouldFail bool
	Timeout    bool
}

type Scraper interface {
	RunScraper(ctx context.Context, url string, params map[string]string, solver CaptchaSolver, allowedDomains []string, proxyAddr, userAgent string) ([]Product, error)
}

type RealSolver struct {
    APIKey       string
    APIURL       string
    ResultAPIURL string
}

type RealScraper struct{}

// Product represents the scraped product data
type Product struct {
	Name  string
	Price string
	Link  string
}

// CaptchaSolver defines an interface for solving CAPTCHAs
type CaptchaSolver interface {
	SubmitCaptcha(ctx context.Context, captchaType, siteKey, pageURL, imageBase64 string) (string, error)
}

func (m *MockSolver) SubmitCaptcha(ctx context.Context, captchaType, siteKey, pageURL, imageBase64 string) (string, error) {
	if m.Timeout {
		return "", errors.New("CAPTCHA solving timed out")
	}
	if m.ShouldFail {
		return "", errors.New("failed to solve CAPTCHA")
	}
	return "mocked-captcha-token", nil
}

// DetectCaptchaType detects the CAPTCHA type in the given DOM selection
func DetectCaptchaType(sel *goquery.Selection) string {
	if sel.Find("div#g-recaptcha").Length() > 0 {
		return "recaptcha_v2"
	}
	if sel.Find("div.h-captcha[data-sitekey]").Length() > 0 {
		return "hcaptcha"
	}
	if sel.Find("img.captcha-image").Length() > 0 || sel.Find("input[name='captcha']").Length() > 0 {
		return "image_captcha"
	}
	return ""
}

func (r *RealScraper) RunScraper(ctx context.Context, url string, params map[string]string, solver CaptchaSolver, allowedDomains []string, proxyAddr, userAgent string) ([]Product, error) {
	return RunScraper(ctx, url, params, solver, allowedDomains, proxyAddr, userAgent)
}

// RunScraper performs scraping with multi-CAPTCHA support and dependency injection
func RunScraper(
	ctx context.Context,
	url string,
	params map[string]string,
	solver CaptchaSolver,
	allowedDomains []string,
	proxyAddr string,
	userAgent string,
) ([]Product, error) {
	var products []Product
	var captchaSolved bool

	c := colly.NewCollector(
		colly.Async(true),
		colly.AllowedDomains(allowedDomains...),
		colly.AllowURLRevisit(),
	)

	// Rate limiting
	c.Limit(&colly.LimitRule{
		DomainGlob:  "*",
		Parallelism: 2,
		Delay:       1 * time.Second,
	})

	// Proxy support
	if proxyAddr != "" {
		rp, err := proxy.RoundRobinProxySwitcher(proxyAddr)
		if err != nil {
			return nil, fmt.Errorf("proxy setup failed: %w", err)
		}
		c.SetProxyFunc(rp)
	}

	// User-Agent
	c.OnRequest(func(r *colly.Request) {
		if userAgent != "" {
			r.Headers.Set("User-Agent", userAgent)
		}
		fmt.Println("Visiting", r.URL.String())
	})

	// CAPTCHA and product extraction
	c.OnHTML("body", func(e *colly.HTMLElement) {
		if captchaSolved {
			return
		}
		captchaType := DetectCaptchaType(e.DOM)
		if captchaType == "" {
			return
		}
		fmt.Println("CAPTCHA detected of type:", captchaType)

		var token string
		var err error

		switch captchaType {
		case "recaptcha_v2", "hcaptcha":
			siteKey := e.DOM.Find("div[data-sitekey]").AttrOr("data-sitekey", "")
			if siteKey == "" {
				fmt.Printf("No sitekey found for %s\n", captchaType)
				return
			}
			token, err = solver.SubmitCaptcha(ctx, captchaType, siteKey, url, "")
		case "image_captcha":
			imgSrc := e.DOM.Find("img.captcha-image").AttrOr("src", "")
			if imgSrc == "" {
				fmt.Println("No CAPTCHA image found")
				return
			}
			resp, err := http.Get(imgSrc)
			if err != nil {
				fmt.Println("Failed to download CAPTCHA image:", err)
				return
			}
			defer resp.Body.Close()
			imgBytes, err := io.ReadAll(resp.Body)
			if err != nil {
				fmt.Println("Failed to read CAPTCHA image:", err)
				return
			}
			imageBase64 := base64.StdEncoding.EncodeToString(imgBytes)
			token, err = solver.SubmitCaptcha(ctx, captchaType, "", "", imageBase64)
		default:
			fmt.Println("Unsupported CAPTCHA type:", captchaType)
			return
		}

		if err != nil {
			fmt.Println("CAPTCHA solving failed:", err)
			return
		}

		fmt.Println("CAPTCHA solved, token:", token)
		captchaSolved = true

		// Example: POST token to the same URL (site-specific logic may vary)
		r := e.Request
		r.Headers.Set("Content-Type", "application/x-www-form-urlencoded")
		formData := fmt.Sprintf("g-recaptcha-response=%s", token)
		err = r.PostRaw(r.URL.String(), []byte(formData))
		if err != nil {
			fmt.Println("Failed to resubmit request with CAPTCHA token:", err)
		}
	})

	c.OnError(func(r *colly.Response, err error) {
		fmt.Printf("Request URL: %s failed with status %d: %v\n", r.Request.URL, r.StatusCode, err)
	})

	c.OnHTML("div.product", func(e *colly.HTMLElement) {
		name := e.ChildText("h2.product-name")
		price := e.ChildText("span.price")
		link := e.ChildAttr("a.product-link", "href")
		products = append(products, Product{Name: name, Price: price, Link: link})
	})

	// Start scraping
	err := c.Visit(url)
	if err != nil {
		return nil, err
	}
	c.Wait()
	return products, nil
}

func NewRealSolver(apiKey string) *RealSolver {
    return &RealSolver{
        APIKey:       apiKey,
        APIURL:       "http://2captcha.com/in.php",
        ResultAPIURL: "http://2captcha.com/res.php",
    }
}

func (rs *RealSolver) SubmitCaptcha(ctx context.Context, captchaType, siteKey, pageURL, imageBase64 string) (string, error) {
	form := map[string][]string{
		"key":  {rs.APIKey},
		"json": {"1"},
	}

	switch captchaType {
	case "recaptcha_v2":
		form["method"] = []string{"userrecaptcha"}
		form["googlekey"] = []string{siteKey}
		form["pageurl"] = []string{pageURL}
	case "hcaptcha":
		form["method"] = []string{"hcaptcha"}
		form["sitekey"] = []string{siteKey}
		form["pageurl"] = []string{pageURL}
	case "image_captcha":
		form["method"] = []string{"base64"}
		form["body"] = []string{imageBase64}
	default:
		return "", fmt.Errorf("unsupported CAPTCHA type: %s", captchaType)
	}

	// Submit CAPTCHA
	resp, err := http.PostForm(rs.APIURL, form)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var submitResp struct {
		Status  int    `json:"status"`
		Request string `json:"request"`
	}
	body, _ := io.ReadAll(resp.Body)
	if err := json.Unmarshal(body, &submitResp); err != nil {
		return "", err
	}
	if submitResp.Status != 1 {
		return "", errors.New("failed to submit CAPTCHA to solver: " + submitResp.Request)
	}

	// Poll for solution (up to 2 minutes)
	for i := 0; i < 24; i++ {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(5 * time.Second):
			resResp, err := http.Get(fmt.Sprintf("%s?key=%s&action=get&id=%s&json=1", rs.ResultAPIURL, rs.APIKey, submitResp.Request))
			if err != nil {
				return "", err
			}
			defer resResp.Body.Close()

			var resData struct {
				Status  int    `json:"status"`
				Request string `json:"request"`
			}
			resBody, _ := io.ReadAll(resResp.Body)
			if err := json.Unmarshal(resBody, &resData); err != nil {
				return "", err
			}
			if resData.Status == 1 {
				return resData.Request, nil // CAPTCHA solved token
			}
			if resData.Request != "CAPCHA_NOT_READY" {
				return "", errors.New("error solving CAPTCHA: " + resData.Request)
			}
		}
	}
	return "", errors.New("CAPTCHA solving timed out")
}
