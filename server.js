// ═══════════════════════════════════════════════════════════════════════════
// ENHANCED STOCK PREDICTION API - WITH SENTIMENT ANALYSIS
// Integrates news sentiment, events, and technical analysis
// ═══════════════════════════════════════════════════════════════════════════

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3001;

// Enable CORS for frontend
app.use(cors());
app.use(express.json());
// Serve static frontend
const path = require('path');
app.use(express.static(__dirname));

// Serve index.html for root route
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

// IMPORTANT: Replace with your actual NewsAPI key
const NEWS_API_KEY = process.env.NEWS_API_KEY || '798868e200de40ceb20d89cdb678e82e';

const VALID_SYMBOLS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS',
    'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'ICICIBANK.NS'
];

// Company name mapping for news search
const COMPANY_NAMES = {
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'Tata Consultancy Services',
    'HDFCBANK.NS': 'HDFC Bank',
    'INFY.NS': 'Infosys',
    'SBIN.NS': 'State Bank of India',
    'BHARTIARTL.NS': 'Bharti Airtel',
    'ITC.NS': 'ITC Limited',
    'ICICIBANK.NS': 'ICICI Bank'
};

// Simple in-memory cache to reduce API calls
const newsCache = new Map();
const CACHE_DURATION = 3600000; // 1 hour in milliseconds

// ═══════════════════════════════════════════════════════════════════════════
// SENTIMENT ANALYSIS (VADER-like algorithm)
// ═══════════════════════════════════════════════════════════════════════════

const SENTIMENT_LEXICON = {
    // Positive words
    'profit': 3, 'growth': 2, 'rise': 2, 'gain': 2, 'beat': 2, 'strong': 2,
    'surge': 3, 'jump': 2, 'soar': 3, 'boom': 3, 'bullish': 2, 'success': 2,
    'positive': 2, 'excellent': 3, 'outstanding': 3, 'record': 2, 'high': 1,
    'up': 1, 'increase': 2, 'expansion': 2, 'upgrade': 2, 'buy': 1,
    
    // Negative words
    'loss': -3, 'fall': -2, 'decline': -2, 'drop': -2, 'plunge': -3,
    'crash': -3, 'bearish': -2, 'weak': -2, 'concern': -2, 'fear': -2,
    'risk': -1, 'down': -1, 'negative': -2, 'cut': -2, 'slash': -2,
    'recession': -3, 'crisis': -3, 'conflict': -2, 'war': -2, 'sell': -1,
    
    // Intensifiers
    'very': 1.5, 'extremely': 2, 'highly': 1.5, 'significantly': 1.5,
    
    // Diminishers
    'slightly': 0.5, 'barely': 0.5, 'somewhat': 0.7
};

const EVENT_KEYWORDS = {
    'geopolitical': ['war', 'conflict', 'sanction', 'tension', 'crisis'],
    'economic_policy': ['policy', 'regulation', 'reform', 'budget', 'tax'],
    'market_event': ['earnings', 'dividend', 'merger', 'acquisition', 'ipo'],
    'sector_trend': ['oil', 'tech', 'banking', 'pharma', 'auto'],
    'global_event': ['fed', 'interest rate', 'inflation', 'recession', 'gdp']
};

/**
 * Analyze sentiment of text using VADER-like approach
 */
function analyzeSentiment(text) {
    if (!text) return { score: 0, magnitude: 0, label: 'neutral' };
    
    const words = text.toLowerCase().split(/\s+/);
    let score = 0;
    let magnitude = 0;
    let intensifier = 1;
    
    for (let i = 0; i < words.length; i++) {
        const word = words[i].replace(/[^\w]/g, '');
        
        // Check for intensifiers/diminishers
        if (SENTIMENT_LEXICON[word] && (word === 'very' || word === 'extremely' || 
            word === 'highly' || word === 'significantly' || word === 'slightly' || 
            word === 'barely' || word === 'somewhat')) {
            intensifier = SENTIMENT_LEXICON[word];
            continue;
        }
        
        // Calculate sentiment
        if (SENTIMENT_LEXICON[word]) {
            const wordScore = SENTIMENT_LEXICON[word] * intensifier;
            score += wordScore;
            magnitude += Math.abs(wordScore);
            intensifier = 1; // Reset
        }
    }
    
    // Normalize score
    const normalizedScore = magnitude > 0 ? score / magnitude : 0;
    
    // Determine label
    let label = 'neutral';
    if (normalizedScore > 0.2) label = 'positive';
    else if (normalizedScore < -0.2) label = 'negative';
    
    return {
        score: parseFloat(normalizedScore.toFixed(3)),
        magnitude: parseFloat(magnitude.toFixed(2)),
        label: label
    };
}

/**
 * Detect events mentioned in text
 */
function detectEvents(text) {
    const detectedEvents = [];
    const lowerText = text.toLowerCase();
    
    for (const [category, keywords] of Object.entries(EVENT_KEYWORDS)) {
        for (const keyword of keywords) {
            if (lowerText.includes(keyword)) {
                detectedEvents.push({
                    category: category,
                    keyword: keyword,
                    impact: getSentimentImpact(keyword)
                });
                break; // One event per category
            }
        }
    }
    
    return detectedEvents;
}

/**
 * Estimate impact of specific events
 */
function getSentimentImpact(keyword) {
    const impacts = {
        // Positive
        'earnings': 0.15, 'dividend': 0.10, 'merger': 0.12, 'acquisition': 0.10,
        'reform': 0.08, 'budget': 0.05,
        
        // Negative
        'war': -0.20, 'conflict': -0.15, 'sanction': -0.18, 'crisis': -0.20,
        'recession': -0.25, 'inflation': -0.10,
        
        // Neutral/Variable
        'policy': 0, 'regulation': -0.05, 'tax': -0.03,
        'fed': 0, 'interest rate': -0.05, 'oil': 0.05
    };
    
    return impacts[keyword] || 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// NEWS FETCHING
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Fetch news from NewsAPI with caching
 */
async function fetchNews(companyName, symbol) {
    // Check cache first
    const cacheKey = `news_${symbol}`;
    const cached = newsCache.get(cacheKey);
    
    if (cached && (Date.now() - cached.timestamp < CACHE_DURATION)) {
        console.log(`Using cached news for ${symbol}`);
        return cached.data;
    }
    
    try {
        const url = 'https://newsapi.org/v2/everything';
        const params = {
            q: `${companyName} stock`,
            language: 'en',
            sortBy: 'publishedAt',
            pageSize: 10,
            apiKey: NEWS_API_KEY
        };
        
        const response = await axios.get(url, { params, timeout: 5000 });
        
        if (response.data && response.data.articles) {
            const newsData = {
                articles: response.data.articles,
                totalResults: response.data.totalResults,
                fetchedAt: new Date().toISOString()
            };
            
            // Cache the result
            newsCache.set(cacheKey, {
                data: newsData,
                timestamp: Date.now()
            });
            
            return newsData;
        }
        
        throw new Error('Invalid news API response');
        
    } catch (error) {
        console.error(`Error fetching news for ${symbol}:`, error.message);
        
        // Return mock data if API fails
        return {
            articles: generateMockNews(companyName),
            totalResults: 5,
            fetchedAt: new Date().toISOString(),
            isMock: true
        };
    }
}

/**
 * Generate mock news when API fails (fallback)
 */
function generateMockNews(companyName) {
    const mockTemplates = [
        `${companyName} reports strong quarterly earnings`,
        `Analysts upgrade ${companyName} stock rating`,
        `${companyName} announces expansion plans`,
        `Market volatility affects ${companyName} shares`,
        `${companyName} faces regulatory scrutiny`
    ];
    
    return mockTemplates.slice(0, 3).map((title, i) => ({
        title: title,
        description: `Latest updates on ${companyName} stock performance and market position.`,
        publishedAt: new Date(Date.now() - i * 86400000).toISOString(),
        source: { name: 'Market News' },
        url: '#'
    }));
}

/**
 * Analyze news sentiment and extract insights
 */
function analyzeNewsSentiment(newsData) {
    if (!newsData.articles || newsData.articles.length === 0) {
        return {
            overallSentiment: 0,
            sentimentLabel: 'neutral',
            newsCount: 0,
            recentNews: [],
            events: [],
            sentimentBreakdown: { positive: 0, neutral: 0, negative: 0 }
        };
    }
    
    const articles = newsData.articles.slice(0, 10); // Analyze top 10
    let totalScore = 0;
    let sentimentBreakdown = { positive: 0, neutral: 0, negative: 0 };
    const allEvents = [];
    const analyzedNews = [];
    
    articles.forEach(article => {
        const text = `${article.title} ${article.description || ''}`;
        const sentiment = analyzeSentiment(text);
        const events = detectEvents(text);
        
        totalScore += sentiment.score;
        sentimentBreakdown[sentiment.label]++;
        allEvents.push(...events);
        
        analyzedNews.push({
            title: article.title,
            sentiment: sentiment,
            events: events,
            publishedAt: article.publishedAt,
            source: article.source.name,
            url: article.url
        });
    });
    
    const avgSentiment = totalScore / articles.length;
    
    return {
        overallSentiment: parseFloat(avgSentiment.toFixed(3)),
        sentimentLabel: avgSentiment > 0.1 ? 'positive' : avgSentiment < -0.1 ? 'negative' : 'neutral',
        newsCount: articles.length,
        recentNews: analyzedNews.slice(0, 5),
        events: allEvents,
        sentimentBreakdown: sentimentBreakdown,
        isMock: newsData.isMock || false
    };
}

// ═══════════════════════════════════════════════════════════════════════════
// STOCK DATA FETCHING (from previous version)
// ═══════════════════════════════════════════════════════════════════════════

async function fetchYahooFinanceData(symbol) {
    try {
        const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}`;
        
        const response = await axios.get(url, {
            params: {
                range: '6mo',
                interval: '1d',
                includePrePost: false,
                events: 'history'
            },
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            },
            timeout: 10000
        });

        if (!response.data?.chart?.result?.[0]) {
            throw new Error('Invalid response from Yahoo Finance');
        }

        const result = response.data.chart.result[0];
        const timestamps = result.timestamp;
        const quotes = result.indicators.quote[0];

        const data = [];
        for (let i = 0; i < timestamps.length; i++) {
            const close = quotes.close[i];
            if (close === null || isNaN(close)) continue;

            data.push({
                date: new Date(timestamps[i] * 1000).toISOString().split('T')[0],
                timestamp: timestamps[i],
                open: parseFloat(quotes.open[i]?.toFixed(2) || close.toFixed(2)),
                high: parseFloat(quotes.high[i]?.toFixed(2) || close.toFixed(2)),
                low: parseFloat(quotes.low[i]?.toFixed(2) || close.toFixed(2)),
                close: parseFloat(close.toFixed(2)),
                volume: quotes.volume[i] || 0
            });
        }

        return {
            symbol,
            currency: result.meta.currency || 'INR',
            timezone: result.meta.timezone || 'IST',
            data,
            dataPoints: data.length,
            isReal: true
        };

    } catch (error) {
        console.error(`Error fetching data for ${symbol}:`, error.message);
        throw error;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TECHNICAL INDICATORS (from previous version)
// ═══════════════════════════════════════════════════════════════════════════

function calculateTechnicalIndicators(data) {
    const closes = data.map(d => d.close);
    
    function calculateSMA(prices, period) {
        const sma = [];
        for (let i = period - 1; i < prices.length; i++) {
            const sum = prices.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
            sma.push(sum / period);
        }
        return sma;
    }

    function calculateEMA(prices, period) {
        const ema = [];
        const multiplier = 2 / (period + 1);
        let previousEMA = prices.slice(0, period).reduce((a, b) => a + b, 0) / period;
        ema.push(previousEMA);

        for (let i = period; i < prices.length; i++) {
            const currentEMA = (prices[i] - previousEMA) * multiplier + previousEMA;
            ema.push(currentEMA);
            previousEMA = currentEMA;
        }
        return ema;
    }

    function calculateRSI(prices, period = 14) {
        const changes = [];
        for (let i = 1; i < prices.length; i++) {
            changes.push(prices[i] - prices[i - 1]);
        }

        const rsi = [];
        let gains = 0, losses = 0;

        for (let i = 0; i < period; i++) {
            if (changes[i] > 0) gains += changes[i];
            else losses -= changes[i];
        }

        let avgGain = gains / period;
        let avgLoss = losses / period;

        for (let i = period; i < changes.length; i++) {
            const rs = avgGain / avgLoss;
            rsi.push(100 - (100 / (1 + rs)));

            const change = changes[i];
            avgGain = ((avgGain * (period - 1)) + (change > 0 ? change : 0)) / period;
            avgLoss = ((avgLoss * (period - 1)) + (change < 0 ? -change : 0)) / period;
        }

        return rsi;
    }

    function calculateMACD(prices) {
        const ema12 = calculateEMA(prices, 12);
        const ema26 = calculateEMA(prices, 26);
        
        const macdLine = [];
        const minLength = Math.min(ema12.length, ema26.length);
        
        for (let i = 0; i < minLength; i++) {
            macdLine.push(ema12[ema12.length - minLength + i] - ema26[ema26.length - minLength + i]);
        }

        const signalLine = calculateEMA(macdLine, 9);
        
        return {
            macd: macdLine[macdLine.length - 1],
            signal: signalLine[signalLine.length - 1],
            histogram: macdLine[macdLine.length - 1] - signalLine[signalLine.length - 1]
        };
    }

    function calculateBollingerBands(prices, period = 20, stdDev = 2) {
        const sma = calculateSMA(prices, period);
        const bands = [];

        for (let i = 0; i < sma.length; i++) {
            const slice = prices.slice(i, i + period);
            const mean = sma[i];
            const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period;
            const std = Math.sqrt(variance);

            bands.push({
                upper: mean + (stdDev * std),
                middle: mean,
                lower: mean - (stdDev * std)
            });
        }

        return bands[bands.length - 1];
    }

    const sma20 = calculateSMA(closes, 20);
    const sma50 = calculateSMA(closes, 50);
    const ema12 = calculateEMA(closes, 12);
    const rsi = calculateRSI(closes, 14);
    const macd = calculateMACD(closes);
    const bollinger = calculateBollingerBands(closes, 20, 2);

    return {
        currentPrice: closes[closes.length - 1],
        sma20: sma20[sma20.length - 1],
        sma50: sma50[sma50.length - 1],
        ema12: ema12[ema12.length - 1],
        rsi: rsi[rsi.length - 1],
        macd: macd,
        bollingerBands: bollinger,
        trend: sma20[sma20.length - 1] > sma50[sma50.length - 1] ? 'bullish' : 'bearish',
        momentum: rsi[rsi.length - 1] > 50 ? 'positive' : 'negative'
    };
}

// ═══════════════════════════════════════════════════════════════════════════
// API ENDPOINTS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * GET /api/stock/:symbol
 * Fetch stock data with sentiment analysis
 */
app.get('/api/stock/:symbol', async (req, res) => {
    try {
        const symbol = req.params.symbol.toUpperCase();

        if (!VALID_SYMBOLS.includes(symbol)) {
            return res.status(400).json({
                error: 'Invalid stock symbol',
                validSymbols: VALID_SYMBOLS
            });
        }

        // Fetch stock data
        const stockData = await fetchYahooFinanceData(symbol);

        // Calculate technical indicators
        const indicators = calculateTechnicalIndicators(stockData.data);

        // Fetch and analyze news
        const companyName = COMPANY_NAMES[symbol];
        const newsData = await fetchNews(companyName, symbol);
        const sentimentAnalysis = analyzeNewsSentiment(newsData);

        // Return combined response
        res.json({
            success: true,
            symbol: stockData.symbol,
            companyName: companyName,
            currency: stockData.currency,
            lastUpdate: new Date().toISOString(),
            dataPoints: stockData.dataPoints,
            data: stockData.data,
            indicators: indicators,
            sentiment: sentimentAnalysis,
            isReal: stockData.isReal
        });

    } catch (error) {
        console.error('API Error:', error.message);
        res.status(500).json({
            success: false,
            error: 'Failed to fetch stock data',
            message: error.message
        });
    }
});

/**
 * GET /api/health
 */
app.get('/api/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        newsApiConfigured: NEWS_API_KEY !== 'YOUR_NEWSAPI_KEY_HERE'
    });
});

/**
 * GET /api/symbols
 */
app.get('/api/symbols', (req, res) => {
    res.json({
        symbols: VALID_SYMBOLS,
        companies: COMPANY_NAMES,
        count: VALID_SYMBOLS.length
    });
});

// ═══════════════════════════════════════════════════════════════════════════
// ERROR HANDLING
// ═══════════════════════════════════════════════════════════════════════════

app.use((req, res) => {
    res.status(404).json({
        error: 'Endpoint not found',
        availableEndpoints: [
            'GET /api/stock/:symbol',
            'GET /api/health',
            'GET /api/symbols'
        ]
    });
});

app.use((err, req, res, next) => {
    console.error('Unhandled error:', err);
    res.status(500).json({
        error: 'Internal server error',
        message: err.message
    });
});

// ═══════════════════════════════════════════════════════════════════════════
// START SERVER
// ═══════════════════════════════════════════════════════════════════════════

app.listen(PORT, () => {
    console.log(`
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     📈 Stock Prediction API - SENTIMENT ANALYSIS ENABLED     ║
║                                                              ║
║  Server running on: http://localhost:${PORT}                 ║
║                                                              ║
║  Features:                                                   ║
║    ✅ Real-time stock data (Yahoo Finance)                  ║
║    ✅ News sentiment analysis (NewsAPI)                     ║
║    ✅ Event detection & tracking                            ║
║    ✅ Technical indicators                                  ║
║                                                              ║
║  NewsAPI Status: ${NEWS_API_KEY !== 'YOUR_NEWSAPI_KEY_HERE' ? '✅ Configured' : '⚠️  NOT CONFIGURED'}                              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    `);
    
    if (NEWS_API_KEY === 'YOUR_NEWSAPI_KEY_HERE') {
        console.log(`
⚠️  WARNING: NewsAPI key not configured!
   
   To enable news sentiment analysis:
   1. Set NEWS_API_KEY environment variable, OR
   2. Edit server.js line 18 and replace YOUR_NEWSAPI_KEY_HERE
   
   Until then, mock news data will be used.
`);
    }
});

process.on('SIGTERM', () => {
    console.log('SIGTERM received, shutting down gracefully...');
    process.exit(0);
});

process.on('SIGINT', () => {
    console.log('\nSIGINT received, shutting down gracefully...');
    process.exit(0);
});
