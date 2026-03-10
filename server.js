const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { spawn } = require('child_process');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(__dirname));

// Cache for news
const newsCache = new Map();
const CACHE_DURATION = 3600000; // 1 hour

// Cache for Yahoo stock data
const stockCache = new Map();
const STOCK_CACHE_DURATION = 1800000; // 30 min

// Cache for Python predictions (always 30 days, sliced per request)
const predCache = new Map();
const PRED_CACHE_DURATION = 1800000; // 30 min — matches stock cache

const NEWS_API_KEY = process.env.NEWS_API_KEY || '';

const YAHOO_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
};

async function fetchYahoo(symbol) {
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?range=1y&interval=1d`;

    try {
        const res = await axios.get(url, { timeout: 10000, headers: YAHOO_HEADERS });
        return res;
    } catch (err) {
        if (err.response?.status === 429) {
            console.warn('Yahoo 429 — retrying after 2s...');
            await new Promise(r => setTimeout(r, 2000));
            return axios.get(url, { timeout: 10000, headers: YAHOO_HEADERS });
        }
        throw err;
    }
}

function analyzeSentiment(text) {
    const positive = ['gain','growth','profit','surge','strong','positive','bullish','rise','beat','up'];
    const negative = ['loss','drop','fall','decline','weak','negative','bearish','miss','risk','down'];

    let score = 0;
    const lower = text.toLowerCase();

    positive.forEach(w => { if (lower.includes(w)) score += 1; });
    negative.forEach(w => { if (lower.includes(w)) score -= 1; });

    if (/(not|no|never|n't).*?(good|strong|beat|profit|gain)/i.test(lower)) {
        score *= -0.6;
    }

    return score;
}

function summarizeSentiment(articles) {
    if (!articles?.length) {
        return { overallSentiment: 0, sentimentLabel: "neutral", newsCount: 0, recentNews: [] };
    }

    // Deduplicate by title (NewsAPI sometimes returns same article with different URLs)
    const seen = new Set();
    articles = articles.filter(a => {
        const key = (a.title || '').trim().toLowerCase();
        if (!key || seen.has(key)) return false;
        seen.add(key);
        return true;
    });

    let totalScore = 0;
    const processed = articles.map(a => {
        const text = (a.title || '') + ' ' + (a.description || '');
        const score = analyzeSentiment(text);
        totalScore += score;

        return {
            title: a.title,
            source: a.source?.name || "Unknown",
            publishedAt: a.publishedAt,
            url: a.url,
            description: a.description,
            sentiment: {
                score,
                label: score > 0 ? "positive" : score < 0 ? "negative" : "neutral"
            }
        };
    });

    const avg = totalScore / articles.length;
    const normalized = Math.max(-1, Math.min(1, avg / 4));

    return {
        overallSentiment: normalized,
        sentimentLabel: normalized > 0.2 ? "positive" : normalized < -0.2 ? "negative" : "neutral",
        newsCount: articles.length,
        recentNews: processed
    };
}

function countTradingDays(fromDate, toDate) {
    let count = 0;
    const cur = new Date(fromDate);
    cur.setHours(0, 0, 0, 0);
    const end = new Date(toDate);
    end.setHours(0, 0, 0, 0);
    while (cur < end) {
        cur.setDate(cur.getDate() + 1);
        const day = cur.getDay();
        if (day !== 0 && day !== 6) count++;
    }
    return count;
}

function runPython(historical, daysToRequest) {
    return new Promise((resolve, reject) => {
        const pythonCmd = process.platform === 'win32' ? 'py' : 'python3';
        const py = spawn(pythonCmd, ['predictor.py', daysToRequest.toString()]);

        let pyData = '';
        let pyError = '';

        py.stdout.on('data', d => pyData += d.toString());
        py.stderr.on('data', d => pyError += d.toString());

        py.on('error', err => reject(new Error('Failed to start Python: ' + err.message)));

        py.on('close', (code) => {
            if (code !== 0) return reject(new Error('Python error\n' + pyError.slice(-400)));
            if (!pyData.trim()) return reject(new Error('Empty response from model'));

            let prediction;
            try { prediction = JSON.parse(pyData); }
            catch (e) { return reject(new Error('Invalid prediction format')); }

            if (prediction.error) return reject(new Error(prediction.error));

            resolve(prediction.predictions);
        });

        py.stdin.write(JSON.stringify(historical));
        py.stdin.end();
    });
}

app.get('/api/stock/:symbol', async (req, res) => {
    const symbol = req.params.symbol;
    const cleanSymbol = symbol.split('.')[0];
    const days = Math.min(parseInt(req.query.days || 30) || 30, 365);

    try {
        // Check stock cache first to avoid hammering Yahoo
        let historical;
        const cachedStock = stockCache.get(symbol);
        if (cachedStock && Date.now() - cachedStock.timestamp < STOCK_CACHE_DURATION) {
            historical = cachedStock.historical;
            console.log(`Stock cache hit for ${symbol}`);
        } else {
            const stockResponse = await fetchYahoo(symbol);
            const result = stockResponse.data.chart?.result?.[0];
            if (!result?.timestamp) {
                return res.json({ success: false, error: "No chart data from Yahoo" });
            }

            const timestamps = result.timestamp;
            const closes = result.indicators.quote[0].close;

            historical = timestamps.map((t, i) => ({
                date: new Date(t * 1000).toISOString().split('T')[0],
                close: closes[i]
            })).filter(d => d.close !== null && !isNaN(d.close));

            if (historical.length < 20) {
                return res.json({ success: false, error: "Not enough historical data" });
            }

            stockCache.set(symbol, { historical, timestamp: Date.now() });
            console.log(`Stock cached for ${symbol}`);
        }

        // Work out how many trading days stale Yahoo's data is vs today
        const lastDataDate = new Date(historical[historical.length - 1].date);
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        const staleDays = Math.max(0, countTradingDays(lastDataDate, today));

        // Always request 30 days from Python so cache covers all horizons
        const MAX_DAYS = 30;
        const daysToRequest = MAX_DAYS + staleDays;

        // Check prediction cache
        let allPredictions;
        const cachedPred = predCache.get(symbol);
        if (cachedPred && Date.now() - cachedPred.timestamp < PRED_CACHE_DURATION) {
            allPredictions = cachedPred.predictions;
            console.log(`Prediction cache hit for ${symbol}`);
        } else {
            try {
                allPredictions = await runPython(historical, daysToRequest);
                predCache.set(symbol, { predictions: allPredictions, timestamp: Date.now() });
                console.log(`Predictions cached for ${symbol}`);
            } catch (err) {
                return res.json({ success: false, error: err.message });
            }
        }

        // Slice to the requested horizon starting from today
        const mlPred = allPredictions.slice(staleDays, staleDays + days);

        let sentiment = { overallSentiment: 0, sentimentLabel: "neutral", newsCount: 0, recentNews: [] };

        const companyNames = {
            'RELIANCE': 'Reliance Industries',
            'TCS': 'Tata Consultancy Services',
            'HDFCBANK': 'HDFC Bank',
            'INFY': 'Infosys',
            'SBIN': 'State Bank of India',
            'BHARTIARTL': 'Bharti Airtel',
            'ITC': 'ITC Limited',
            'ICICIBANK': 'ICICI Bank'
        };
        const newsQuery = companyNames[cleanSymbol] || cleanSymbol;

        if (NEWS_API_KEY) {
            const cacheKey = cleanSymbol;
            const cached = newsCache.get(cacheKey);

            if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
                sentiment = cached.sentiment;
                console.log(`News cache hit for ${cleanSymbol}`);
            } else {
                try {
                    // TCS is more commonly referenced by ticker in news; others use full name
                    const tickerSearch = ['TCS', 'INFY'];
                    const searchQuery = tickerSearch.includes(cleanSymbol)
                        ? `"${cleanSymbol}"`
                        : companyNames[cleanSymbol]
                            ? `"${companyNames[cleanSymbol]}"`
                            : `"${cleanSymbol}"`;
                    const newsURL = `https://newsapi.org/v2/everything?q=${encodeURIComponent(searchQuery)}&language=en&sortBy=publishedAt&pageSize=6&apiKey=${NEWS_API_KEY}`;
                    const newsRes = await axios.get(newsURL, { timeout: 8000 });

                    const keywords = [cleanSymbol.toLowerCase(), (companyNames[cleanSymbol] || '').toLowerCase()].filter(Boolean);
                    const filtered = (newsRes.data.articles || []).filter(a => {
                        const title = (a.title || '').toLowerCase();
                        return keywords.some(k => k && title.includes(k));
                    }).slice(0, 6);
                    sentiment = summarizeSentiment(filtered);

                    newsCache.set(cacheKey, { sentiment, timestamp: Date.now() });
                    console.log(`News cached for ${cleanSymbol}`);
                } catch (newsErr) {
                    console.warn("News fetch failed:", newsErr.message);
                }
            }
        }

        const sentimentImpact = sentiment.overallSentiment;
        const sentimentAdjusted = mlPred.map((price, i) => {
            const decay = Math.exp(-i * 0.08);
            const factor = 1 + (sentimentImpact * 0.015 * decay);
            return Number((price * factor).toFixed(2));
        });

        res.json({
            success: true,
            data: historical,
            prediction: {
                ml: mlPred,
                sentimentAdjusted,
                trend: mlPred[mlPred.length - 1] > mlPred[0] ? "bullish" : "bearish"
            },
            sentiment
        });

    } catch (err) {
        console.error(err);
        const msg = err.response?.status ? `Yahoo error ${err.response.status}` : err.message;
        res.json({ success: false, error: msg });
    }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
