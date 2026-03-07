const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { spawn } = require('child_process');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(__dirname));

// Simple in-memory cache for news
const newsCache = new Map();
const CACHE_DURATION = 3600000; // 1 hour

const NEWS_API_KEY = process.env.NEWS_API_KEY || '';

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

app.get('/api/stock/:symbol', async (req, res) => {
    const symbol = req.params.symbol;
    const cleanSymbol = symbol.split('.')[0];
    const days = Math.min(parseInt(req.query.days || 30) || 30, 365);

    let responded = false;

    try {
        const yahooURL = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?range=1y&interval=1d`;
        const stockResponse = await axios.get(yahooURL, { timeout: 10000 });

        const result = stockResponse.data.chart?.result?.[0];
        if (!result?.timestamp) {
            return res.json({ success: false, error: "No chart data from Yahoo" });
        }

        const timestamps = result.timestamp;
        const closes = result.indicators.quote[0].close;

        const historical = timestamps.map((t, i) => ({
            date: new Date(t * 1000).toISOString().split('T')[0],
            close: closes[i]
        })).filter(d => d.close !== null && !isNaN(d.close));

        if (historical.length < 20) {
            return res.json({ success: false, error: "Not enough historical data" });
        }

        // Work out how many trading days stale Yahoo's data is vs today
        const lastDataDate = new Date(historical[historical.length - 1].date);
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        const staleDays = countTradingDays(lastDataDate, today);

        // Ask Python for enough extra days to bridge the gap, so that
        // prediction[staleDays] is "tomorrow" from today's perspective
        const daysToRequest = days + staleDays;

        const pythonCmd = process.platform === 'win32' ? 'py' : 'python3';
        const py = spawn(pythonCmd, ['predictor.py', daysToRequest.toString()]);

        let pyData = '';
        let pyError = '';

        py.stdout.on('data', d => pyData += d.toString());
        py.stderr.on('data', d => pyError += d.toString());

        py.on('error', err => {
            if (responded) return;
            responded = true;
            console.error("Spawn error:", err);
            res.json({ success: false, error: "Failed to start Python" });
        });

        py.on('close', async (code) => {
            if (responded) return;
            responded = true;

            if (code !== 0) {
                console.error(`Python exited ${code}\n${pyError}`);
                return res.json({ success: false, error: "Python error\n" + pyError.slice(-400) });
            }

            if (!pyData.trim()) {
                return res.json({ success: false, error: "Empty response from model" });
            }

            let prediction;
            try {
                prediction = JSON.parse(pyData);
            } catch (e) {
                console.error("Invalid JSON:", pyData);
                return res.json({ success: false, error: "Invalid prediction format" });
            }

            if (prediction.error) {
                return res.json({ success: false, error: prediction.error });
            }

            // Slice off the stale-gap days so index 0 = tomorrow from today
            const mlPred = (prediction.predictions || []).slice(staleDays, staleDays + days);

            let sentiment = { overallSentiment: 0, sentimentLabel: "neutral", newsCount: 0, recentNews: [] };

            if (NEWS_API_KEY) {
                const cacheKey = cleanSymbol;
                const cached = newsCache.get(cacheKey);

                if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
                    sentiment = cached.sentiment;
                    console.log(`News cache hit for ${cleanSymbol}`);
                } else {
                    try {
                        const newsURL = `https://newsapi.org/v2/everything?q=${cleanSymbol}&language=en&sortBy=publishedAt&pageSize=6&apiKey=${NEWS_API_KEY}`;
                        const newsRes = await axios.get(newsURL, { timeout: 8000 });
                        sentiment = summarizeSentiment(newsRes.data.articles || []);

                        newsCache.set(cacheKey, {
                            sentiment,
                            timestamp: Date.now()
                        });
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
                    trend: prediction.trend || "neutral"
                },
                sentiment
            });
        });

        py.stdin.write(JSON.stringify(historical));
        py.stdin.end();

    } catch (err) {
        if (!responded) {
            console.error(err);
            const msg = err.response?.status ? `Yahoo error ${err.response.status}` : err.message;
            res.json({ success: false, error: msg });
        }
    }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
