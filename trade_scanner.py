# AI-Powered Trade Scanner for Financial Markets
# This system analyzes news, reports, and social media for trading insights

import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')

import re
import yfinance as yf
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from newsapi import NewsApiClient
import tweepy
import requests
from tqdm import tqdm
import time
import logging
import smtplib
from email.message import EmailMessage



# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   filename='trade_scanner.log')
logger = logging.getLogger('trade_scanner')

# Download necessary NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt_tab')

nltk.download('stopwords')

class TradeScanner:
    def __init__(self, news_api_key, twitter_api_key, twitter_api_secret, 
                 twitter_access_token, twitter_access_secret):
        """
        Initialize the TradeScanner with API credentials
        """
        self.news_api = NewsApiClient(api_key=news_api_key)
        
        # Initialize Twitter API client
        auth = tweepy.OAuthHandler(twitter_api_key, twitter_api_secret)
        auth.set_access_token(twitter_access_token, twitter_access_secret)
        self.twitter_api = tweepy.API(auth)
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Common financial terms and stock tickers
        self.financial_terms = [
            'stock', 'market', 'trading', 'investor', 'bearish', 'bullish',
            'dividend', 'earnings', 'quarter', 'revenue', 'profit', 'loss',
            'growth', 'decline', 'rally', 'correction', 'crash', 'volatility'
        ]
        
        # Load list of stock tickers (this could be expanded)
        self.stock_tickers = self._load_stock_tickers()
        
        # Store the analyzed data
        self.data = {
            'news': [],
            'tweets': [],
            'sentiment': {},
            'mentioned_stocks': {},
            'trending_topics': []
        }
    
    def _load_stock_tickers(self):
        """
        Load a list of common stock tickers (can be expanded)
        """
        # This is a simplified list - in production, you'd want a complete database
        major_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'JPM', 
                         'BAC', 'WMT', 'PG', 'JNJ', 'V', 'MA', 'DIS', 'NFLX', 
                         'NVDA', 'AMD', 'INTC', 'XOM', 'CVX', 'PFE', 'MRK']
        
        return set(major_tickers)
    
    def fetch_news(self, query='business OR finance OR stock market', days=1):
        """
        Fetch financial news from the past n days
        """
        logger.info(f"Fetching news with query: {query} for past {days} days")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            articles = self.news_api.get_everything(
                q=query,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=100
            )
            
            if articles['status'] == 'ok':
                logger.info(f"Successfully fetched {len(articles['articles'])} news articles")
                self.data['news'] = articles['articles']
                return articles['articles']
            else:
                logger.error(f"Failed to fetch news: {articles['status']}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []
    
    def fetch_tweets(self, queries=['stocks', 'investing', 'finance'], count=100):
        """
        Fetch tweets related to finance and investing
        """
        logger.info(f"Fetching tweets with queries: {queries}")
        all_tweets = []
        
        try:
            for query in queries:
                tweets = self.twitter_api.search_tweets(q=query, count=count, 
                                                       lang='en', tweet_mode='extended')
                all_tweets.extend([tweet._json for tweet in tweets])
            
            logger.info(f"Successfully fetched {len(all_tweets)} tweets")
            self.data['tweets'] = all_tweets
            return all_tweets
            
        except Exception as e:
            logger.error(f"Error fetching tweets: {str(e)}")
            return []
    
    def analyze_text_sentiment(self, text):
        """
        Analyze sentiment of a text snippet
        """
        if not text:
            return {
                'compound': 0,
                'neg': 0,
                'neu': 0,
                'pos': 0,
                'textblob_polarity': 0
            }
        
        # VADER sentiment analysis
        vader_sentiment = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob for additional perspective
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        
        result = vader_sentiment
        result['textblob_polarity'] = textblob_polarity
        
        return result
    
    def extract_mentioned_stocks(self, text):
        """
        Extract stock ticker mentions from text
        """
        if not text:
            return []
        
        # Convert to uppercase for matching
        text_upper = text.upper()
        
        # Find all words
        words = re.findall(r'\b[A-Z]+\b', text_upper)
        
        # Filter to only include known stock tickers
        mentioned_tickers = [word for word in words if word in self.stock_tickers]
        
        return mentioned_tickers
    
    def analyze_news_articles(self):
        """
        Analyze sentiment and extract stock mentions from news articles
        """
        logger.info("Analyzing news articles")
        results = []
        
        for article in tqdm(self.data['news']):
            # Combine title and description for analysis
            text = f"{article.get('title', '')} {article.get('description', '')}"
            
            # Analyze sentiment
            sentiment = self.analyze_text_sentiment(text)
            
            # Extract mentioned stocks
            mentioned_stocks = self.extract_mentioned_stocks(text)
            
            # Update stock mention counts
            for ticker in mentioned_stocks:
                if ticker not in self.data['mentioned_stocks']:
                    self.data['mentioned_stocks'][ticker] = 0
                self.data['mentioned_stocks'][ticker] += 1
            
            results.append({
                'source': article.get('source', {}).get('name', 'Unknown'),
                'title': article.get('title', ''),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', ''),
                'sentiment': sentiment,
                'mentioned_stocks': mentioned_stocks
            })
        
        logger.info(f"Analyzed {len(results)} news articles")
        return results
    
    def analyze_tweets(self):
        """
        Analyze sentiment and extract stock mentions from tweets
        """
        logger.info("Analyzing tweets")
        results = []
        
        for tweet in tqdm(self.data['tweets']):
            # Get full text
            if 'retweeted_status' in tweet and 'full_text' in tweet['retweeted_status']:
                text = tweet['retweeted_status']['full_text']
            else:
                text = tweet.get('full_text', tweet.get('text', ''))
            
            # Analyze sentiment
            sentiment = self.analyze_text_sentiment(text)
            
            # Extract mentioned stocks
            mentioned_stocks = self.extract_mentioned_stocks(text)
            
            # Update stock mention counts
            for ticker in mentioned_stocks:
                if ticker not in self.data['mentioned_stocks']:
                    self.data['mentioned_stocks'][ticker] = 0
                self.data['mentioned_stocks'][ticker] += 1
            
            results.append({
                'user': tweet.get('user', {}).get('screen_name', 'Unknown'),
                'text': text,
                'created_at': tweet.get('created_at', ''),
                'retweet_count': tweet.get('retweet_count', 0),
                'favorite_count': tweet.get('favorite_count', 0),
                'sentiment': sentiment,
                'mentioned_stocks': mentioned_stocks
            })
        
        logger.info(f"Analyzed {len(results)} tweets")
        return results
    
    def identify_trending_topics(self):
        """
        Identify trending financial topics from news and tweets
        """
        logger.info("Identifying trending topics")
        
        # Combine all text data
        all_text = []
        
        # Add news titles and descriptions
        for article in self.data['news']:
            all_text.append(f"{article.get('title', '')} {article.get('description', '')}")
        
        # Add tweet text
        for tweet in self.data['tweets']:
            if 'retweeted_status' in tweet and 'full_text' in tweet['retweeted_status']:
                all_text.append(tweet['retweeted_status']['full_text'])
            else:
                all_text.append(tweet.get('full_text', tweet.get('text', '')))
        
        # Combine all text
        combined_text = ' '.join(all_text)
        
        # Tokenize
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(combined_text.lower())
        
        # Filter tokens
        filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        
        # Get frequency distribution
        freq_dist = nltk.FreqDist(filtered_tokens)
        
        # Get top 20 terms
        top_terms = freq_dist.most_common(20)
        
        self.data['trending_topics'] = top_terms
        return top_terms
    
    def calculate_stock_sentiment(self):
        """
        Calculate overall sentiment for mentioned stocks
        """
        logger.info("Calculating stock sentiment")
        stock_sentiment = {}
        
        # Process news articles
        for article in self.data['news']:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            sentiment = self.analyze_text_sentiment(text)
            mentioned_stocks = self.extract_mentioned_stocks(text)
            
            for ticker in mentioned_stocks:
                if ticker not in stock_sentiment:
                    stock_sentiment[ticker] = {
                        'mentions': 0,
                        'sentiment_sum': 0,
                        'sources': []
                    }
                
                stock_sentiment[ticker]['mentions'] += 1
                stock_sentiment[ticker]['sentiment_sum'] += sentiment['compound']
                stock_sentiment[ticker]['sources'].append({
                    'type': 'news',
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'title': article.get('title', ''),
                    'sentiment': sentiment['compound']
                })
        
        # Process tweets
        for tweet in self.data['tweets']:
            if 'retweeted_status' in tweet and 'full_text' in tweet['retweeted_status']:
                text = tweet['retweeted_status']['full_text']
            else:
                text = tweet.get('full_text', tweet.get('text', ''))
            
            sentiment = self.analyze_text_sentiment(text)
            mentioned_stocks = self.extract_mentioned_stocks(text)
            
            for ticker in mentioned_stocks:
                if ticker not in stock_sentiment:
                    stock_sentiment[ticker] = {
                        'mentions': 0,
                        'sentiment_sum': 0,
                        'sources': []
                    }
                
                stock_sentiment[ticker]['mentions'] += 1
                stock_sentiment[ticker]['sentiment_sum'] += sentiment['compound']
                stock_sentiment[ticker]['sources'].append({
                    'type': 'tweet',
                    'user': tweet.get('user', {}).get('screen_name', 'Unknown'),
                    'text': text[:100] + '...',
                    'sentiment': sentiment['compound']
                })
        
        # Calculate average sentiment
        for ticker in stock_sentiment:
            if stock_sentiment[ticker]['mentions'] > 0:
                stock_sentiment[ticker]['avg_sentiment'] = stock_sentiment[ticker]['sentiment_sum'] / stock_sentiment[ticker]['mentions']
            else:
                stock_sentiment[ticker]['avg_sentiment'] = 0
        
        self.data['stock_sentiment'] = stock_sentiment
        return stock_sentiment
    
    def get_stock_price_data(self, tickers, days=30):
        """
        Get historical price data for mentioned stocks
        """
        logger.info(f"Getting price data for {len(tickers)} stocks over {days} days")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        price_data = {}
        
        for ticker in tickers:
            try:
                # Get data from yfinance
                stock_data = yf.download(ticker, start=start_date, end=end_date)
                
                if not stock_data.empty:
                    # Calculate returns
                    stock_data['Return'] = stock_data['Close'].pct_change()
                    
                    # Calculate volatility (rolling 5-day standard deviation)
                    stock_data['Volatility'] = stock_data['Return'].rolling(window=5).std()
                    
                    price_data[ticker] = stock_data
                    
                    logger.info(f"Successfully retrieved data for {ticker}")
                else:
                    logger.warning(f"No data found for {ticker}")
            
            except Exception as e:
                logger.error(f"Error getting data for {ticker}: {str(e)}")
        
        return price_data
    
    def generate_trading_insights(self):
        """
        Generate actionable trading insights based on sentiment and price data
        """
        logger.info("Generating trading insights")
        
        insights = []
        
        # Get stock sentiment data
        stock_sentiment = self.data.get('stock_sentiment', {})
        
        # Sort stocks by mention count
        sorted_stocks = sorted(stock_sentiment.items(), 
                             key=lambda x: x[1]['mentions'], 
                             reverse=True)
        
        # Get top mentioned stocks
        top_stocks = [ticker for ticker, data in sorted_stocks[:10]]
        
        # Get price data for top stocks
        price_data = self.get_stock_price_data(top_stocks)
        
        # Generate insights for each top stock
        for ticker, sentiment_data in sorted_stocks[:10]:
            if ticker in price_data:
                stock_prices = price_data[ticker]
                
                # Current trend (last 5 days)
                if len(stock_prices) >= 5:
                    # Extract scalar values using .item() to convert from Series/numpy types to Python native types
                    latest_close = float(stock_prices['Close'].iloc[-1])
                    five_days_ago_close = float(stock_prices['Close'].iloc[-5])
                    recent_return = ((latest_close / five_days_ago_close) - 1) * 100
                    current_price = latest_close
                    
                    # Current volatility
                    current_volatility = stock_prices['Volatility'].iloc[-1] if not np.isnan(stock_prices['Volatility'].iloc[-1]) else 0
                    
                    # Sentiment metrics
                    mention_count = sentiment_data['mentions']
                    avg_sentiment = sentiment_data['avg_sentiment']
                    
                    # Generate insight based on sentiment and price action
                    insight = {
                        'ticker': ticker,
                        'current_price': current_price,
                        'recent_return': recent_return,
                        'volatility': current_volatility,
                        'mention_count': mention_count,
                        'sentiment_score': avg_sentiment,
                        'sentiment_category': self._categorize_sentiment(avg_sentiment),
                        'price_action': self._categorize_price_action(recent_return),
                        'signals': []
                    }
                    
                    # Potential bullish signals
                    if avg_sentiment > 0.2 and recent_return < 0:
                        insight['signals'].append({
                            'type': 'bullish',
                            'strength': 'medium',
                            'description': f"Positive sentiment ({avg_sentiment:.2f}) despite recent price decline ({recent_return:.2f}%). Potential buying opportunity if fundamentals support it."
                        })
                    
                    if avg_sentiment > 0.3 and recent_return > 0:
                        insight['signals'].append({
                            'type': 'bullish',
                            'strength': 'strong',
                            'description': f"Strong positive sentiment ({avg_sentiment:.2f}) with confirming price action ({recent_return:.2f}%). Momentum may continue if volume supports."
                        })
                    
                    # Potential bearish signals
                    if avg_sentiment < -0.2 and recent_return > 0:
                        insight['signals'].append({
                            'type': 'bearish',
                            'strength': 'medium',
                            'description': f"Negative sentiment ({avg_sentiment:.2f}) despite recent price increase ({recent_return:.2f}%). Watch for potential reversal."
                        })
                    
                    if avg_sentiment < -0.3 and recent_return < 0:
                        insight['signals'].append({
                            'type': 'bearish',
                            'strength': 'strong',
                            'description': f"Strong negative sentiment ({avg_sentiment:.2f}) with confirming price action ({recent_return:.2f}%). Downtrend may continue."
                        })
                    
                    # Volatility-based signals
                    if current_volatility > 0.02 and abs(avg_sentiment) > 0.3:
                        insight['signals'].append({
                            'type': 'volatility',
                            'strength': 'high',
                            'description': f"High volatility ({current_volatility:.4f}) with strong sentiment ({avg_sentiment:.2f}). Consider adjusting position sizes or using options strategies."
                        })
                    
                    # If no specific signals, provide general insight
                    if not insight['signals']:
                        if abs(avg_sentiment) < 0.1:
                            insight['signals'].append({
                                'type': 'neutral',
                                'strength': 'low',
                                'description': f"Mixed sentiment ({avg_sentiment:.2f}) without clear direction. Consider waiting for stronger signals."
                            })
                        else:
                            direction = "positive" if avg_sentiment > 0 else "negative"
                            insight['signals'].append({
                                'type': 'general',
                                'strength': 'low',
                                'description': f"General {direction} sentiment ({avg_sentiment:.2f}) without strong price confirmation. Monitor for developing patterns."
                            })
                    
                    insights.append(insight)
        
        return insights
        
    def _categorize_sentiment(self, sentiment_score):
        """
        Categorize sentiment score into descriptive category
        """
        if sentiment_score >= 0.5:
            return "Very Positive"
        elif sentiment_score >= 0.2:
            return "Positive"
        elif sentiment_score > -0.2:
            return "Neutral"
        elif sentiment_score > -0.5:
            return "Negative"
        else:
            return "Very Negative"
    
    def _categorize_price_action(self, percent_change):
        """
        Categorize recent price action into descriptive category
        """
        # Convert to float if it's a Series
        if hasattr(percent_change, 'item'):
            percent_change = percent_change.item()
        
        if percent_change >= 5:
            return "Strong Uptrend"
        elif percent_change >= 1:
            return "Uptrend"
        elif percent_change > -1:
            return "Sideways"
        elif percent_change > -5:
            return "Downtrend"
        else:
            return "Strong Downtrend"
    
    
    def generate_report(self, insights, format='text'):
        """
        Generate a formatted report of trading insights
        """
        logger.info(f"Generating {format} report")
        
        if format == 'text':
            
            
            # Top trending topics
            report = "TRENDING MARKET TOPICS:\n"
            report += "-" * 30 + "\n"
            for term, count in self.data['trending_topics'][:10]:
                report += f"- {term.upper()}: mentioned {count} times\n"
            
            report += "\n"
            
            # Top mentioned stocks
            report += "MOST DISCUSSED STOCKS:\n"
            report += "-" * 30 + "\n"
            sorted_stocks = sorted(
                self.data.get('mentioned_stocks', {}).items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for ticker, count in sorted_stocks[:10]:
                report += f"- {ticker}: {count} mentions\n"
            
            report += "\n"
            
            # Trading insights
            report += "ACTIONABLE TRADING INSIGHTS:\n"
            report += "=" * 60 + "\n\n"
            
            for i, insight in enumerate(insights, 1):
                ticker = insight['ticker']
                report += f"{i}. {ticker} (${insight['current_price']:.2f})\n"
                report += "-" * 30 + "\n"
                report += f"   Market Sentiment: {insight['sentiment_category']} ({insight['sentiment_score']:.2f})\n"
                report += f"   Recent Performance: {insight['price_action']} ({insight['recent_return']:.2f}%)\n"
                report += f"   Mention Count: {insight['mention_count']}\n"
                report += f"   Volatility: {insight['volatility']:.4f}\n\n"
                
                report += "   SIGNALS:\n"
                for signal in insight['signals']:
                    report += f"   * {signal['type'].upper()} ({signal['strength']}): {signal['description']}\n"
                
                report += "\n"
            
            report += "=" * 60 + "\n"
            report += "DISCLAIMER: These insights are generated by AI analysis of public market sentiment and price data. "
            report += "They should not be considered as financial advice. Always perform your own due diligence "
            report += "and consult with a financial advisor before making investment decisions.\n"
            
            return report
            
        elif format == 'json':
            # Return structured JSON data
            return {
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'trending_topics': self.data['trending_topics'][:10],
                'top_stocks': sorted(
                    self.data.get('mentioned_stocks', {}).items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10],
                'insights': insights
            }
        
        else:
            logger.error(f"Unsupported report format: {format}")
            return None
    
    def run_analysis(self):
        """
        Run the complete analysis pipeline
        """
        logger.info("Starting full analysis pipeline")
        
        # 1. Fetch data
        self.fetch_news()
        self.fetch_tweets()
        
        # 2. Analyze data
        self.analyze_news_articles()
        self.analyze_tweets()
        
        # 3. Generate insights
        self.identify_trending_topics()
        self.calculate_stock_sentiment()
        insights = self.generate_trading_insights()
        
        # 4. Generate report
        report = self.generate_report(insights)
        
        logger.info("Analysis pipeline completed successfully")
        
        return report

# Example usage
if __name__ == "__main__":
    # API keys would be stored securely in production
    NEWS_API_KEY = "62e97016f02f4223a08fcf56f6ea09d5"
    TWITTER_API_KEY = "4R5FD9gnMPetyzN09iJTbH9lO"
    TWITTER_API_SECRET = "5LpjDJOwuVeRoJgDVZJUPt40OITMdfk2oFmAzJ90EmDAU4ggB2"
    TWITTER_ACCESS_TOKEN = "1906773685190426624-DKTkgBr8RCsgkr6hKvnR3dv9tuyzaU"
    TWITTER_ACCESS_SECRET = "7oizhKcjiQj2IzeLSWqngc39Xw1wbsSiC1btcnV8p0Z95"
    
    # Initialize the trade scanner
    scanner = TradeScanner(
        news_api_key=NEWS_API_KEY,
        twitter_api_key=TWITTER_API_KEY,
        twitter_api_secret=TWITTER_API_SECRET,
        twitter_access_token=TWITTER_ACCESS_TOKEN,
        twitter_access_secret=TWITTER_ACCESS_SECRET
    )
    
    subject="AI-POWERED TRADE SCANNER: MARKET INSIGHTS REPORT"
    body="body_content"
    from_email = "alankritidubey26725@gmail.com"
    to_email ="adubey2@buffalo.edu"
    #multiple=["evangelene.grace@gmail.com","evangelenegrace.al@gmail.com", "adubey2@buffalo.edu", "vk52@buffalo.edu"]
    password="nrhj jvek gxkh smes"
    to_email_mul=(",").join(multiple)
    
    
    
    # Run analysis
    report = scanner.run_analysis()
    
    msg=EmailMessage()
    msg["Subject"]=subject
    msg["From"]=from_email
    msg["To"]=multiple
    msg.set_content(report)
    
    with smtplib.SMTP_SSL("smtp.gmail.com",465) as my_server:
        my_server.login(from_email,password)
        my_server.send_message(msg)
        print("email sent")
    
    # Print the report
    print(report)
    
    # You could also save the report to a file
    with open(f"trade_scanner_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", "w") as f:
        f.write(report)