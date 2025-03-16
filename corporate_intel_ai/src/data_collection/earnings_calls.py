"""
Earnings Call Transcripts Collection Module

This module handles the collection of earnings call transcripts from various sources.
It provides functionality to search, download, and parse earnings call transcripts
for competitive intelligence analysis.
"""

import os
import requests
import logging
import json
import re
import time
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from bs4 import BeautifulSoup

# Configure logging
logger = logging.getLogger(__name__)

class EarningsCallCollector:
    """
    Class for collecting and processing earnings call transcripts.
    """
    
    def __init__(self, api_key: Optional[str] = None, output_dir: str = "data/earnings_calls", 
                 use_cache: bool = True, cache_expiry_days: int = 30):
        """
        Initialize the earnings call transcript collector.
        
        Args:
            api_key: API key for premium transcript services (optional)
            output_dir: Directory to store downloaded transcripts
            use_cache: Whether to use caching for downloaded data
            cache_expiry_days: Number of days after which cache entries expire
        """
        self.api_key = api_key
        self.output_dir = output_dir
        self.use_cache = use_cache
        self.cache_expiry_days = cache_expiry_days
        self.cache_dir = os.path.join(output_dir, "_cache")
        
        # API endpoints
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        self.seeking_alpha_url = "https://seeking-alpha.p.rapidapi.com/transcripts/get-details"
        self.motley_fool_url = "https://www.fool.com/earnings/call-transcripts"
        
        # Create output and cache directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Headers for API requests
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Corporate Intelligence Automation/1.0"
        }
        
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def _get_cache_key(self, prefix: str, identifier: str) -> str:
        """
        Generate a cache key for a given prefix and identifier.
        
        Args:
            prefix: Type of data being cached
            identifier: Unique identifier for the data
            
        Returns:
            Cache key string
        """
        return f"{prefix}_{identifier.replace('/', '_').replace('-', '_').replace(' ', '_')}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve data from cache if it exists and is not expired.
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            Cached data if found and not expired, None otherwise
        """
        if not self.use_cache:
            return None
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if not os.path.exists(cache_file):
            return None
            
        # Check if cache is expired
        file_time = os.path.getmtime(cache_file)
        file_age = datetime.now().timestamp() - file_time
        if file_age > self.cache_expiry_days * 24 * 60 * 60:
            # Cache expired
            os.remove(cache_file)
            return None
            
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error reading from cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: Any) -> None:
        """
        Save data to cache.
        
        Args:
            cache_key: Cache key to store data under
            data: Data to cache
        """
        if not self.use_cache:
            return
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Error writing to cache: {e}")
    
    def _make_request(self, url: str, method: str = "GET", params: Optional[Dict] = None, 
                     data: Optional[Dict] = None, headers: Optional[Dict] = None) -> requests.Response:
        """
        Make a request to an API with rate limiting.
        
        Args:
            url: URL to request
            method: HTTP method (GET, POST)
            params: Optional query parameters
            data: Optional request body data
            headers: Optional headers to override defaults
            
        Returns:
            Response object
        """
        # Sleep to respect API rate limits
        time.sleep(0.5)
        
        request_headers = self.headers.copy()
        if headers:
            request_headers.update(headers)
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=request_headers, params=params)
            elif method.upper() == "POST":
                response = requests.post(url, headers=request_headers, params=params, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Error making request to {url}: {e}")
            raise
    
    def search_transcripts(self, 
                          ticker: str,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          quarter: Optional[str] = None,
                          year: Optional[int] = None) -> List[Dict]:
        """
        Search for earnings call transcripts based on ticker and date range.
        
        Args:
            ticker: Company ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            quarter: Fiscal quarter (Q1, Q2, Q3, Q4)
            year: Fiscal year
            
        Returns:
            List of dictionaries containing transcript metadata
        """
        logger.info(f"Searching for earnings call transcripts for {ticker}")
        
        # Create a cache key based on search parameters
        cache_params = f"{ticker}_{start_date}_{end_date}_{quarter}_{year}"
        cache_key = self._get_cache_key("search", cache_params)
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached transcript search results for {ticker}")
            return cached_result
        
        # Try multiple sources for transcripts
        results = []
        
        # Try Alpha Vantage first for earnings dates
        try:
            alpha_vantage_results = self._search_alpha_vantage(ticker, start_date, end_date)
            if alpha_vantage_results:
                results.extend(alpha_vantage_results)
        except Exception as e:
            logger.warning(f"Alpha Vantage search failed: {e}")
        
        # Try Seeking Alpha if we have an API key
        if self.api_key:
            try:
                seeking_alpha_results = self._search_seeking_alpha(ticker, start_date, end_date, quarter, year)
                if seeking_alpha_results:
                    results.extend(seeking_alpha_results)
            except Exception as e:
                logger.warning(f"Seeking Alpha search failed: {e}")
        
        # Try Motley Fool as a fallback
        try:
            motley_fool_results = self._search_motley_fool(ticker, start_date, end_date)
            if motley_fool_results:
                results.extend(motley_fool_results)
        except Exception as e:
            logger.warning(f"Motley Fool search failed: {e}")
        
        # Filter by quarter and year if specified
        if quarter or year:
            filtered_results = []
            for result in results:
                if quarter and result.get('quarter') != quarter:
                    continue
                if year and result.get('year') != year:
                    continue
                filtered_results.append(result)
            results = filtered_results
        
        # Remove duplicates based on date
        unique_results = {}
        for result in results:
            date = result.get('date')
            if date and date not in unique_results:
                unique_results[date] = result
        
        results = list(unique_results.values())
        
        # Cache and return results
        self._save_to_cache(cache_key, results)
        return results
    
    def _search_alpha_vantage(self, ticker: str, start_date: Optional[str] = None, 
                             end_date: Optional[str] = None) -> List[Dict]:
        """
        Search for earnings call dates using Alpha Vantage API.
        
        Args:
            ticker: Company ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of dictionaries containing earnings call metadata
        """
        if not self.api_key:
            logger.warning("Alpha Vantage API key not provided")
            return []
        
        # Construct query parameters
        params = {
            "function": "EARNINGS_CALENDAR",
            "symbol": ticker,
            "apikey": self.api_key
        }
        
        # Make the request
        try:
            response = self._make_request(self.alpha_vantage_url, params=params)
            data = response.json()
            
            # Extract earnings call information
            results = []
            
            # Alpha Vantage returns earnings calendar data
            earnings_data = data.get("earnings_calendar", [])
            for entry in earnings_data:
                report_date = entry.get("reportDate")
                if not report_date:
                    continue
                
                # Filter by date range if specified
                if start_date and report_date < start_date:
                    continue
                if end_date and report_date > end_date:
                    continue
                
                # Extract fiscal quarter and year
                fiscal_date_ending = entry.get("fiscalDateEnding", "")
                fiscal_year = None
                fiscal_quarter = None
                
                if fiscal_date_ending:
                    fiscal_year = int(fiscal_date_ending.split("-")[0])
                    month = int(fiscal_date_ending.split("-")[1])
                    if 1 <= month <= 3:
                        fiscal_quarter = "Q1"
                    elif 4 <= month <= 6:
                        fiscal_quarter = "Q2"
                    elif 7 <= month <= 9:
                        fiscal_quarter = "Q3"
                    else:
                        fiscal_quarter = "Q4"
                
                results.append({
                    "ticker": ticker,
                    "date": report_date,
                    "quarter": fiscal_quarter,
                    "year": fiscal_year,
                    "estimated_eps": entry.get("estimatedEPS"),
                    "reported_eps": entry.get("reportedEPS"),
                    "surprise_percent": entry.get("surprisePercentage"),
                    "source": "alpha_vantage",
                    "transcript_url": None  # Alpha Vantage doesn't provide transcript URLs
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Alpha Vantage: {e}")
            return []
    
    def _search_seeking_alpha(self, ticker: str, start_date: Optional[str] = None, 
                             end_date: Optional[str] = None, quarter: Optional[str] = None, 
                             year: Optional[int] = None) -> List[Dict]:
        """
        Search for earnings call transcripts using Seeking Alpha API.
        
        Args:
            ticker: Company ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            quarter: Fiscal quarter (Q1, Q2, Q3, Q4)
            year: Fiscal year
            
        Returns:
            List of dictionaries containing transcript metadata
        """
        # Seeking Alpha requires a different API key format
        if not self.api_key:
            logger.warning("Seeking Alpha API key not provided")
            return []
        
        # Construct headers for Seeking Alpha API
        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }
        
        # Construct query parameters
        params = {
            "symbol": ticker,
            "type": "earnings"
        }
        
        # Make the request
        try:
            response = self._make_request(self.seeking_alpha_url, headers=headers, params=params)
            data = response.json()
            
            # Extract transcript information
            results = []
            
            transcripts = data.get("data", [])
            for transcript in transcripts:
                # Extract date
                publish_date = transcript.get("publishedAt", "")
                if not publish_date:
                    continue
                
                # Convert date format if needed
                try:
                    date_obj = datetime.fromisoformat(publish_date.replace("Z", "+00:00"))
                    formatted_date = date_obj.strftime("%Y-%m-%d")
                except:
                    formatted_date = publish_date
                
                # Filter by date range if specified
                if start_date and formatted_date < start_date:
                    continue
                if end_date and formatted_date > end_date:
                    continue
                
                # Extract quarter and year from title
                title = transcript.get("title", "")
                extracted_quarter = None
                extracted_year = None
                
                # Try to extract quarter and year from title
                quarter_match = re.search(r'Q([1-4])\s+(\d{4})', title)
                if quarter_match:
                    extracted_quarter = f"Q{quarter_match.group(1)}"
                    extracted_year = int(quarter_match.group(2))
                
                # Filter by quarter and year if specified
                if quarter and extracted_quarter != quarter:
                    continue
                if year and extracted_year != year:
                    continue
                
                results.append({
                    "ticker": ticker,
                    "date": formatted_date,
                    "quarter": extracted_quarter,
                    "year": extracted_year,
                    "title": title,
                    "source": "seeking_alpha",
                    "transcript_id": transcript.get("id"),
                    "transcript_url": f"https://seekingalpha.com/article/{transcript.get('id')}"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Seeking Alpha: {e}")
            return []
    
    def _search_motley_fool(self, ticker: str, start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> List[Dict]:
        """
        Search for earnings call transcripts on Motley Fool website.
        
        Args:
            ticker: Company ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of dictionaries containing transcript metadata
        """
        # Construct URL for Motley Fool transcript search
        url = f"{self.motley_fool_url}/{ticker}"
        
        try:
            # Make the request
            response = requests.get(url, headers={"User-Agent": "Corporate Intelligence Automation/1.0"})
            
            # Check if the page exists
            if response.status_code == 404:
                logger.warning(f"No transcripts found for {ticker} on Motley Fool")
                return []
            
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract transcript links
            results = []
            
            # Find transcript links
            transcript_links = soup.select("a[href*='/earnings/call-transcripts/']")
            for link in transcript_links:
                href = link.get('href')
                title = link.text.strip()
                
                # Extract date from URL
                date_match = re.search(r'/(\d{4})-(\d{2})-(\d{2})/', href)
                if not date_match:
                    continue
                
                formatted_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                
                # Filter by date range if specified
                if start_date and formatted_date < start_date:
                    continue
                if end_date and formatted_date > end_date:
                    continue
                
                # Extract quarter and year from title
                quarter_match = re.search(r'Q([1-4])\s+(\d{4})', title)
                extracted_quarter = None
                extracted_year = None
                
                if quarter_match:
                    extracted_quarter = f"Q{quarter_match.group(1)}"
                    extracted_year = int(quarter_match.group(2))
                
                results.append({
                    "ticker": ticker,
                    "date": formatted_date,
                    "quarter": extracted_quarter,
                    "year": extracted_year,
                    "title": title,
                    "source": "motley_fool",
                    "transcript_url": f"https://www.fool.com{href}"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Motley Fool: {e}")
            return []
    
    def download_transcript(self, 
                           ticker: str, 
                           date: str, 
                           quarter: Optional[str] = None,
                           transcript_url: Optional[str] = None,
                           transcript_id: Optional[str] = None,
                           save_path: Optional[str] = None) -> str:
        """
        Download a specific earnings call transcript.
        
        Args:
            ticker: Company ticker symbol
            date: Date of earnings call in YYYY-MM-DD format
            quarter: Fiscal quarter (Q1, Q2, Q3, Q4)
            transcript_url: URL to transcript (if available)
            transcript_id: ID of transcript (for certain sources)
            save_path: Optional path to save the transcript
            
        Returns:
            Path to the downloaded transcript
        """
        # Determine save path
        if save_path is None:
            quarter_str = f"_{quarter}" if quarter else ""
            save_path = os.path.join(self.output_dir, f"{ticker}_{date}{quarter_str}.txt")
        
        # Check if file already exists
        if os.path.exists(save_path):
            logger.info(f"Transcript already exists at {save_path}")
            return save_path
        
        logger.info(f"Downloading transcript for {ticker} on {date}")
        
        # Try to download from URL if provided
        if transcript_url:
            try:
                if "seekingalpha.com" in transcript_url:
                    content = self._download_seeking_alpha_transcript(transcript_url, transcript_id)
                elif "fool.com" in transcript_url:
                    content = self._download_motley_fool_transcript(transcript_url)
                else:
                    # Generic URL download
                    response = requests.get(transcript_url, headers={"User-Agent": "Corporate Intelligence Automation/1.0"})
                    response.raise_for_status()
                    content = response.text
                
                if content:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    logger.info(f"Successfully downloaded transcript to {save_path}")
                    return save_path
            
            except Exception as e:
                logger.warning(f"Error downloading transcript from URL: {e}")
        
        # If URL download failed or no URL provided, try alternative sources
        try:
            # Try to find transcript from Motley Fool
            motley_fool_url = f"{self.motley_fool_url}/{ticker}"
            response = requests.get(motley_fool_url, headers={"User-Agent": "Corporate Intelligence Automation/1.0"})
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find transcript links that match the date
                for link in soup.select("a[href*='/earnings/call-transcripts/']"):
                    href = link.get('href')
                    if date in href:
                        transcript_url = f"https://www.fool.com{href}"
                        content = self._download_motley_fool_transcript(transcript_url)
                        
                        if content:
                            with open(save_path, 'w', encoding='utf-8') as f:
                                f.write(content)
                            
                            logger.info(f"Successfully downloaded transcript to {save_path}")
                            return save_path
        
        except Exception as e:
            logger.warning(f"Error finding alternative transcript source: {e}")
        
        # If all attempts failed, create an empty file with a note
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"# Transcript for {ticker} on {date} could not be found.\n")
        
        logger.warning(f"Could not find transcript for {ticker} on {date}")
        return save_path
    
    def _download_seeking_alpha_transcript(self, url: str, transcript_id: Optional[str] = None) -> Optional[str]:
        """
        Download a transcript from Seeking Alpha.
        
        Args:
            url: URL to the transcript
            transcript_id: ID of the transcript
            
        Returns:
            Transcript content as string, or None if download failed
        """
        if not self.api_key:
            logger.warning("Seeking Alpha API key not provided")
            return None
        
        # If we have a transcript ID, use the API
        if transcript_id:
            # Construct headers for Seeking Alpha API
            headers = {
                "X-RapidAPI-Key": self.api_key,
                "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
            }
            
            # Construct query parameters
            params = {
                "id": transcript_id
            }
            
            try:
                # Use the content endpoint
                content_url = "https://seeking-alpha.p.rapidapi.com/articles/get-details"
                response = self._make_request(content_url, headers=headers, params=params)
                data = response.json()
                
                # Extract content
                content = data.get("data", {}).get("content", "")
                if content:
                    # Clean up HTML
                    soup = BeautifulSoup(content, 'html.parser')
                    text_content = soup.get_text(separator='\n\n')
                    return text_content
            
            except Exception as e:
                logger.error(f"Error downloading from Seeking Alpha API: {e}")
        
        # Fall back to web scraping if API fails or no transcript ID
        try:
            response = requests.get(url, headers={"User-Agent": "Corporate Intelligence Automation/1.0"})
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract article content
            article_content = soup.select_one("div#content-rail")
            if article_content:
                # Remove unnecessary elements
                for element in article_content.select("div.ad-wrap, div.sa-art-rec, div#author-hq"):
                    element.decompose()
                
                # Get text content
                text_content = article_content.get_text(separator='\n\n')
                return text_content
            
            return None
            
        except Exception as e:
            logger.error(f"Error scraping Seeking Alpha: {e}")
            return None
    
    def _download_motley_fool_transcript(self, url: str) -> Optional[str]:
        """
        Download a transcript from Motley Fool.
        
        Args:
            url: URL to the transcript
            
        Returns:
            Transcript content as string, or None if download failed
        """
        try:
            response = requests.get(url, headers={"User-Agent": "Corporate Intelligence Automation/1.0"})
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract article content
            article_content = soup.select_one("span.article-content")
            if article_content:
                # Get text content
                text_content = article_content.get_text(separator='\n\n')
                return text_content
            
            return None
            
        except Exception as e:
            logger.error(f"Error scraping Motley Fool: {e}")
            return None
    
    def parse_transcript(self, transcript_path: str) -> Dict:
        """
        Parse a downloaded earnings call transcript to extract structured data.
        
        Args:
            transcript_path: Path to the downloaded transcript
            
        Returns:
            Dictionary containing parsed data from the transcript
        """
        logger.info(f"Parsing transcript: {transcript_path}")
        
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Initialize the parsed data structure
            parsed_data = {
                "participants": {
                    "executives": [],
                    "analysts": []
                },
                "sections": {
                    "prepared_remarks": "",
                    "qa_session": []
                },
                "key_metrics": {},
                "metadata": {
                    "company": "",
                    "date": "",
                    "quarter": "",
                    "year": ""
                }
            }
            
            # Extract metadata from filename if possible
            filename = os.path.basename(transcript_path)
            ticker_match = re.search(r'^([A-Z]+)_', filename)
            date_match = re.search(r'_(\d{4}-\d{2}-\d{2})', filename)
            quarter_match = re.search(r'_(Q[1-4])', filename)
            
            if ticker_match:
                parsed_data["metadata"]["company"] = ticker_match.group(1)
            if date_match:
                parsed_data["metadata"]["date"] = date_match.group(1)
            if quarter_match:
                parsed_data["metadata"]["quarter"] = quarter_match.group(1)
                
            # Extract year from date if available
            if parsed_data["metadata"]["date"]:
                year_match = re.search(r'^(\d{4})', parsed_data["metadata"]["date"])
                if year_match:
                    parsed_data["metadata"]["year"] = year_match.group(1)
            
            # Extract participants
            # Look for common patterns in earnings call transcripts
            executive_patterns = [
                r'([A-Z][a-z]+ [A-Z][a-z]+) -- (Chief Executive Officer|CEO|Chief Financial Officer|CFO|President|Chairman|COO|CTO)',
                r'([A-Z][a-z]+ [A-Z][a-z]+), (Chief Executive Officer|CEO|Chief Financial Officer|CFO|President|Chairman|COO|CTO)'
            ]
            
            analyst_patterns = [
                r'([A-Z][a-z]+ [A-Z][a-z]+) -- ([A-Za-z\s]+Securities|[A-Za-z\s]+Capital|[A-Za-z\s]+Partners|[A-Za-z\s]+Research)',
                r'([A-Z][a-z]+ [A-Z][a-z]+), ([A-Za-z\s]+Securities|[A-Za-z\s]+Capital|[A-Za-z\s]+Partners|[A-Za-z\s]+Research)'
            ]
            
            # Extract executives
            for pattern in executive_patterns:
                for match in re.finditer(pattern, content):
                    name = match.group(1)
                    title = match.group(2)
                    executive = {"name": name, "title": title}
                    if executive not in parsed_data["participants"]["executives"]:
                        parsed_data["participants"]["executives"].append(executive)
            
            # Extract analysts
            for pattern in analyst_patterns:
                for match in re.finditer(pattern, content):
                    name = match.group(1)
                    firm = match.group(2)
                    analyst = {"name": name, "firm": firm}
                    if analyst not in parsed_data["participants"]["analysts"]:
                        parsed_data["participants"]["analysts"].append(analyst)
            
            # Identify sections (prepared remarks vs Q&A)
            # Common markers for Q&A section
            qa_markers = [
                r'Question-and-Answer Session',
                r'Questions and Answers',
                r'Q&A Session',
                r'Questions & Answers'
            ]
            
            qa_start_index = len(content)
            for marker in qa_markers:
                marker_index = content.find(marker)
                if marker_index != -1 and marker_index < qa_start_index:
                    qa_start_index = marker_index
            
            if qa_start_index < len(content):
                # Split content into prepared remarks and Q&A
                parsed_data["sections"]["prepared_remarks"] = content[:qa_start_index].strip()
                qa_content = content[qa_start_index:].strip()
                
                # Parse Q&A into individual questions and answers
                qa_pairs = []
                
                # Look for patterns like "Q - Analyst Name" or "Question:"
                q_patterns = [
                    r'Q - ([A-Za-z\s]+)',
                    r'Question:',
                    r'Question --'
                ]
                
                a_patterns = [
                    r'A - ([A-Za-z\s]+)',
                    r'Answer:',
                    r'Answer --'
                ]
                
                # Find all question starts
                q_starts = []
                for pattern in q_patterns:
                    for match in re.finditer(pattern, qa_content):
                        q_starts.append(match.start())
                
                q_starts.sort()
                
                # Extract Q&A pairs
                for i in range(len(q_starts)):
                    q_start = q_starts[i]
                    q_end = q_starts[i+1] if i+1 < len(q_starts) else len(qa_content)
                    
                    # Find answer within this segment
                    a_start = None
                    for pattern in a_patterns:
                        a_match = re.search(pattern, qa_content[q_start:q_end])
                        if a_match:
                            a_start = q_start + a_match.start()
                            break
                    
                    if a_start:
                        question = qa_content[q_start:a_start].strip()
                        answer = qa_content[a_start:q_end].strip()
                    else:
                        # If no clear answer marker, just take the whole segment as the Q&A
                        question = qa_content[q_start:q_end].strip()
                        answer = ""
                    
                    qa_pairs.append({
                        "question": question,
                        "answer": answer
                    })
                
                parsed_data["sections"]["qa_session"] = qa_pairs
            else:
                # No Q&A section found, treat everything as prepared remarks
                parsed_data["sections"]["prepared_remarks"] = content.strip()
            
            # Extract key metrics
            # Look for financial metrics
            metric_patterns = {
                "revenue": r'revenue of \$?(\d+\.?\d*)\s?(million|billion|m|b)',
                "earnings_per_share": r'earnings per share of \$?(\d+\.?\d*)',
                "net_income": r'net income of \$?(\d+\.?\d*)\s?(million|billion|m|b)',
                "operating_margin": r'operating margin of (\d+\.?\d*)%',
                "guidance": r'guidance of \$?(\d+\.?\d*)\s?(million|billion|m|b)'
            }
            
            for metric, pattern in metric_patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                values = []
                
                for match in matches:
                    value = match.group(1)
                    unit = match.group(2) if len(match.groups()) > 1 else ""
                    
                    # Convert to numeric
                    try:
                        numeric_value = float(value)
                        
                        # Scale based on unit
                        if unit.lower() in ['billion', 'b']:
                            numeric_value *= 1_000_000_000
                        elif unit.lower() in ['million', 'm']:
                            numeric_value *= 1_000_000
                            
                        values.append(numeric_value)
                    except ValueError:
                        continue
                
                if values:
                    parsed_data["key_metrics"][metric] = values
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing transcript: {e}")
            raise
    
    def extract_key_insights(self, parsed_transcript: Dict) -> Dict:
        """
        Extract key insights from a parsed earnings call transcript.
        
        Args:
            parsed_transcript: Dictionary containing parsed transcript data
            
        Returns:
            Dictionary containing key insights from the transcript
        """
        logger.info("Extracting key insights from transcript")
        
        insights = {
            "forward_looking_statements": [],
            "financial_highlights": [],
            "strategic_initiatives": [],
            "market_comments": [],
            "competition_mentions": [],
            "risk_factors": []
        }
        
        # Get all text content
        all_text = parsed_transcript["sections"]["prepared_remarks"]
        for qa_pair in parsed_transcript["sections"]["qa_session"]:
            all_text += " " + qa_pair.get("question", "") + " " + qa_pair.get("answer", "")
        
        # Extract forward-looking statements
        forward_patterns = [
            r'(we expect|we anticipate|we project|we aim|looking forward|in the future|next quarter|next year|guidance)',
            r'(our outlook|our guidance|our forecast|our projection)'
        ]
        
        for pattern in forward_patterns:
            for match in re.finditer(pattern, all_text, re.IGNORECASE):
                # Get the sentence containing the match
                start = max(0, all_text.rfind('.', 0, match.start()) + 1)
                end = all_text.find('.', match.end())
                if end == -1:
                    end = len(all_text)
                
                statement = all_text[start:end].strip()
                if statement and statement not in insights["forward_looking_statements"]:
                    insights["forward_looking_statements"].append(statement)
        
        # Extract financial highlights
        financial_patterns = [
            r'(revenue|sales) (grew|increased|decreased|declined) by (\d+\.?\d*)%',
            r'(profit|income|earnings) (grew|increased|decreased|declined) by (\d+\.?\d*)%',
            r'(margin|eps|earnings per share) (grew|increased|decreased|declined)',
            r'record (revenue|sales|profit|income|earnings)'
        ]
        
        for pattern in financial_patterns:
            for match in re.finditer(pattern, all_text, re.IGNORECASE):
                start = max(0, all_text.rfind('.', 0, match.start()) + 1)
                end = all_text.find('.', match.end())
                if end == -1:
                    end = len(all_text)
                
                statement = all_text[start:end].strip()
                if statement and statement not in insights["financial_highlights"]:
                    insights["financial_highlights"].append(statement)
        
        # Extract strategic initiatives
        strategic_patterns = [
            r'(strategy|strategic|initiative|investment|acquisition|partnership)',
            r'(new product|new service|new offering|new market|expansion|growth opportunity)'
        ]
        
        for pattern in strategic_patterns:
            for match in re.finditer(pattern, all_text, re.IGNORECASE):
                start = max(0, all_text.rfind('.', 0, match.start()) + 1)
                end = all_text.find('.', match.end())
                if end == -1:
                    end = len(all_text)
                
                statement = all_text[start:end].strip()
                if statement and statement not in insights["strategic_initiatives"]:
                    insights["strategic_initiatives"].append(statement)
        
        # Extract market comments
        market_patterns = [
            r'(market|industry|sector) (growth|decline|trend|condition)',
            r'(market|industry|sector) (opportunity|challenge|headwind|tailwind)',
            r'(economic|economy|macro|macroeconomic)'
        ]
        
        for pattern in market_patterns:
            for match in re.finditer(pattern, all_text, re.IGNORECASE):
                start = max(0, all_text.rfind('.', 0, match.start()) + 1)
                end = all_text.find('.', match.end())
                if end == -1:
                    end = len(all_text)
                
                statement = all_text[start:end].strip()
                if statement and statement not in insights["market_comments"]:
                    insights["market_comments"].append(statement)
        
        # Extract competition mentions
        competition_patterns = [
            r'(compet|rival|peer|player in the market)',
            r'(market share|competitive landscape|competitive position)'
        ]
        
        for pattern in competition_patterns:
            for match in re.finditer(pattern, all_text, re.IGNORECASE):
                start = max(0, all_text.rfind('.', 0, match.start()) + 1)
                end = all_text.find('.', match.end())
                if end == -1:
                    end = len(all_text)
                
                statement = all_text[start:end].strip()
                if statement and statement not in insights["competition_mentions"]:
                    insights["competition_mentions"].append(statement)
        
        # Extract risk factors
        risk_patterns = [
            r'(risk|challenge|headwind|obstacle|hurdle)',
            r'(concern|worry|uncertain|difficult|problem)',
            r'(regulatory|regulation|compliance|legal)'
        ]
        
        for pattern in risk_patterns:
            for match in re.finditer(pattern, all_text, re.IGNORECASE):
                start = max(0, all_text.rfind('.', 0, match.start()) + 1)
                end = all_text.find('.', match.end())
                if end == -1:
                    end = len(all_text)
                
                statement = all_text[start:end].strip()
                if statement and statement not in insights["risk_factors"]:
                    insights["risk_factors"].append(statement)
        
        return insights
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text using simple keyword-based approach.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        # Simple keyword-based sentiment analysis
        positive_words = [
            'growth', 'increase', 'profit', 'success', 'strong', 'positive', 'opportunity',
            'improve', 'gain', 'exceed', 'beat', 'confident', 'optimistic', 'favorable',
            'excellent', 'robust', 'momentum', 'progress', 'achievement', 'outperform'
        ]
        
        negative_words = [
            'decline', 'decrease', 'loss', 'challenge', 'weak', 'negative', 'risk',
            'difficult', 'below', 'miss', 'disappoint', 'concern', 'cautious', 'unfavorable',
            'poor', 'slow', 'headwind', 'obstacle', 'underperform', 'uncertain'
        ]
        
        uncertainty_words = [
            'may', 'might', 'could', 'possibly', 'perhaps', 'uncertain', 'unclear',
            'not sure', 'doubt', 'question', 'ambiguous', 'unpredictable', 'unknown'
        ]
        
        # Normalize text
        text_lower = text.lower()
        
        # Count occurrences
        positive_count = sum(text_lower.count(' ' + word + ' ') for word in positive_words)
        negative_count = sum(text_lower.count(' ' + word + ' ') for word in negative_words)
        uncertainty_count = sum(text_lower.count(' ' + word + ' ') for word in uncertainty_words)
        
        # Calculate total sentiment words
        total_sentiment_words = positive_count + negative_count
        
        # Calculate sentiment score (-1 to 1)
        sentiment_score = 0
        if total_sentiment_words > 0:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
        
        # Determine sentiment label
        sentiment_label = "neutral"
        if sentiment_score > 0.2:
            sentiment_label = "positive"
        elif sentiment_score < -0.2:
            sentiment_label = "negative"
        
        # Calculate confidence level (inverse of uncertainty)
        total_words = len(text_lower.split())
        uncertainty_ratio = uncertainty_count / max(1, total_words)
        confidence_level = 1 - min(1, uncertainty_ratio * 10)  # Scale for better distribution
        
        return {
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "confidence_level": confidence_level,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "uncertainty_count": uncertainty_count
        }
    
    def analyze_transcript(self, transcript_path: str) -> Dict:
        """
        Perform comprehensive analysis of an earnings call transcript.
        
        Args:
            transcript_path: Path to the transcript file
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        logger.info(f"Analyzing transcript: {transcript_path}")
        
        # Parse the transcript
        parsed_data = self.parse_transcript(transcript_path)
        
        # Extract key insights
        insights = self.extract_key_insights(parsed_data)
        
        # Analyze sentiment of different sections
        sentiment_analysis = {
            "overall": self.analyze_sentiment(parsed_data["sections"]["prepared_remarks"]),
            "prepared_remarks": self.analyze_sentiment(parsed_data["sections"]["prepared_remarks"]),
            "qa_session": {}
        }
        
        # Analyze sentiment of each Q&A pair
        for i, qa_pair in enumerate(parsed_data["sections"]["qa_session"]):
            question = qa_pair.get("question", "")
            answer = qa_pair.get("answer", "")
            
            if question and answer:
                sentiment_analysis["qa_session"][f"qa_{i+1}"] = {
                    "question": self.analyze_sentiment(question),
                    "answer": self.analyze_sentiment(answer)
                }
        
        # Combine results
        analysis_results = {
            "metadata": parsed_data["metadata"],
            "participants": parsed_data["participants"],
            "key_metrics": parsed_data["key_metrics"],
            "insights": insights,
            "sentiment": sentiment_analysis
        }
        
        return analysis_results
    
    def batch_process_transcripts(self, tickers: List[str], start_date: Optional[str] = None, 
                                 end_date: Optional[str] = None, max_per_ticker: int = 4) -> Dict[str, List[Dict]]:
        """
        Process multiple earnings call transcripts for multiple companies.
        
        Args:
            tickers: List of company ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_per_ticker: Maximum number of transcripts to process per ticker
            
        Returns:
            Dictionary mapping tickers to lists of analysis results
        """
        logger.info(f"Batch processing transcripts for {len(tickers)} companies")
        
        results = {}
        
        for ticker in tickers:
            logger.info(f"Processing {ticker}")
            
            # Search for transcripts
            transcripts = self.search_transcripts(ticker, start_date, end_date)
            
            # Limit to max_per_ticker
            transcripts = transcripts[:max_per_ticker]
            
            ticker_results = []
            
            for transcript in transcripts:
                try:
                    # Download transcript
                    transcript_path = self.download_transcript(
                        ticker=transcript.get("ticker"),
                        date=transcript.get("date"),
                        quarter=transcript.get("quarter"),
                        transcript_url=transcript.get("transcript_url"),
                        transcript_id=transcript.get("transcript_id")
                    )
                    
                    # Analyze transcript
                    analysis = self.analyze_transcript(transcript_path)
                    
                    ticker_results.append(analysis)
                    
                except Exception as e:
                    logger.error(f"Error processing transcript for {ticker}: {e}")
            
            results[ticker] = ticker_results
        
        return results
    
    def compare_transcripts(self, ticker: str, num_transcripts: int = 4) -> Dict:
        """
        Compare multiple earnings call transcripts for a company to identify trends.
        
        Args:
            ticker: Company ticker symbol
            num_transcripts: Number of most recent transcripts to compare
            
        Returns:
            Dictionary containing trend analysis
        """
        logger.info(f"Comparing {num_transcripts} transcripts for {ticker}")
        
        # Search for transcripts
        transcripts = self.search_transcripts(ticker)
        
        # Sort by date (most recent first)
        transcripts.sort(key=lambda x: x.get("date", ""), reverse=True)
        
        # Limit to num_transcripts
        transcripts = transcripts[:num_transcripts]
        
        # Download and analyze transcripts
        analyzed_transcripts = []
        
        for transcript in transcripts:
            try:
                # Download transcript
                transcript_path = self.download_transcript(
                    ticker=transcript.get("ticker"),
                    date=transcript.get("date"),
                    quarter=transcript.get("quarter"),
                    transcript_url=transcript.get("transcript_url"),
                    transcript_id=transcript.get("transcript_id")
                )
                
                # Analyze transcript
                analysis = self.analyze_transcript(transcript_path)
                
                analyzed_transcripts.append({
                    "date": transcript.get("date"),
                    "quarter": transcript.get("quarter"),
                    "year": transcript.get("year"),
                    "analysis": analysis
                })
                
            except Exception as e:
                logger.error(f"Error analyzing transcript for {ticker}: {e}")
        
        # Sort by date (oldest first for trend analysis)
        analyzed_transcripts.sort(key=lambda x: x.get("date", ""))
        
        # Extract trends
        trends = {
            "sentiment_trend": [],
            "key_metrics_trend": {},
            "recurring_themes": [],
            "emerging_topics": [],
            "fading_topics": []
        }
        
        # Analyze sentiment trend
        for transcript in analyzed_transcripts:
            date = transcript.get("date")
            quarter = transcript.get("quarter")
            sentiment = transcript.get("analysis", {}).get("sentiment", {}).get("overall", {})
            
            if date and sentiment:
                trends["sentiment_trend"].append({
                    "date": date,
                    "quarter": quarter,
                    "sentiment_score": sentiment.get("sentiment_score"),
                    "sentiment_label": sentiment.get("sentiment_label"),
                    "confidence_level": sentiment.get("confidence_level")
                })
        
        # Analyze key metrics trends
        metrics = set()
        for transcript in analyzed_transcripts:
            analysis = transcript.get("analysis", {})
            for metric in analysis.get("key_metrics", {}):
                metrics.add(metric)
        
        for metric in metrics:
            trends["key_metrics_trend"][metric] = []
            
            for transcript in analyzed_transcripts:
                date = transcript.get("date")
                quarter = transcript.get("quarter")
                analysis = transcript.get("analysis", {})
                metric_values = analysis.get("key_metrics", {}).get(metric, [])
                
                if date and metric_values:
                    # Use the first value if multiple are found
                    value = metric_values[0] if metric_values else None
                    
                    trends["key_metrics_trend"][metric].append({
                        "date": date,
                        "quarter": quarter,
                        "value": value
                    })
        
        # Identify recurring themes, emerging and fading topics
        all_insights = []
        for transcript in analyzed_transcripts:
            date = transcript.get("date")
            analysis = transcript.get("analysis", {})
            insights = analysis.get("insights", {})
            
            transcript_insights = []
            for category, statements in insights.items():
                for statement in statements:
                    transcript_insights.append({
                        "date": date,
                        "category": category,
                        "statement": statement
                    })
            
            all_insights.append(transcript_insights)
        
        # Find recurring themes (present in most transcripts)
        if all_insights:
            # Count theme occurrences across transcripts
            theme_counts = {}
            for transcript_insights in all_insights:
                themes_in_transcript = set()
                for insight in transcript_insights:
                    category = insight.get("category")
                    statement = insight.get("statement")
                    if category and statement:
                        # Use simplified version of statement as key
                        simple_statement = re.sub(r'\b(the|a|an|and|or|in|on|at|to|for|with|by|of)\b', '', statement.lower())
                        simple_statement = re.sub(r'\s+', ' ', simple_statement).strip()
                        
                        if simple_statement:
                            key = f"{category}:{simple_statement}"
                            themes_in_transcript.add(key)
                
                # Increment count for each unique theme in this transcript
                for theme in themes_in_transcript:
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1
            
            # Identify recurring themes (present in at least half of transcripts)
            min_occurrences = max(1, len(all_insights) // 2)
            for theme, count in theme_counts.items():
                if count >= min_occurrences:
                    category, statement = theme.split(':', 1)
                    trends["recurring_themes"].append({
                        "category": category,
                        "statement": statement,
                        "occurrences": count
                    })
            
            # Sort by occurrence count
            trends["recurring_themes"].sort(key=lambda x: x.get("occurrences", 0), reverse=True)
            
            # Identify emerging topics (present in most recent transcript but not earlier ones)
            if len(all_insights) >= 2:
                recent_themes = set()
                for insight in all_insights[-1]:  # Most recent transcript
                    category = insight.get("category")
                    statement = insight.get("statement")
                    if category and statement:
                        simple_statement = re.sub(r'\b(the|a|an|and|or|in|on|at|to|for|with|by|of)\b', '', statement.lower())
                        simple_statement = re.sub(r'\s+', ' ', simple_statement).strip()
                        if simple_statement:
                            recent_themes.add(f"{category}:{simple_statement}")
                
                # Check if these themes were not in earlier transcripts
                for theme in recent_themes:
                    is_emerging = True
                    for transcript_insights in all_insights[:-1]:  # All but the most recent
                        for insight in transcript_insights:
                            category = insight.get("category")
                            statement = insight.get("statement")
                            if category and statement:
                                simple_statement = re.sub(r'\b(the|a|an|and|or|in|on|at|to|for|with|by|of)\b', '', statement.lower())
                                simple_statement = re.sub(r'\s+', ' ', simple_statement).strip()
                                if simple_statement and f"{category}:{simple_statement}" == theme:
                                    is_emerging = False
                                    break
                    
                    if is_emerging:
                        category, statement = theme.split(':', 1)
                        trends["emerging_topics"].append({
                            "category": category,
                            "statement": statement
                        })
                
                # Identify fading topics (present in earlier transcripts but not the most recent)
                earlier_themes = set()
                for transcript_insights in all_insights[:-1]:  # All but the most recent
                    for insight in transcript_insights:
                        category = insight.get("category")
                        statement = insight.get("statement")
                        if category and statement:
                            simple_statement = re.sub(r'\b(the|a|an|and|or|in|on|at|to|for|with|by|of)\b', '', statement.lower())
                            simple_statement = re.sub(r'\s+', ' ', simple_statement).strip()
                            if simple_statement:
                                earlier_themes.add(f"{category}:{simple_statement}")
                
                # Check if these themes are not in the most recent transcript
                for theme in earlier_themes:
                    is_fading = True
                    for insight in all_insights[-1]:  # Most recent transcript
                        category = insight.get("category")
                        statement = insight.get("statement")
                        if category and statement:
                            simple_statement = re.sub(r'\b(the|a|an|and|or|in|on|at|to|for|with|by|of)\b', '', statement.lower())
                            simple_statement = re.sub(r'\s+', ' ', simple_statement).strip()
                            if simple_statement and f"{category}:{simple_statement}" == theme:
                                is_fading = False
                                break
                    
                    if is_fading:
                        category, statement = theme.split(':', 1)
                        trends["fading_topics"].append({
                            "category": category,
                            "statement": statement
                        })
        
        return {
            "ticker": ticker,
            "num_transcripts_analyzed": len(analyzed_transcripts),
            "date_range": {
                "start": analyzed_transcripts[0].get("date") if analyzed_transcripts else None,
                "end": analyzed_transcripts[-1].get("date") if analyzed_transcripts else None
            },
            "trends": trends
        }
