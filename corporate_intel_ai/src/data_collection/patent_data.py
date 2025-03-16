"""
Patent Data Collection Module

This module handles the collection of patent data from public patent databases.
It provides functionality to search, download, and parse patent information
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
import concurrent.futures

# Configure logging
logger = logging.getLogger(__name__)

class PatentDataCollector:
    """
    Class for collecting and processing patent data from public databases.
    """
    
    # Patent types
    PATENT_TYPES = {
        "utility": "Utility Patent",
        "design": "Design Patent",
        "plant": "Plant Patent",
        "reissue": "Reissue Patent",
        "defensive": "Defensive Publication",
        "statutory": "Statutory Invention Registration"
    }
    
    def __init__(self, api_key: Optional[str] = None, output_dir: str = "data/patents", 
                 use_cache: bool = True, cache_expiry_days: int = 30):
        """
        Initialize the patent data collector.
        
        Args:
            api_key: API key for premium patent data services (optional)
            output_dir: Directory to store downloaded patent data
            use_cache: Whether to use caching for downloaded patent data
            cache_expiry_days: Number of days after which cache entries expire
        """
        # USPTO API endpoints
        self.uspto_api_url = "https://developer.uspto.gov/ibd-api/v1/application/grants"
        # Google Patents API endpoint (unofficial)
        self.google_patents_url = "https://patents.google.com/xhr/query"
        
        self.api_key = api_key
        self.output_dir = output_dir
        self.use_cache = use_cache
        self.cache_expiry_days = cache_expiry_days
        self.cache_dir = os.path.join(output_dir, "_cache")
        
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
            self.headers["X-API-KEY"] = api_key
    
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
        Make a request to a patent API with rate limiting.
        
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
    
    def search_patents(self, 
                      company_name: Optional[str] = None,
                      inventor_name: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      keywords: Optional[List[str]] = None,
                      patent_type: Optional[str] = None,
                      max_results: int = 100) -> List[Dict]:
        """
        Search for patents based on company name and other criteria.
        
        Args:
            company_name: Name of the company/assignee
            inventor_name: Name of the inventor
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            keywords: List of keywords to search in patent text
            patent_type: Type of patent
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing patent metadata
        """
        # Validate inputs
        if not any([company_name, inventor_name, keywords]):
            raise ValueError("At least one of company_name, inventor_name, or keywords must be provided")
        
        # Create a cache key based on search parameters
        cache_params = f"{company_name}_{inventor_name}_{start_date}_{end_date}_{str(keywords)}_{patent_type}_{max_results}"
        cache_key = self._get_cache_key("search", cache_params)
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached patent search results for {cache_params}")
            return cached_result
        
        # Log search parameters
        search_params = []
        if company_name:
            search_params.append(f"company: {company_name}")
        if inventor_name:
            search_params.append(f"inventor: {inventor_name}")
        if keywords:
            search_params.append(f"keywords: {keywords}")
            
        logger.info(f"Searching for patents with parameters: {', '.join(search_params)}")
        
        # Try USPTO API first
        try:
            results = self._search_uspto_patents(
                company_name=company_name,
                inventor_name=inventor_name,
                start_date=start_date,
                end_date=end_date,
                keywords=keywords,
                patent_type=patent_type,
                max_results=max_results
            )
            
            # If we got results, cache and return them
            if results:
                self._save_to_cache(cache_key, results)
                return results
                
        except Exception as e:
            logger.warning(f"USPTO API search failed: {e}")
        
        # Fall back to Google Patents search
        try:
            results = self._search_google_patents(
                company_name=company_name,
                inventor_name=inventor_name,
                start_date=start_date,
                end_date=end_date,
                keywords=keywords,
                patent_type=patent_type,
                max_results=max_results
            )
            
            # Cache and return results
            self._save_to_cache(cache_key, results)
            return results
            
        except Exception as e:
            logger.error(f"Google Patents search failed: {e}")
            return []
    
    def _search_uspto_patents(self, company_name: Optional[str] = None, 
                             inventor_name: Optional[str] = None,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             keywords: Optional[List[str]] = None,
                             patent_type: Optional[str] = None,
                             max_results: int = 100) -> List[Dict]:
        """
        Search for patents using the USPTO API.
        
        Args:
            company_name: Name of the company/assignee
            inventor_name: Name of the inventor
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            keywords: List of keywords to search in patent text
            patent_type: Type of patent
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing patent metadata
        """
        # Construct query parameters for USPTO API
        query_params = {
            "start": 0,
            "rows": min(max_results, 100),  # USPTO API limits to 100 results per request
            "format": "json"
        }
        
        # Build query string
        query_parts = []
        
        if company_name:
            query_parts.append(f'assigneeEntityName:("{company_name}")')
        
        if inventor_name:
            query_parts.append(f'inventorName:("{inventor_name}")')
        
        if keywords:
            keyword_query = ' OR '.join([f'"{keyword}"' for keyword in keywords])
            query_parts.append(f'patentTitle:({keyword_query}) OR abstractText:({keyword_query})')
        
        if patent_type:
            query_parts.append(f'patentType:("{patent_type}")')
        
        if start_date and end_date:
            query_parts.append(f'grantDate:[{start_date}T00:00:00Z TO {end_date}T23:59:59Z]')
        
        # Combine all query parts with AND
        if query_parts:
            query_params["q"] = ' AND '.join(query_parts)
        
        # Make the request
        try:
            response = self._make_request(self.uspto_api_url, params=query_params)
            data = response.json()
            
            # Extract patent information
            patents = []
            for patent in data.get("results", []):
                patents.append({
                    "patent_number": patent.get("patentNumber"),
                    "title": patent.get("patentTitle"),
                    "abstract": patent.get("abstractText"),
                    "filing_date": patent.get("filingDate"),
                    "grant_date": patent.get("grantDate"),
                    "inventors": [inv.get("inventorFullName") for inv in patent.get("inventors", [])],
                    "assignees": [asg.get("assigneeEntityName") for asg in patent.get("assignees", [])],
                    "patent_type": patent.get("patentType"),
                    "source": "uspto"
                })
            
            return patents
            
        except Exception as e:
            logger.error(f"Error searching USPTO patents: {e}")
            return []
    
    def _search_google_patents(self, company_name: Optional[str] = None, 
                              inventor_name: Optional[str] = None,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              keywords: Optional[List[str]] = None,
                              patent_type: Optional[str] = None,
                              max_results: int = 100) -> List[Dict]:
        """
        Search for patents using Google Patents.
        
        Args:
            company_name: Name of the company/assignee
            inventor_name: Name of the inventor
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            keywords: List of keywords to search in patent text
            patent_type: Type of patent
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing patent metadata
        """
        # Build Google Patents query string
        query_parts = []
        
        if company_name:
            query_parts.append(f'assignee:"{company_name}"')
        
        if inventor_name:
            query_parts.append(f'inventor:"{inventor_name}"')
        
        if keywords:
            for keyword in keywords:
                query_parts.append(f'"{keyword}"')
        
        if patent_type:
            query_parts.append(f'type:"{patent_type}"')
        
        if start_date and end_date:
            query_parts.append(f'after:{start_date} before:{end_date}')
        
        # Combine query parts
        query = " ".join(query_parts)
        
        # Construct request parameters
        params = {
            "url": f"q={query}",
            "exp": "",
            "num": min(max_results, 100),  # Limit to 100 results per request
            "download": "false"
        }
        
        try:
            # Make the request
            response = self._make_request(
                self.google_patents_url, 
                method="POST", 
                data=params,
                headers={"Referer": "https://patents.google.com/"}
            )
            
            data = response.json()
            
            # Extract patent information
            patents = []
            for result in data.get("results", {}).get("cluster", []):
                patent_info = result.get("result", {})
                
                # Extract basic information
                patent = {
                    "patent_number": patent_info.get("patent", {}).get("publication_number"),
                    "title": patent_info.get("patent", {}).get("title"),
                    "abstract": patent_info.get("patent", {}).get("abstract"),
                    "filing_date": patent_info.get("patent", {}).get("filing_date"),
                    "grant_date": patent_info.get("patent", {}).get("grant_date"),
                    "inventors": [inv.get("name") for inv in patent_info.get("patent", {}).get("inventor", [])],
                    "assignees": [asg.get("name") for asg in patent_info.get("patent", {}).get("assignee", [])],
                    "patent_type": patent_info.get("patent", {}).get("type"),
                    "claims": [],
                    "description": ""
                }
                
                patents.append(patent)
            
            return patents
            
        except Exception as e:
            logger.error(f"Error searching Google Patents: {e}")
            return []
    
    def download_patent(self, patent_number: str, save_path: Optional[str] = None) -> str:
        """
        Download a specific patent by patent number.
        
        Args:
            patent_number: Patent number
            save_path: Optional path to save the patent data
            
        Returns:
            Path to the downloaded patent data
        """
        # Determine save path
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"patent_{patent_number}.json")
        
        # Check if file already exists
        if os.path.exists(save_path):
            logger.info(f"Patent already exists at {save_path}")
            return save_path
        
        # Try USPTO API first
        try:
            # Construct URL for the patent
            url = f"{self.uspto_api_url}/{patent_number}"
            
            # Make the request
            response = self._make_request(url)
            data = response.json()
            
            # Save the patent data
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Successfully downloaded patent data to {save_path}")
            return save_path
            
        except Exception as e:
            logger.warning(f"Error downloading patent from USPTO: {e}")
        
        # Fall back to Google Patents
        try:
            # Construct Google Patents URL
            url = f"https://patents.google.com/patent/{patent_number}/en"
            
            # Make the request
            response = requests.get(url, headers={"User-Agent": "Corporate Intelligence Automation/1.0"})
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract patent data
            patent_data = {
                "patent_number": patent_number,
                "title": soup.select_one("h1.title").text.strip() if soup.select_one("h1.title") else "",
                "abstract": soup.select_one("div.abstract").text.strip() if soup.select_one("div.abstract") else "",
                "description": soup.select_one("section.description").text.strip() if soup.select_one("section.description") else "",
                "claims": [claim.text.strip() for claim in soup.select("div.claims")],
                "inventors": [inv.text.strip() for inv in soup.select("dd.inventor")],
                "assignees": [asg.text.strip() for asg in soup.select("dd.assignee")],
                "filing_date": soup.select_one("time[itemprop='filingDate']").text.strip() if soup.select_one("time[itemprop='filingDate']") else "",
                "publication_date": soup.select_one("time[itemprop='publicationDate']").text.strip() if soup.select_one("time[itemprop='publicationDate']") else "",
                "source": "google_patents_html"
            }
            
            # Save the patent data
            with open(save_path, 'w') as f:
                json.dump(patent_data, f, indent=2)
            
            logger.info(f"Successfully downloaded patent data to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error downloading patent from Google Patents: {e}")
            raise
    
    def parse_patent(self, patent_path: str) -> Dict:
        """
        Parse downloaded patent data to extract structured information.
        
        Args:
            patent_path: Path to the downloaded patent data
            
        Returns:
            Dictionary containing parsed data from the patent
        """
        # Check cache first
        cache_key = self._get_cache_key("parsed", os.path.basename(patent_path))
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached parsed data for {patent_path}")
            return cached_result
        
        logger.info(f"Parsing patent data: {patent_path}")
        
        try:
            # Read the patent data
            with open(patent_path, 'r') as f:
                patent_data = json.load(f)
            
            # Determine source
            source = patent_data.get("source", "unknown")
            
            # Parse based on source
            if source == "uspto":
                parsed_data = self._parse_uspto_patent(patent_data)
            elif source in ["google_patents", "google_patents_html"]:
                parsed_data = self._parse_google_patent(patent_data)
            else:
                # Generic parsing
                parsed_data = {
                    "patent_number": patent_data.get("patent_number"),
                    "title": patent_data.get("title"),
                    "abstract": patent_data.get("abstract"),
                    "filing_date": patent_data.get("filing_date"),
                    "publication_date": patent_data.get("publication_date") or patent_data.get("grant_date"),
                    "inventors": patent_data.get("inventors", []),
                    "assignees": patent_data.get("assignees", []),
                    "claims": patent_data.get("claims", []),
                    "description": patent_data.get("description", "")
                }
            
            # Add source and path information
            parsed_data["source"] = source
            parsed_data["patent_path"] = patent_path
            
            # Cache the parsed results
            self._save_to_cache(cache_key, parsed_data)
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing patent {patent_path}: {e}")
            return {"status": "error", "error": str(e), "patent_path": patent_path}
    
    def _parse_uspto_patent(self, patent_data: Dict) -> Dict:
        """
        Parse USPTO patent data.
        
        Args:
            patent_data: Raw patent data from USPTO
            
        Returns:
            Dictionary containing parsed data
        """
        # Extract basic information
        parsed_data = {
            "patent_number": patent_data.get("patentNumber"),
            "title": patent_data.get("patentTitle"),
            "abstract": patent_data.get("abstractText"),
            "filing_date": patent_data.get("filingDate"),
            "publication_date": patent_data.get("grantDate"),
            "inventors": [inv.get("inventorFullName") for inv in patent_data.get("inventors", [])],
            "assignees": [asg.get("assigneeEntityName") for asg in patent_data.get("assignees", [])],
            "patent_type": patent_data.get("patentType"),
            "claims": [],
            "description": ""
        }
        
        # Extract claims
        claims_text = patent_data.get("claimsText", "")
        if claims_text:
            # Split claims by claim number pattern
            claim_pattern = r'(\d+\.\s+)'
            claims_parts = re.split(claim_pattern, claims_text)
            
            # Process claims parts
            claims = []
            for i in range(1, len(claims_parts), 2):
                claim_num = claims_parts[i].strip()
                claim_text = claims_parts[i+1].strip() if i+1 < len(claims_parts) else ""
                if claim_text:
                    claims.append({"number": claim_num, "text": claim_text})
            
            parsed_data["claims"] = claims
        
        # Extract description
        parsed_data["description"] = patent_data.get("descriptionText", "")
        
        return parsed_data
    
    def _parse_google_patent(self, patent_data: Dict) -> Dict:
        """
        Parse Google Patents data.
        
        Args:
            patent_data: Raw patent data from Google Patents
            
        Returns:
            Dictionary containing parsed data
        """
        # Check if this is HTML-parsed data or API data
        if "patent" in patent_data and isinstance(patent_data["patent"], dict):
            # API data
            patent_info = patent_data["patent"]
            
            parsed_data = {
                "patent_number": patent_info.get("publication_number"),
                "title": patent_info.get("title"),
                "abstract": patent_info.get("abstract"),
                "filing_date": patent_info.get("filing_date"),
                "publication_date": patent_info.get("grant_date") or patent_info.get("publication_date"),
                "inventors": [inv.get("name") for inv in patent_info.get("inventor", [])],
                "assignees": [asg.get("name") for asg in patent_info.get("assignee", [])],
                "patent_type": patent_info.get("type"),
                "claims": [{"number": str(i+1), "text": claim} for i, claim in enumerate(patent_info.get("claim", []))],
                "description": patent_info.get("description", "")
            }
        else:
            # HTML-parsed data
            parsed_data = {
                "patent_number": patent_data.get("patent_number"),
                "title": patent_data.get("title"),
                "abstract": patent_data.get("abstract"),
                "filing_date": patent_data.get("filing_date"),
                "publication_date": patent_data.get("publication_date"),
                "inventors": patent_data.get("inventors", []),
                "assignees": patent_data.get("assignees", []),
                "claims": [],
                "description": patent_data.get("description", "")
            }
            
            # Process claims
            raw_claims = patent_data.get("claims", [])
            if isinstance(raw_claims, list):
                parsed_data["claims"] = [{"number": str(i+1), "text": claim} for i, claim in enumerate(raw_claims)]
            else:
                # Try to parse claims from text
                claims_text = str(raw_claims)
                claim_pattern = r'(\d+\.\s+)([^0-9]+)'
                claims_matches = re.findall(claim_pattern, claims_text)
                
                parsed_data["claims"] = [
                    {"number": num.strip(), "text": text.strip()} 
                    for num, text in claims_matches
                ]
        
        return parsed_data

    def analyze_patent_trends(self, company_name: str, timeframe: str = "5y") -> pd.DataFrame:
        """
        Analyze patent filing trends for a company over time.
        
        Args:
            company_name: Name of the company
            timeframe: Timeframe for analysis (e.g., "1y", "5y", "10y")
            
        Returns:
            DataFrame with patent trend analysis
        """
        logger.info(f"Analyzing patent trends for {company_name} over {timeframe}")
        
        # Determine date range based on timeframe
        end_date = datetime.now()
        if timeframe.endswith('y'):
            years = int(timeframe[:-1])
            start_date = end_date - timedelta(days=years*365)
        elif timeframe.endswith('m'):
            months = int(timeframe[:-1])
            start_date = end_date - timedelta(days=months*30)
        else:
            raise ValueError(f"Invalid timeframe format: {timeframe}. Use format like '5y' or '6m'")
        
        # Format dates for search
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Search for patents in the timeframe
        patents = self.search_patents(
            company_name=company_name,
            start_date=start_date_str,
            end_date=end_date_str,
            max_results=1000
        )
        
        if not patents:
            logger.warning(f"No patents found for {company_name} in the specified timeframe")
            return pd.DataFrame()
        
        # Create DataFrame from patents
        df = pd.DataFrame(patents)
        
        # Convert dates to datetime
        date_columns = ['filing_date', 'grant_date', 'publication_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Use filing_date for trend analysis
        date_col = 'filing_date'
        if date_col not in df.columns or df[date_col].isna().all():
            # Fall back to grant_date if filing_date is not available
            date_col = 'grant_date' if 'grant_date' in df.columns else 'publication_date'
        
        # Group by year and month
        if date_col in df.columns and not df[date_col].isna().all():
            df['year'] = df[date_col].dt.year
            df['month'] = df[date_col].dt.month
            
            # Count patents by year and month
            trend_df = df.groupby(['year', 'month']).size().reset_index(name='count')
            
            # Create date column for plotting
            trend_df['date'] = pd.to_datetime(trend_df['year'].astype(str) + '-' + trend_df['month'].astype(str))
            trend_df = trend_df.sort_values('date')
            
            return trend_df
        else:
            logger.warning(f"No valid date information found in patents for {company_name}")
            return pd.DataFrame()
    
    def analyze_technology_areas(self, patents: List[Dict]) -> pd.DataFrame:
        """
        Analyze the technology areas covered by a set of patents.
        
        Args:
            patents: List of patent dictionaries from search_patents
            
        Returns:
            DataFrame with technology area analysis
        """
        if not patents:
            return pd.DataFrame()
        
        # Extract technology areas from patent data
        tech_areas = []
        
        for patent in patents:
            # Extract technology areas from patent classification codes
            # This is a simplified approach - in a real implementation,
            # we would map patent classification codes to technology areas
            
            # Extract keywords from title and abstract
            title = patent.get('title', '')
            abstract = patent.get('abstract', '')
            
            # Combine text for analysis
            text = f"{title} {abstract}"
            
            # Simple keyword-based technology area classification
            tech_keywords = {
                'artificial intelligence': ['ai', 'artificial intelligence', 'machine learning', 'neural network'],
                'blockchain': ['blockchain', 'distributed ledger', 'cryptocurrency'],
                'cloud computing': ['cloud', 'saas', 'paas', 'iaas'],
                'cybersecurity': ['security', 'encryption', 'firewall', 'authentication'],
                'internet of things': ['iot', 'internet of things', 'connected device'],
                'robotics': ['robot', 'automation', 'autonomous'],
                'biotechnology': ['biotech', 'genomic', 'crispr', 'biological'],
                'renewable energy': ['solar', 'wind', 'renewable', 'sustainable'],
                'semiconductor': ['semiconductor', 'transistor', 'integrated circuit'],
                'telecommunications': ['telecom', '5g', 'wireless', 'network protocol']
            }
            
            # Find matching technology areas
            patent_areas = []
            text_lower = text.lower()
            
            for area, keywords in tech_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    patent_areas.append(area)
            
            # If no specific areas identified, classify as 'other'
            if not patent_areas:
                patent_areas = ['other']
            
            # Add to results with patent info
            for area in patent_areas:
                tech_areas.append({
                    'patent_number': patent.get('patent_number'),
                    'title': title,
                    'technology_area': area,
                    'filing_date': patent.get('filing_date'),
                    'grant_date': patent.get('grant_date')
                })
        
        # Create DataFrame
        tech_df = pd.DataFrame(tech_areas)
        
        return tech_df
    
    def compare_patent_portfolios(self, companies: List[str], 
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Compare patent portfolios across multiple companies.
        
        Args:
            companies: List of company names to compare
            start_date: Start date for comparison (YYYY-MM-DD)
            end_date: End date for comparison (YYYY-MM-DD)
            
        Returns:
            Dictionary of DataFrames with comparison analyses
        """
        logger.info(f"Comparing patent portfolios for companies: {companies}")
        
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            # Default to 5 years
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        # Collect patents for each company
        company_patents = {}
        for company in companies:
            patents = self.search_patents(
                company_name=company,
                start_date=start_date,
                end_date=end_date,
                max_results=500
            )
            company_patents[company] = patents
            logger.info(f"Found {len(patents)} patents for {company}")
        
        # Create comparison analyses
        comparison = {}
        
        # 1. Patent count comparison
        patent_counts = {company: len(patents) for company, patents in company_patents.items()}
        comparison['patent_counts'] = pd.DataFrame(list(patent_counts.items()), 
                                                 columns=['Company', 'Patent Count'])
        
        # 2. Technology area comparison
        tech_areas_by_company = {}
        for company, patents in company_patents.items():
            tech_df = self.analyze_technology_areas(patents)
            if not tech_df.empty:
                # Count patents by technology area
                tech_counts = tech_df['technology_area'].value_counts().reset_index()
                tech_counts.columns = ['Technology Area', 'Patent Count']
                tech_counts['Company'] = company
                tech_areas_by_company[company] = tech_counts
        
        # Combine technology area data
        if tech_areas_by_company:
            comparison['technology_areas'] = pd.concat(tech_areas_by_company.values())
        
        # 3. Filing trend comparison
        filing_trends = {}
        for company, patents in company_patents.items():
            if patents:
                df = pd.DataFrame(patents)
                # Convert dates to datetime
                for date_col in ['filing_date', 'grant_date', 'publication_date']:
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                # Use filing_date for trend analysis
                date_col = 'filing_date'
                if date_col not in df.columns or df[date_col].isna().all():
                    date_col = 'grant_date' if 'grant_date' in df.columns else 'publication_date'
                
                if date_col in df.columns and not df[date_col].isna().all():
                    df['year'] = df[date_col].dt.year
                    yearly_counts = df['year'].value_counts().sort_index().reset_index()
                    yearly_counts.columns = ['Year', 'Patent Count']
                    yearly_counts['Company'] = company
                    filing_trends[company] = yearly_counts
        
        # Combine filing trend data
        if filing_trends:
            comparison['filing_trends'] = pd.concat(filing_trends.values())
        
        return comparison
    
    def batch_process_companies(self, companies: List[str], form_types: List[str] = None, 
                               start_date: Optional[str] = None, 
                               end_date: Optional[str] = None,
                               max_workers: int = 5) -> Dict[str, List[Dict]]:
        """
        Process multiple companies in parallel to collect patent data.
        
        Args:
            companies: List of company names to process
            form_types: List of patent types to search for (optional)
            start_date: Start date for search (YYYY-MM-DD)
            end_date: End date for search (YYYY-MM-DD)
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping company names to their patent data
        """
        logger.info(f"Batch processing {len(companies)} companies for patent data")
        
        results = {}
        
        # Process companies in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create future tasks
            future_to_company = {
                executor.submit(
                    self.search_patents,
                    company_name=company,
                    patent_type=form_types[0] if form_types else None,
                    start_date=start_date,
                    end_date=end_date
                ): company for company in companies
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_company):
                company = future_to_company[future]
                try:
                    patents = future.result()
                    results[company] = patents
                    logger.info(f"Completed processing {company}: found {len(patents)} patents")
                except Exception as e:
                    logger.error(f"Error processing {company}: {e}")
                    results[company] = []
        
        return results
    
    def extract_key_inventors(self, patents: List[Dict], top_n: int = 10) -> pd.DataFrame:
        """
        Extract and rank key inventors from a set of patents.
        
        Args:
            patents: List of patent dictionaries
            top_n: Number of top inventors to return
            
        Returns:
            DataFrame with inventor rankings
        """
        if not patents:
            return pd.DataFrame()
        
        # Extract all inventors
        all_inventors = []
        for patent in patents:
            inventors = patent.get('inventors', [])
            if inventors:
                for inventor in inventors:
                    all_inventors.append({
                        'name': inventor,
                        'patent_number': patent.get('patent_number'),
                        'patent_title': patent.get('title'),
                        'filing_date': patent.get('filing_date')
                    })
        
        if not all_inventors:
            return pd.DataFrame()
        
        # Create DataFrame
        inventors_df = pd.DataFrame(all_inventors)
        
        # Count patents by inventor
        inventor_counts = inventors_df['name'].value_counts().reset_index()
        inventor_counts.columns = ['Inventor', 'Patent Count']
        
        # Get top N inventors
        top_inventors = inventor_counts.head(top_n)
        
        return top_inventors
    
    def extract_patent_citations(self, patent_number: str) -> Dict:
        """
        Extract citation information for a specific patent.
        
        Args:
            patent_number: Patent number to analyze
            
        Returns:
            Dictionary with citation information
        """
        logger.info(f"Extracting citation information for patent {patent_number}")
        
        # Create cache key
        cache_key = self._get_cache_key("citations", patent_number)
        
        # Check cache
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            # Construct Google Patents URL
            url = f"https://patents.google.com/patent/{patent_number}/en"
            
            # Make the request
            response = requests.get(url, headers={"User-Agent": "Corporate Intelligence Automation/1.0"})
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract citation information
            citations = {
                'cited_by': [],
                'cites': []
            }
            
            # Extract patents that cite this patent
            cited_by_section = soup.select_one("section[itemprop='citedBy']")
            if cited_by_section:
                for cite in cited_by_section.select("tr"):
                    patent_link = cite.select_one("a[href*='/patent/']")
                    if patent_link:
                        patent_id = patent_link.get('href').split('/patent/')[1].split('/')[0]
                        citations['cited_by'].append({
                            'patent_number': patent_id,
                            'title': patent_link.text.strip()
                        })
            
            # Extract patents cited by this patent
            cites_section = soup.select_one("section[itemprop='cites']")
            if cites_section:
                for cite in cites_section.select("tr"):
                    patent_link = cite.select_one("a[href*='/patent/']")
                    if patent_link:
                        patent_id = patent_link.get('href').split('/patent/')[1].split('/')[0]
                        citations['cites'].append({
                            'patent_number': patent_id,
                            'title': patent_link.text.strip()
                        })
            
            # Save to cache
            self._save_to_cache(cache_key, citations)
            
            return citations
            
        except Exception as e:
            logger.error(f"Error extracting citations for patent {patent_number}: {e}")
            return {'cited_by': [], 'cites': []}
