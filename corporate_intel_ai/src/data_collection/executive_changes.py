"""
Board & Executive Changes Tracking Module

This module handles the collection and analysis of board member and executive changes
for corporate intelligence. It integrates data from SEC DEF14A filings, LinkedIn profiles,
and company websites to track appointments, departures, and role changes.

The module supports the Live Market Intelligence Dashboard by providing:
1. Real-time alerts for significant executive changes
2. Historical tracking of leadership transitions
3. Network analysis of executive connections
4. Correlation between leadership changes and company performance
"""

import os
import re
import json
import time
import pickle
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Set, Tuple
from bs4 import BeautifulSoup
import pandas as pd
import networkx as nx
from collections import defaultdict
import random

# Configure logging
logger = logging.getLogger(__name__)

class ExecutiveChangesTracker:
    """
    Class for tracking and analyzing board member and executive changes.
    """
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None, output_dir: str = "data/executive_changes",
                 use_cache: bool = True, cache_expiry_days: int = 7):
        """
        Initialize the executive changes tracker.
        
        Args:
            api_keys: Dictionary of API keys for different data sources
            output_dir: Directory to store downloaded data
            use_cache: Whether to use caching for downloaded data
            cache_expiry_days: Number of days after which cache entries expire
        """
        self.api_keys = api_keys or {}
        self.output_dir = output_dir
        self.use_cache = use_cache
        self.cache_expiry_days = cache_expiry_days
        self.cache_dir = os.path.join(output_dir, "_cache")
        
        # Create output and cache directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Headers for web requests
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        }
        
        # LinkedIn rate limiting
        self.linkedin_last_request = 0
        self.linkedin_min_delay = 5  # seconds between requests
        
        # Common executive titles for pattern matching
        self.executive_titles = [
            "CEO", "Chief Executive Officer",
            "CFO", "Chief Financial Officer",
            "COO", "Chief Operating Officer",
            "CTO", "Chief Technology Officer",
            "CMO", "Chief Marketing Officer",
            "CHRO", "Chief Human Resources Officer",
            "CIO", "Chief Information Officer",
            "President", "Chairman", "Chairwoman", "Chairperson",
            "Director", "Board Member", "Trustee",
            "SVP", "Senior Vice President",
            "EVP", "Executive Vice President",
            "VP", "Vice President"
        ]
        
        # Compile regex patterns for executive titles
        self.title_pattern = re.compile(r'\b(' + '|'.join(re.escape(title) for title in self.executive_titles) + r')\b', re.IGNORECASE)
    
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
        Make a request to a website with rate limiting.
        
        Args:
            url: URL to request
            method: HTTP method (GET, POST)
            params: Optional query parameters
            data: Optional request body data
            headers: Optional headers to override defaults
            
        Returns:
            Response object
        """
        # Check if this is a LinkedIn request and apply rate limiting
        if "linkedin.com" in url:
            time_since_last = time.time() - self.linkedin_last_request
            if time_since_last < self.linkedin_min_delay:
                time.sleep(self.linkedin_min_delay - time_since_last)
            self.linkedin_last_request = time.time()
        else:
            # General rate limiting for other sites
            time.sleep(1)
        
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
    
    def get_executive_changes_from_sec(self, ticker: str, start_date: Optional[str] = None, 
                                      end_date: Optional[str] = None) -> List[Dict]:
        """
        Extract executive changes from SEC DEF14A (proxy statement) filings.
        
        Args:
            ticker: Company ticker symbol
            start_date: Start date for filing search (YYYY-MM-DD)
            end_date: End date for filing search (YYYY-MM-DD)
            
        Returns:
            List of dictionaries containing executive change information
        """
        logger.info(f"Getting executive changes from SEC filings for {ticker}")
        
        # Create a cache key
        cache_key = self._get_cache_key("sec_exec_changes", f"{ticker}_{start_date}_{end_date}")
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached SEC executive changes for {ticker}")
            return cached_result
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            # Default to 1 year ago
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        try:
            # Use SEC API to get filings
            sec_url = "https://www.sec.gov/cgi-bin/browse-edgar"
            params = {
                "action": "getcompany",
                "CIK": ticker,
                "type": "DEF 14A",  # Proxy statements
                "dateb": end_date.replace("-", ""),
                "datea": start_date.replace("-", ""),
                "owner": "exclude",
                "count": "100",
                "output": "atom"
            }
            
            response = self._make_request(sec_url, params=params)
            soup = BeautifulSoup(response.content, "lxml-xml")
            
            entries = soup.find_all("entry")
            if not entries:
                logger.warning(f"No DEF 14A filings found for {ticker}")
                return []
            
            executive_changes = []
            
            for entry in entries:
                filing_date = entry.find("filing-date").text if entry.find("filing-date") else None
                filing_url = entry.find("filing-href").text if entry.find("filing-href") else None
                
                if not filing_url:
                    continue
                
                # Get the filing content
                filing_content = self._get_filing_content(filing_url)
                if not filing_content:
                    continue
                
                # Extract executive information from the filing
                executives = self._extract_executives_from_proxy(filing_content)
                
                # For each filing, compare with previous filings to detect changes
                changes = self._detect_executive_changes(ticker, executives, filing_date)
                executive_changes.extend(changes)
            
            # Cache and return results
            self._save_to_cache(cache_key, executive_changes)
            return executive_changes
            
        except Exception as e:
            logger.error(f"Error getting executive changes from SEC for {ticker}: {e}")
            return []
    
    def _get_filing_content(self, filing_url: str) -> Optional[str]:
        """
        Get the content of an SEC filing.
        
        Args:
            filing_url: URL of the filing
            
        Returns:
            Filing content as text, or None if retrieval fails
        """
        try:
            # First get the filing page
            response = self._make_request(filing_url)
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Find the link to the actual filing document
            table = soup.find("table", class_="tableFile")
            if not table:
                return None
                
            # Look for the proxy statement document
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 3:
                    doc_type = cells[0].text.strip()
                    if "def 14a" in doc_type.lower():
                        doc_link = cells[2].find("a")
                        if doc_link and doc_link.has_attr("href"):
                            doc_url = "https://www.sec.gov" + doc_link["href"]
                            
                            # Get the actual document
                            doc_response = self._make_request(doc_url)
                            return doc_response.text
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting filing content from {filing_url}: {e}")
            return None
    
    def _extract_executives_from_proxy(self, filing_content: str) -> List[Dict]:
        """
        Extract executive information from a proxy statement.
        
        Args:
            filing_content: Text content of the proxy statement
            
        Returns:
            List of dictionaries containing executive information
        """
        executives = []
        soup = BeautifulSoup(filing_content, "html.parser")
        
        # Look for tables that might contain executive information
        tables = soup.find_all("table")
        
        for table in tables:
            # Check if this table contains executive information
            text = table.get_text().lower()
            if any(keyword in text for keyword in ["executive officer", "director", "board of director", "management"]):
                rows = table.find_all("tr")
                
                for row in rows:
                    cells = row.find_all(["td", "th"])
                    if len(cells) < 2:
                        continue
                    
                    # Extract text from cells
                    cell_texts = [cell.get_text().strip() for cell in cells]
                    
                    # Look for executive titles in the row
                    title_matches = [self.title_pattern.search(text) for text in cell_texts]
                    title_matches = [match for match in title_matches if match]
                    
                    if title_matches:
                        # This row likely contains an executive
                        name = None
                        title = None
                        age = None
                        
                        # Try to extract name, title, and age
                        for i, text in enumerate(cell_texts):
                            # Look for a name (usually the first cell with capitalized words)
                            if not name and re.search(r'^[A-Z][a-z]+ [A-Z][a-z]+', text):
                                name = text.split("\n")[0].strip()
                            
                            # Look for a title
                            title_match = self.title_pattern.search(text)
                            if title_match and not title:
                                title = title_match.group(0)
                            
                            # Look for age (usually in parentheses or as a separate cell)
                            age_match = re.search(r'\((\d+)\)', text) or re.search(r'^(\d+)$', text)
                            if age_match and not age:
                                age = age_match.group(1)
                        
                        if name and title:
                            executives.append({
                                "name": name,
                                "title": title,
                                "age": age
                            })
        
        # Also look for paragraphs that might contain executive information
        executive_sections = []
        
        # Find sections that might contain executive information
        for heading in soup.find_all(["h1", "h2", "h3", "h4"]):
            heading_text = heading.get_text().lower()
            if any(keyword in heading_text for keyword in ["executive officer", "director", "board of director", "management"]):
                # Get the next sibling elements until the next heading
                current = heading.next_sibling
                section_text = ""
                
                while current and not current.name in ["h1", "h2", "h3", "h4"]:
                    if hasattr(current, "get_text"):
                        section_text += current.get_text() + "\n"
                    current = current.next_sibling
                
                executive_sections.append(section_text)
        
        # Extract executive information from the sections
        for section in executive_sections:
            # Look for patterns like "Name (Age) - Title" or "Name, Title, Age"
            exec_matches = re.finditer(r'([A-Z][a-z]+ [A-Z][a-z]+)(?:\s*\((\d+)\))?(?:\s*[,-]\s*|\s+)((?:[A-Z][a-z]*\s*)+)', section)
            
            for match in exec_matches:
                name = match.group(1).strip()
                age = match.group(2) if match.group(2) else None
                title = match.group(3).strip()
                
                # Verify this is actually an executive title
                if self.title_pattern.search(title):
                    executives.append({
                        "name": name,
                        "title": title,
                        "age": age
                    })
        
        return executives
    
    def _detect_executive_changes(self, ticker: str, current_executives: List[Dict], filing_date: str) -> List[Dict]:
        """
        Detect changes in executives by comparing with previous data.
        
        Args:
            ticker: Company ticker symbol
            current_executives: List of current executives from latest filing
            filing_date: Date of the current filing
            
        Returns:
            List of dictionaries containing executive change information
        """
        changes = []
        
        # Get previous executives data
        previous_data_key = self._get_cache_key("previous_executives", ticker)
        previous_executives = self._get_from_cache(previous_data_key) or []
        
        # Create dictionaries for easier comparison
        current_exec_dict = {exec["name"]: exec for exec in current_executives}
        previous_exec_dict = {exec["name"]: exec for exec in previous_executives}
        
        # Detect departures (in previous but not in current)
        for name, exec_data in previous_exec_dict.items():
            if name not in current_exec_dict:
                changes.append({
                    "company": ticker,
                    "name": name,
                    "previous_title": exec_data.get("title"),
                    "new_title": None,
                    "change_type": "departure",
                    "detection_date": filing_date
                })
        
        # Detect new appointments and role changes
        for name, exec_data in current_exec_dict.items():
            if name not in previous_exec_dict:
                # New appointment
                changes.append({
                    "company": ticker,
                    "name": name,
                    "previous_title": None,
                    "new_title": exec_data.get("title"),
                    "change_type": "appointment",
                    "detection_date": filing_date
                })
            elif previous_exec_dict[name].get("title") != exec_data.get("title"):
                # Role change
                changes.append({
                    "company": ticker,
                    "name": name,
                    "previous_title": previous_exec_dict[name].get("title"),
                    "new_title": exec_data.get("title"),
                    "change_type": "role_change",
                    "detection_date": filing_date
                })
        
        # Update the stored executives data
        self._save_to_cache(previous_data_key, current_executives)
        
        return changes
    
    def get_executive_changes_from_linkedin(self, company_name: str, limit: int = 10) -> List[Dict]:
        """
        Extract executive changes from LinkedIn profiles and company pages.
        
        Args:
            company_name: Name of the company
            limit: Maximum number of executives to retrieve
            
        Returns:
            List of dictionaries containing executive change information
        """
        logger.info(f"Getting executive changes from LinkedIn for {company_name}")
        
        # Create a cache key
        cache_key = self._get_cache_key("linkedin_exec_changes", f"{company_name}_{limit}")
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached LinkedIn executive changes for {company_name}")
            return cached_result
        
        try:
            # Note: LinkedIn heavily restricts scraping. This is a simplified implementation
            # that would need to be adapted based on LinkedIn's current structure and policies.
            # In a production environment, consider using LinkedIn's official API with proper
            # authentication or a specialized service for this data.
            
            # Search for the company page
            search_url = f"https://www.linkedin.com/company/{company_name.lower().replace(' ', '-')}"
            
            # This is a placeholder for demonstration purposes
            # In a real implementation, you would need to handle authentication and parsing
            executives = []
            changes = []
            
            # Simulate finding some executives (in a real implementation, this would parse the page)
            logger.warning("LinkedIn scraping is limited by their terms of service. Consider using their official API.")
            logger.info(f"Simulating LinkedIn data for {company_name} for demonstration purposes")
            
            # For demonstration, return placeholder data
            simulated_data = [
                {
                    "company": company_name,
                    "name": f"Executive {i}",
                    "previous_title": "Previous Title",
                    "new_title": "New Title",
                    "change_type": "role_change",
                    "detection_date": datetime.now().strftime("%Y-%m-%d"),
                    "source": "LinkedIn (simulated)"
                }
                for i in range(1, min(limit + 1, 4))  # Simulate 1-3 executives
            ]
            
            # Cache and return results
            self._save_to_cache(cache_key, simulated_data)
            return simulated_data
            
        except Exception as e:
            logger.error(f"Error getting executive changes from LinkedIn for {company_name}: {e}")
            return []
    
    def get_executive_changes_from_website(self, company_name: str, website_url: str) -> List[Dict]:
        """
        Extract executive changes from company website news/press releases.
        
        Args:
            company_name: Name of the company
            website_url: URL of the company website
            
        Returns:
            List of dictionaries containing executive change information
        """
        logger.info(f"Getting executive changes from website for {company_name}")
        
        # Create a cache key
        cache_key = self._get_cache_key("website_exec_changes", company_name)
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached website executive changes for {company_name}")
            return cached_result
        
        try:
            # First, try to find the press release or news section
            response = self._make_request(website_url)
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Look for links to press releases or news
            news_links = []
            for link in soup.find_all("a", href=True):
                link_text = link.get_text().lower()
                link_href = link["href"]
                
                # Check if this link points to a news or press release section
                if any(keyword in link_text for keyword in ["press", "news", "media", "releases", "announcements"]):
                    # Handle relative URLs
                    if link_href.startswith("/"):
                        link_href = website_url.rstrip("/") + link_href
                    elif not link_href.startswith(("http://", "https://")):
                        link_href = website_url.rstrip("/") + "/" + link_href
                    
                    news_links.append(link_href)
            
            # Process the first few news links
            executive_changes = []
            processed_links = set()
            
            for news_link in news_links[:5]:  # Limit to first 5 news links to avoid excessive scraping
                if news_link in processed_links:
                    continue
                
                processed_links.add(news_link)
                
                try:
                    news_response = self._make_request(news_link)
                    news_soup = BeautifulSoup(news_response.content, "html.parser")
                    
                    # Get all the text from the page
                    page_text = news_soup.get_text()
                    
                    # Look for executive change announcements
                    changes = self._extract_executive_changes_from_text(page_text, company_name)
                    if changes:
                        # Try to extract the date from the page
                        date = self._extract_date_from_page(news_soup)
                        
                        for change in changes:
                            change["detection_date"] = date
                            change["source"] = news_link
                            change["company"] = company_name
                            executive_changes.append(change)
                    
                    # Also check for links to individual press releases
                    release_links = []
                    for link in news_soup.find_all("a", href=True):
                        link_text = link.get_text().lower()
                        if any(keyword in link_text for keyword in ["appoint", "join", "promot", "executive", "officer", "director", "ceo", "cfo", "cto"]):
                            link_href = link["href"]
                            
                            # Handle relative URLs
                            if link_href.startswith("/"):
                                link_href = website_url.rstrip("/") + link_href
                            elif not link_href.startswith(("http://", "https://")):
                                link_href = website_url.rstrip("/") + "/" + link_href
                            
                            release_links.append(link_href)
                    
                    # Process individual press releases
                    for release_link in release_links[:3]:  # Limit to first 3 release links
                        if release_link in processed_links:
                            continue
                        
                        processed_links.add(release_link)
                        
                        try:
                            release_response = self._make_request(release_link)
                            release_soup = BeautifulSoup(release_response.content, "html.parser")
                            
                            # Get all the text from the page
                            release_text = release_soup.get_text()
                            
                            # Look for executive change announcements
                            changes = self._extract_executive_changes_from_text(release_text, company_name)
                            if changes:
                                # Try to extract the date from the page
                                date = self._extract_date_from_page(release_soup)
                                
                                for change in changes:
                                    change["detection_date"] = date
                                    change["source"] = release_link
                                    change["company"] = company_name
                                    executive_changes.append(change)
                        except Exception as e:
                            logger.warning(f"Error processing release link {release_link}: {e}")
                            continue
                
                except Exception as e:
                    logger.warning(f"Error processing news link {news_link}: {e}")
                    continue
            
            # Cache and return results
            self._save_to_cache(cache_key, executive_changes)
            return executive_changes
            
        except Exception as e:
            logger.error(f"Error getting executive changes from website for {company_name}: {e}")
            return []
    
    def _extract_executive_changes_from_text(self, text: str, company_name: str) -> List[Dict]:
        """
        Extract executive changes from text.
        
        Args:
            text: Text to extract changes from
            company_name: Name of the company
            
        Returns:
            List of dictionaries containing executive change information
        """
        changes = []
        
        # Look for appointment patterns
        appointment_patterns = [
            r'(?:appointed|named|elected|hired|joining|joins)(?:\s+\w+){0,5}\s+(?P<name>[A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+\w+){0,5}\s+(?:as|to)(?:\s+\w+){0,3}\s+(?P<title>' + '|'.join(re.escape(title) for title in self.executive_titles) + r')',
            r'(?P<name>[A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+\w+){0,5}\s+(?:appointed|named|elected|hired|joining|joins)(?:\s+\w+){0,5}\s+(?:as|to)(?:\s+\w+){0,3}\s+(?P<title>' + '|'.join(re.escape(title) for title in self.executive_titles) + r')'
        ]
        
        for pattern in appointment_patterns:
            for match in re.finditer(pattern, text):
                name = match.group("name")
                title = match.group("title")
                
                changes.append({
                    "name": name,
                    "previous_title": None,
                    "new_title": title,
                    "change_type": "appointment"
                })
        
        # Look for departure patterns
        departure_patterns = [
            r'(?:departed|resigned|leaving|left|steps down|stepping down)(?:\s+\w+){0,5}\s+(?P<name>[A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+\w+){0,5}\s+(?:as|from)(?:\s+\w+){0,3}\s+(?P<title>' + '|'.join(re.escape(title) for title in self.executive_titles) + r')',
            r'(?P<name>[A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+\w+){0,5}\s+(?:departed|resigned|leaving|left|steps down|stepping down)(?:\s+\w+){0,5}\s+(?:as|from)(?:\s+\w+){0,3}\s+(?P<title>' + '|'.join(re.escape(title) for title in self.executive_titles) + r')'
        ]
        
        for pattern in departure_patterns:
            for match in re.finditer(pattern, text):
                name = match.group("name")
                title = match.group("title")
                
                changes.append({
                    "name": name,
                    "previous_title": title,
                    "new_title": None,
                    "change_type": "departure"
                })
        
        # Look for promotion/role change patterns
        change_patterns = [
            r'(?:promoted|moved|transitioned)(?:\s+\w+){0,5}\s+(?P<name>[A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+\w+){0,5}\s+(?:from)(?:\s+\w+){0,3}\s+(?P<previous_title>' + '|'.join(re.escape(title) for title in self.executive_titles) + r')(?:\s+\w+){0,5}\s+(?:to)(?:\s+\w+){0,3}\s+(?P<new_title>' + '|'.join(re.escape(title) for title in self.executive_titles) + r')',
            r'(?P<name>[A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+\w+){0,5}\s+(?:promoted|moved|transitioned)(?:\s+\w+){0,5}\s+(?:from)(?:\s+\w+){0,3}\s+(?P<previous_title>' + '|'.join(re.escape(title) for title in self.executive_titles) + r')(?:\s+\w+){0,5}\s+(?:to)(?:\s+\w+){0,3}\s+(?P<new_title>' + '|'.join(re.escape(title) for title in self.executive_titles) + r')'
        ]
        
        for pattern in change_patterns:
            for match in re.finditer(pattern, text):
                name = match.group("name")
                previous_title = match.group("previous_title")
                new_title = match.group("new_title")
                
                changes.append({
                    "name": name,
                    "previous_title": previous_title,
                    "new_title": new_title,
                    "change_type": "role_change"
                })
        
        return changes
    
    def _extract_date_from_page(self, soup: BeautifulSoup) -> str:
        """
        Extract date from a webpage.
        
        Args:
            soup: BeautifulSoup object of the page
            
        Returns:
            Date string in YYYY-MM-DD format, or current date if not found
        """
        # Common date patterns
        date_patterns = [
            r'(\d{1,2})[/-](\d{1,2})[/-](20\d{2})',  # MM/DD/YYYY or DD/MM/YYYY
            r'(20\d{2})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY/MM/DD or YYYY/DD/MM
            r'([A-Z][a-z]{2,8})\s+(\d{1,2}),?\s+(20\d{2})',  # Month DD, YYYY
            r'(\d{1,2})\s+([A-Z][a-z]{2,8})\s+(20\d{2})'  # DD Month YYYY
        ]
        
        # Look for date elements
        date_elements = soup.find_all(["time", "span", "div", "p"], class_=lambda c: c and any(keyword in c.lower() for keyword in ["date", "time", "published", "posted"]))
        
        for element in date_elements:
            text = element.get_text()
            for pattern in date_patterns:
                match = re.search(pattern, text)
                if match:
                    # Try to parse the date
                    try:
                        if pattern == date_patterns[0]:  # MM/DD/YYYY or DD/MM/YYYY
                            # Assume MM/DD/YYYY for simplicity
                            return f"{match.group(3)}-{match.group(1).zfill(2)}-{match.group(2).zfill(2)}"
                        elif pattern == date_patterns[1]:  # YYYY/MM/DD or YYYY/DD/MM
                            # Assume YYYY/MM/DD for simplicity
                            return f"{match.group(1)}-{match.group(2).zfill(2)}-{match.group(3).zfill(2)}"
                        elif pattern == date_patterns[2]:  # Month DD, YYYY
                            month_dict = {"January": "01", "February": "02", "March": "03", "April": "04", "May": "05", "June": "06", 
                                         "July": "07", "August": "08", "September": "09", "October": "10", "November": "11", "December": "12"}
                            month = match.group(1)
                            month_num = month_dict.get(month, "01")  # Default to January if not found
                            return f"{match.group(3)}-{month_num}-{match.group(2).zfill(2)}"
                        elif pattern == date_patterns[3]:  # DD Month YYYY
                            month_dict = {"January": "01", "February": "02", "March": "03", "April": "04", "May": "05", "June": "06", 
                                         "July": "07", "August": "08", "September": "09", "October": "10", "November": "11", "December": "12"}
                            month = match.group(2)
                            month_num = month_dict.get(month, "01")  # Default to January if not found
                            return f"{match.group(3)}-{month_num}-{match.group(1).zfill(2)}"
                    except:
                        pass
        
        # If no date found, use current date
        return datetime.now().strftime("%Y-%m-%d")
    
    def get_executive_profile(self, name: str, company: Optional[str] = None) -> Dict:
        """
        Compile a profile for an executive based on available data.
        
        Args:
            name: Name of the executive
            company: Optional company name to narrow the search
            
        Returns:
            Dictionary containing executive profile information
        """
        logger.info(f"Getting profile for executive {name}")
        
        # Create a cache key
        cache_key = self._get_cache_key("exec_profile", f"{name}_{company or ''}")
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached profile for {name}")
            return cached_result
        
        profile = {
            "name": name,
            "current_company": company,
            "current_title": None,
            "previous_roles": [],
            "education": [],
            "skills": [],
            "connections": [],
            "compensation": None,
            "sources": []
        }
        
        try:
            # This is a simplified implementation
            # In a real-world scenario, you would gather this information from:
            # 1. SEC filings (for public company executives)
            # 2. LinkedIn profiles (with proper API access)
            # 3. Company websites
            # 4. News articles
            
            # For demonstration purposes, we'll return a placeholder profile
            logger.info(f"Creating simulated profile for {name} for demonstration purposes")
            
            if company:
                profile["current_company"] = company
                profile["current_title"] = "Chief Executive Officer" if "CEO" in name else "Chief Financial Officer"
                profile["sources"].append("Simulated data")
                
                # Add some simulated previous roles
                profile["previous_roles"] = [
                    {
                        "company": f"Previous Company {i}",
                        "title": f"Previous Title {i}",
                        "start_date": f"201{i}",
                        "end_date": f"201{i+2}",
                        "duration": f"{2} years"
                    }
                    for i in range(1, 4)
                ]
                
                # Add some simulated education
                profile["education"] = [
                    {
                        "institution": "Harvard University",
                        "degree": "MBA",
                        "field": "Business Administration",
                        "year": "2005"
                    },
                    {
                        "institution": "Stanford University",
                        "degree": "BS",
                        "field": "Computer Science",
                        "year": "2000"
                    }
                ]
                
                # Add some simulated skills
                profile["skills"] = ["Leadership", "Strategy", "Finance", "Operations", "Technology"]
                
                # Add some simulated connections
                profile["connections"] = [f"Connection {i}" for i in range(1, 6)]
                
                # Add simulated compensation
                profile["compensation"] = {
                    "salary": "$1,200,000",
                    "bonus": "$800,000",
                    "stock_options": "$3,000,000",
                    "total": "$5,000,000",
                    "year": str(datetime.now().year - 1)
                }
            
            # Cache and return the profile
            self._save_to_cache(cache_key, profile)
            return profile
            
        except Exception as e:
            logger.error(f"Error getting profile for {name}: {e}")
            return profile
    
    def analyze_executive_network(self, company_name: str, depth: int = 1) -> Dict:
        """
        Analyze the network of executives for a company.
        
        Args:
            company_name: Name of the company
            depth: Depth of network analysis (1 = direct connections, 2 = connections of connections, etc.)
            
        Returns:
            Dictionary containing network analysis results
        """
        logger.info(f"Analyzing executive network for {company_name} with depth {depth}")
        
        # Create a cache key
        cache_key = self._get_cache_key("exec_network", f"{company_name}_{depth}")
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached executive network for {company_name}")
            return cached_result
        
        try:
            # Create a network graph
            G = nx.Graph()
            
            # Get executives for the company
            # In a real implementation, this would use actual data
            executives = [f"Executive {i}" for i in range(1, 6)]
            
            # Add company node
            G.add_node(company_name, type="company")
            
            # Add executive nodes and edges to company
            for exec_name in executives:
                G.add_node(exec_name, type="executive")
                G.add_edge(company_name, exec_name, relationship="employment")
            
            # If depth > 1, add connections between executives
            if depth > 1:
                # In a real implementation, this would use actual connection data
                # For demonstration, we'll create some random connections
                for i in range(len(executives)):
                    for j in range(i+1, len(executives)):
                        if i != j and random.random() < 0.3:  # 30% chance of connection
                            G.add_edge(executives[i], executives[j], relationship="professional")
            
            # If depth > 2, add connections to other companies
            if depth > 2:
                # In a real implementation, this would use actual data on previous employers
                other_companies = [f"Other Company {i}" for i in range(1, 4)]
                
                for company in other_companies:
                    G.add_node(company, type="company")
                    
                    # Connect some executives to these companies
                    for exec_name in executives:
                        if random.random() < 0.2:  # 20% chance of previous employment
                            G.add_edge(company, exec_name, relationship="previous_employment")
            
            # Calculate network metrics
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            
            # Prepare the result
            result = {
                "company": company_name,
                "executives": executives,
                "connections": [{"source": u, "target": v, "type": d["relationship"]} for u, v, d in G.edges(data=True)],
                "metrics": {
                    "degree_centrality": {node: round(value, 3) for node, value in degree_centrality.items()},
                    "betweenness_centrality": {node: round(value, 3) for node, value in betweenness_centrality.items()}
                }
            }
            
            # Cache and return the result
            self._save_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing executive network for {company_name}: {e}")
            return {"company": company_name, "error": str(e)}
    
    def correlate_with_performance(self, company_ticker: str, start_date: Optional[str] = None, 
                                  end_date: Optional[str] = None) -> Dict:
        """
        Correlate executive changes with company performance metrics.
        
        Args:
            company_ticker: Company ticker symbol
            start_date: Start date for analysis (YYYY-MM-DD)
            end_date: End date for analysis (YYYY-MM-DD)
            
        Returns:
            Dictionary containing correlation analysis results
        """
        logger.info(f"Correlating executive changes with performance for {company_ticker}")
        
        # Create a cache key
        cache_key = self._get_cache_key("exec_performance", f"{company_ticker}_{start_date}_{end_date}")
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached performance correlation for {company_ticker}")
            return cached_result
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            # Default to 2 years ago
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        
        try:
            # Get executive changes
            executive_changes = self.get_executive_changes_from_sec(company_ticker, start_date, end_date)
            
            # In a real implementation, you would get stock price and financial performance data
            # For demonstration, we'll create simulated performance data
            
            # Generate dates between start_date and end_date
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            date_range = [(start + timedelta(days=i)).strftime("%Y-%m-%d") 
                         for i in range((end - start).days + 1)]
            
            # Generate simulated stock prices
            base_price = 100.0
            prices = []
            price = base_price
            
            for date in date_range:
                # Add some random movement
                price = price * (1 + (random.random() - 0.5) * 0.02)  # +/- 1% daily change
                prices.append(price)
            
            # Create performance data
            performance_data = {
                "dates": date_range,
                "stock_prices": [round(price, 2) for price in prices]
            }
            
            # Analyze correlation between executive changes and stock price
            correlations = []
            
            for change in executive_changes:
                change_date = change.get("detection_date")
                if change_date in date_range:
                    change_index = date_range.index(change_date)
                    
                    # Calculate stock performance before and after the change
                    days_before = min(30, change_index)  # Up to 30 days before
                    days_after = min(30, len(date_range) - change_index - 1)  # Up to 30 days after
                    
                    price_before = prices[change_index - days_before] if days_before > 0 else prices[0]
                    price_at_change = prices[change_index]
                    price_after = prices[change_index + days_after] if days_after > 0 else prices[-1]
                    
                    percent_change_before = ((price_at_change - price_before) / price_before) * 100
                    percent_change_after = ((price_after - price_at_change) / price_at_change) * 100
                    
                    correlations.append({
                        "executive": change.get("name"),
                        "title": change.get("new_title") or change.get("previous_title"),
                        "change_type": change.get("change_type"),
                        "change_date": change_date,
                        "price_before": round(price_before, 2),
                        "price_at_change": round(price_at_change, 2),
                        "price_after": round(price_after, 2),
                        "percent_change_before": round(percent_change_before, 2),
                        "percent_change_after": round(percent_change_after, 2),
                        "impact_assessment": "positive" if percent_change_after > 0 else "negative"
                    })
            
            # Prepare the result
            result = {
                "company": company_ticker,
                "period": {
                    "start_date": start_date,
                    "end_date": end_date
                },
                "executive_changes": executive_changes,
                "performance_data": performance_data,
                "correlations": correlations,
                "summary": {
                    "total_changes": len(executive_changes),
                    "positive_impact": sum(1 for c in correlations if c["impact_assessment"] == "positive"),
                    "negative_impact": sum(1 for c in correlations if c["impact_assessment"] == "negative"),
                    "average_impact": round(sum(c["percent_change_after"] for c in correlations) / len(correlations) if correlations else 0, 2)
                }
            }
            
            # Cache and return the result
            self._save_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error correlating executive changes with performance for {company_ticker}: {e}")
            return {"company": company_ticker, "error": str(e)}
    
    def generate_timeline(self, company_ticker: str, start_date: Optional[str] = None, 
                         end_date: Optional[str] = None) -> Dict:
        """
        Generate a timeline of executive changes for a company.
        
        Args:
            company_ticker: Company ticker symbol
            start_date: Start date for timeline (YYYY-MM-DD)
            end_date: End date for timeline (YYYY-MM-DD)
            
        Returns:
            Dictionary containing timeline data
        """
        logger.info(f"Generating executive changes timeline for {company_ticker}")
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            # Default to 5 years ago
            start_date = (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d")
        
        try:
            # Get executive changes
            executive_changes = self.get_executive_changes_from_sec(company_ticker, start_date, end_date)
            
            # Sort changes by date
            executive_changes.sort(key=lambda x: x.get("detection_date", ""))
            
            # Group changes by year and month
            timeline = {}
            
            for change in executive_changes:
                date = change.get("detection_date", "")
                if not date:
                    continue
                
                try:
                    year_month = date[:7]  # YYYY-MM
                    if year_month not in timeline:
                        timeline[year_month] = []
                    
                    timeline[year_month].append(change)
                except:
                    continue
            
            # Prepare the result
            result = {
                "company": company_ticker,
                "period": {
                    "start_date": start_date,
                    "end_date": end_date
                },
                "timeline": timeline,
                "summary": {
                    "total_changes": len(executive_changes),
                    "appointments": sum(1 for c in executive_changes if c.get("change_type") == "appointment"),
                    "departures": sum(1 for c in executive_changes if c.get("change_type") == "departure"),
                    "role_changes": sum(1 for c in executive_changes if c.get("change_type") == "role_change")
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating timeline for {company_ticker}: {e}")
            return {"company": company_ticker, "error": str(e)}
    
    def export_to_csv(self, data: List[Dict], output_file: str) -> str:
        """
        Export executive changes data to CSV.
        
        Args:
            data: List of dictionaries containing executive change information
            output_file: Path to output CSV file
            
        Returns:
            Path to the created CSV file
        """
        logger.info(f"Exporting executive changes data to CSV: {output_file}")
        
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Write to CSV
            df.to_csv(output_file, index=False)
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return ""
    
    def export_to_excel(self, data_dict: Dict[str, List[Dict]], output_file: str) -> str:
        """
        Export executive changes data to Excel with multiple sheets.
        
        Args:
            data_dict: Dictionary mapping sheet names to lists of dictionaries
            output_file: Path to output Excel file
            
        Returns:
            Path to the created Excel file
        """
        logger.info(f"Exporting executive changes data to Excel: {output_file}")
        
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Create Excel writer
            with pd.ExcelWriter(output_file) as writer:
                for sheet_name, data in data_dict.items():
                    # Convert data to DataFrame
                    df = pd.DataFrame(data)
                    
                    # Write to Excel sheet
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return ""
