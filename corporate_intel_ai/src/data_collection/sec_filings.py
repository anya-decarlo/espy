"""
SEC Filings Collection Module

This module handles the collection of SEC filings data from EDGAR database.
It provides functionality to search, download, and parse various SEC forms
including 10-K, 10-Q, 8-K, S-1, 13F, and other relevant filings.
"""

import os
import requests
import logging
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Set, Tuple, Any
import pandas as pd
from bs4 import BeautifulSoup
import concurrent.futures
from pathlib import Path
import pickle

# Configure logging
logger = logging.getLogger(__name__)

class SECFilingsCollector:
    """
    Class for collecting and processing SEC filings from EDGAR database.
    """
    
    # Common SEC form types
    FORM_TYPES = {
        "10K": "Annual report",
        "10Q": "Quarterly report",
        "8K": "Current report",
        "S1": "Registration statement for securities",
        "13F": "Institutional investment manager holdings",
        "DEF14A": "Definitive proxy statement",
        "4": "Statement of changes in beneficial ownership",
        "13D": "Beneficial ownership report",
        "13G": "Beneficial ownership report",
        "424B": "Prospectus",
        "6K": "Foreign issuer current report",
        "20F": "Foreign issuer annual report"
    }
    
    def __init__(self, user_agent: str, output_dir: str = "data/sec_filings", rate_limit_sleep: int = 0.1, 
                 use_cache: bool = True, cache_expiry_days: int = 30):
        """
        Initialize the SEC filings collector.
        
        Args:
            user_agent: Email address to identify with SEC EDGAR
            output_dir: Directory to store downloaded filings
            rate_limit_sleep: Sleep time between requests to respect SEC rate limits (seconds)
            use_cache: Whether to use caching for downloaded filings
            cache_expiry_days: Number of days after which cache entries expire
        """
        self.base_url = "https://www.sec.gov/Archives/edgar/data"
        self.edgar_search_url = "https://www.sec.gov/cgi-bin/browse-edgar"
        self.submissions_url = "https://data.sec.gov/submissions"
        self.user_agent = user_agent
        self.output_dir = output_dir
        self.rate_limit_sleep = rate_limit_sleep
        self.use_cache = use_cache
        self.cache_expiry_days = cache_expiry_days
        self.cache_dir = os.path.join(output_dir, "_cache")
        
        # Create output and cache directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Headers for SEC EDGAR requests
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov"
        }
    
    def _get_cache_key(self, prefix: str, identifier: str) -> str:
        """
        Generate a cache key for a given prefix and identifier.
        
        Args:
            prefix: Type of data being cached
            identifier: Unique identifier for the data
            
        Returns:
            Cache key string
        """
        return f"{prefix}_{identifier.replace('/', '_').replace('-', '_')}"
    
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
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> requests.Response:
        """
        Make a request to SEC EDGAR with rate limiting.
        
        Args:
            url: URL to request
            params: Optional query parameters
            
        Returns:
            Response object
        """
        # Sleep to respect SEC rate limits
        time.sleep(self.rate_limit_sleep)
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Error making request to {url}: {e}")
            raise
    
    def get_company_cik(self, ticker_or_name: str) -> Optional[str]:
        """
        Get CIK number for a company by ticker symbol or name.
        
        Args:
            ticker_or_name: Company ticker symbol or name
            
        Returns:
            CIK number as string, or None if not found
        """
        logger.info(f"Looking up CIK for {ticker_or_name}")
        
        # Try to get CIK from ticker
        try:
            # Check if input is already a CIK
            if ticker_or_name.isdigit() and len(ticker_or_name) >= 10:
                return ticker_or_name.zfill(10)
            
            # Construct URL for CIK lookup
            url = f"{self.submissions_url}/CIK{ticker_or_name}.json"
            response = self._make_request(url)
            data = response.json()
            
            # Extract CIK
            cik = data.get("cik")
            if cik:
                # Format CIK with leading zeros
                return str(cik).zfill(10)
            
        except Exception as e:
            logger.warning(f"Could not find CIK for {ticker_or_name} using direct lookup: {e}")
        
        # Fall back to search
        try:
            params = {
                "company": ticker_or_name,
                "owner": "exclude",
                "action": "getcompany"
            }
            response = self._make_request(self.edgar_search_url, params)
            
            # Parse HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for CIK in the response
            cik_match = re.search(r'CIK=(\d+)', response.text)
            if cik_match:
                return cik_match.group(1).zfill(10)
            
        except Exception as e:
            logger.error(f"Error searching for CIK for {ticker_or_name}: {e}")
        
        return None
    
    def search_filings(self, 
                      ticker_or_cik: str, 
                      form_types: Union[str, List[str]] = "10-K", 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None,
                      count: int = 100) -> List[Dict]:
        """
        Search for SEC filings based on ticker symbol or CIK and form types.
        
        Args:
            ticker_or_cik: Company ticker symbol or CIK number
            form_types: SEC form type(s) (10-K, 10-Q, 8-K, etc.) as string or list
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            count: Maximum number of filings to return
            
        Returns:
            List of dictionaries containing filing metadata
        """
        # Check cache first
        cache_key = self._get_cache_key("search", f"{ticker_or_cik}_{str(form_types)}_{start_date}_{end_date}_{count}")
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached search results for {ticker_or_cik}")
            return cached_result
        
        # Convert form_types to list if it's a string
        if isinstance(form_types, str):
            form_types = [form_types]
        
        # Normalize form types (remove dashes)
        form_types = [ft.replace('-', '') for ft in form_types]
        
        # Get CIK if ticker is provided
        if not ticker_or_cik.isdigit():
            cik = self.get_company_cik(ticker_or_cik)
            if not cik:
                logger.error(f"Could not find CIK for {ticker_or_cik}")
                return []
        else:
            cik = ticker_or_cik.zfill(10)
        
        logger.info(f"Searching for {form_types} filings for CIK {cik}")
        
        # Construct URL for submissions API
        url = f"{self.submissions_url}/CIK{cik}.json"
        
        try:
            # Get company submissions
            response = self._make_request(url)
            data = response.json()
            
            # Extract filings
            filings = []
            recent_filings = data.get("filings", {}).get("recent", {})
            
            if not recent_filings:
                logger.warning(f"No recent filings found for CIK {cik}")
                return []
            
            # Get filing data
            form_types_list = recent_filings.get("form", [])
            filing_dates = recent_filings.get("filingDate", [])
            accession_numbers = recent_filings.get("accessionNumber", [])
            filing_urls = recent_filings.get("primaryDocument", [])
            filing_descriptions = recent_filings.get("primaryDocDescription", [])
            
            # Process filings
            for i in range(min(len(form_types_list), count)):
                form = form_types_list[i].replace('-', '')
                
                # Check if form type matches any requested form types
                if any(form.startswith(ft) for ft in form_types) or not form_types:
                    # Check date range if provided
                    filing_date = filing_dates[i]
                    
                    if start_date and filing_date < start_date:
                        continue
                    
                    if end_date and filing_date > end_date:
                        continue
                    
                    # Add filing to results
                    filings.append({
                        "cik": cik,
                        "company_name": data.get("name", ""),
                        "form_type": form_types_list[i],
                        "filing_date": filing_date,
                        "accession_number": accession_numbers[i],
                        "primary_document": filing_urls[i],
                        "description": filing_descriptions[i] if i < len(filing_descriptions) else ""
                    })
            
            # Cache the results
            self._save_to_cache(cache_key, filings)
            
            return filings
            
        except Exception as e:
            logger.error(f"Error searching for filings: {e}")
            return []
    
    def download_filing(self, accession_number: str, cik: str, save_path: Optional[str] = None) -> str:
        """
        Download a specific SEC filing by accession number and CIK.
        
        Args:
            accession_number: SEC filing accession number
            cik: Company CIK number
            save_path: Optional path to save the filing
            
        Returns:
            Path to the downloaded filing
        """
        # Determine save path
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"{cik}_{accession_number}.txt")
        
        # Check if file already exists
        if os.path.exists(save_path):
            logger.info(f"Filing already exists at {save_path}")
            return save_path
        
        # Format accession number for URL
        formatted_accession = accession_number.replace('-', '')
        
        # Construct URL for the filing
        url = f"{self.base_url}/{cik}/{formatted_accession}/{accession_number}.txt"
        
        # Download the filing
        try:
            response = self._make_request(url)
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Successfully downloaded filing to {save_path}")
            return save_path
        
        except Exception as e:
            logger.error(f"Error downloading filing: {e}")
            raise
    
    def download_filing_documents(self, 
                                 accession_number: str, 
                                 cik: str, 
                                 output_dir: Optional[str] = None) -> List[str]:
        """
        Download all documents associated with a filing.
        
        Args:
            accession_number: SEC filing accession number
            cik: Company CIK number
            output_dir: Optional directory to save the documents
            
        Returns:
            List of paths to the downloaded documents
        """
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, f"{cik}_{accession_number}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct URL for the filing index
        index_url = f"{self.base_url}/{cik}/{accession_number.replace('-', '')}/{accession_number}-index.html"
        
        try:
            # Get the filing index
            response = self._make_request(index_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all document links
            document_links = []
            for table in soup.find_all('table'):
                for row in table.find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        document_type = cells[0].text.strip()
                        document_desc = cells[1].text.strip()
                        document_link = cells[2].find('a')
                        
                        if document_link and document_link.has_attr('href'):
                            href = document_link['href']
                            if href.startswith('/'):
                                document_links.append({
                                    'type': document_type,
                                    'description': document_desc,
                                    'url': f"https://www.sec.gov{href}"
                                })
            
            # Download each document
            downloaded_paths = []
            for doc in document_links:
                try:
                    # Extract filename from URL
                    filename = os.path.basename(doc['url'])
                    save_path = os.path.join(output_dir, filename)
                    
                    # Download the document
                    doc_response = self._make_request(doc['url'])
                    
                    with open(save_path, 'wb') as f:
                        f.write(doc_response.content)
                    
                    downloaded_paths.append(save_path)
                    logger.info(f"Downloaded document: {filename}")
                    
                except Exception as e:
                    logger.error(f"Error downloading document {doc['url']}: {e}")
            
            return downloaded_paths
            
        except Exception as e:
            logger.error(f"Error downloading filing documents: {e}")
            return []
    
    def parse_filing(self, filing_path: str, form_type: Optional[str] = None) -> Dict:
        """
        Parse a downloaded SEC filing to extract structured data.
        
        Args:
            filing_path: Path to the downloaded filing
            form_type: SEC form type to determine parsing strategy
            
        Returns:
            Dictionary containing parsed data from the filing
        """
        # Check cache first
        cache_key = self._get_cache_key("parsed", f"{os.path.basename(filing_path)}")
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached parsed data for {filing_path}")
            return cached_result
        
        logger.info(f"Parsing filing: {filing_path}")
        
        try:
            # Read the filing content
            with open(filing_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Determine form type if not provided
            if not form_type:
                form_type_match = re.search(r'CONFORMED SUBMISSION TYPE:\s*(\w+)', content)
                if form_type_match:
                    form_type = form_type_match.group(1)
                else:
                    logger.warning(f"Could not determine form type for {filing_path}")
                    form_type = "UNKNOWN"
            
            # Extract basic filing information
            filing_info = {
                "form_type": form_type,
                "filing_path": filing_path
            }
            
            # Extract company information
            company_name_match = re.search(r'COMPANY CONFORMED NAME:\s*(.+?)$', content, re.MULTILINE)
            if company_name_match:
                filing_info["company_name"] = company_name_match.group(1).strip()
            
            cik_match = re.search(r'CENTRAL INDEX KEY:\s*(\d+)', content)
            if cik_match:
                filing_info["cik"] = cik_match.group(1)
            
            # Extract filing date
            filing_date_match = re.search(r'FILED AS OF DATE:\s*(\d+)', content)
            if filing_date_match:
                date_str = filing_date_match.group(1)
                filing_info["filing_date"] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            
            # Parse based on form type
            if form_type.startswith("10-K") or form_type == "10K":
                filing_info.update(self._parse_10k(content))
            elif form_type.startswith("10-Q") or form_type == "10Q":
                filing_info.update(self._parse_10q(content))
            elif form_type.startswith("8-K") or form_type == "8K":
                filing_info.update(self._parse_8k(content))
            elif form_type.startswith("13F"):
                filing_info.update(self._parse_13f(content, filing_path))
            elif form_type.startswith("DEF 14A") or form_type == "DEF14A":
                filing_info.update(self._parse_proxy(content))
            elif form_type.startswith("S-1") or form_type == "S1":
                filing_info.update(self._parse_s1(content))
            
            # Cache the parsed results
            self._save_to_cache(cache_key, filing_info)
            
            return filing_info
            
        except Exception as e:
            logger.error(f"Error parsing filing {filing_path}: {e}")
            return {"status": "error", "error": str(e), "filing_path": filing_path}
    
    def _parse_13f(self, content: str, filing_path: str) -> Dict:
        """
        Parse 13F institutional investment holdings.
        
        Args:
            content: Filing content
            filing_path: Path to the filing file (needed for XML extraction)
            
        Returns:
            Dictionary containing parsed data
        """
        parsed_data = {
            "type": "institutional_holdings",
            "holdings": [],
            "report_period": None,
            "manager_name": None,
            "manager_address": None,
            "total_value": None
        }
        
        try:
            # Extract the XML file from the filing directory
            filing_dir = os.path.dirname(filing_path)
            xml_files = [f for f in os.listdir(filing_dir) if f.endswith('.xml')]
            
            if not xml_files:
                # Try to find XML content within the text filing
                xml_start = content.find('<?xml')
                if xml_start != -1:
                    xml_end = content.find('</XML>', xml_start)
                    if xml_end != -1:
                        xml_content = content[xml_start:xml_end + 6]
                        
                        # Parse the XML content
                        root = ET.fromstring(xml_content)
                        return self._parse_13f_xml(root, parsed_data)
                
                logger.warning(f"No XML files found for 13F filing: {filing_path}")
                return parsed_data
            
            # Process the first XML file (usually the primary document)
            xml_path = os.path.join(filing_dir, xml_files[0])
            
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                return self._parse_13f_xml(root, parsed_data)
                
            except Exception as e:
                logger.error(f"Error parsing 13F XML: {e}")
                return parsed_data
                
        except Exception as e:
            logger.error(f"Error processing 13F filing: {e}")
            return parsed_data
    
    def _parse_13f_xml(self, root: ET.Element, parsed_data: Dict) -> Dict:
        """
        Parse 13F XML content.
        
        Args:
            root: XML root element
            parsed_data: Dictionary to update with parsed data
            
        Returns:
            Updated dictionary with parsed data
        """
        # Define XML namespaces
        ns = {
            'ns1': 'http://www.sec.gov/edgar/document/thirteenf/informationtable',
            'ns2': 'http://www.sec.gov/edgar/thirteenffiler'
        }
        
        # Extract report period
        period_element = root.find('.//ns2:reportCalendarOrQuarter', ns)
        if period_element is not None:
            parsed_data["report_period"] = period_element.text
        
        # Extract manager information
        manager_name = root.find('.//ns2:name', ns)
        if manager_name is not None:
            parsed_data["manager_name"] = manager_name.text
        
        # Extract address
        address_elements = root.findall('.//ns2:street1', ns) + root.findall('.//ns2:street2', ns) + \
                          root.findall('.//ns2:city', ns) + root.findall('.//ns2:stateOrCountry', ns) + \
                          root.findall('.//ns2:zipCode', ns)
        if address_elements:
            parsed_data["manager_address"] = ' '.join([e.text for e in address_elements if e.text])
        
        # Extract holdings
        info_tables = root.findall('.//ns1:infoTable', ns)
        
        for table in info_tables:
            try:
                name_element = table.find('.//ns1:nameOfIssuer', ns)
                title_element = table.find('.//ns1:titleOfClass', ns)
                cusip_element = table.find('.//ns1:cusip', ns)
                value_element = table.find('.//ns1:value', ns)
                shares_element = table.find('.//ns1:sshPrnamt', ns)
                
                if name_element is not None and cusip_element is not None:
                    holding = {
                        "name": name_element.text,
                        "title": title_element.text if title_element is not None else None,
                        "cusip": cusip_element.text,
                        "value": int(value_element.text) * 1000 if value_element is not None else None,  # Values are in thousands
                        "shares": int(shares_element.text) if shares_element is not None else None
                    }
                    parsed_data["holdings"].append(holding)
            except Exception as e:
                logger.warning(f"Error parsing holding: {e}")
        
        # Calculate total value
        if parsed_data["holdings"]:
            total_value = sum(h["value"] for h in parsed_data["holdings"] if h["value"] is not None)
            parsed_data["total_value"] = total_value
        
        return parsed_data
    
    def _parse_s1(self, content: str) -> Dict:
        """
        Parse S-1 registration statement.
        
        Args:
            content: Filing content
            
        Returns:
            Dictionary containing parsed data
        """
        parsed_data = {
            "type": "registration_statement",
            "sections": {},
            "offering_details": {}
        }
        
        # Extract key sections
        sections = {
            "prospectus_summary": self._extract_section(content, "PROSPECTUS SUMMARY", "RISK FACTORS"),
            "risk_factors": self._extract_section(content, "RISK FACTORS", "USE OF PROCEEDS"),
            "use_of_proceeds": self._extract_section(content, "USE OF PROCEEDS", "DIVIDEND POLICY"),
            "business": self._extract_section(content, "BUSINESS", "MANAGEMENT"),
            "management": self._extract_section(content, "MANAGEMENT", "EXECUTIVE COMPENSATION")
        }
        
        parsed_data["sections"] = sections
        
        # Extract offering details
        try:
            # Find offering size
            offering_match = re.search(r'aggregate offering price[^\$]*\$([0-9,]+)', content, re.IGNORECASE)
            if offering_match:
                parsed_data["offering_details"]["size"] = offering_match.group(1).replace(',', '')
            
            # Find proposed ticker symbol
            ticker_match = re.search(r'proposed trading symbol[^A-Z]*([\w.]+)', content, re.IGNORECASE)
            if ticker_match:
                parsed_data["offering_details"]["ticker"] = ticker_match.group(1)
            
            # Find underwriters
            underwriter_section = self._extract_section(content, "UNDERWRITING", "LEGAL MATTERS")
            if underwriter_section:
                underwriters = re.findall(r'([A-Z][A-Za-z\s]+(?:LLC|Inc\.|LP|Securities))', underwriter_section)
                if underwriters:
                    parsed_data["offering_details"]["underwriters"] = underwriters
        except Exception as e:
            logger.warning(f"Error extracting offering details: {e}")
        
        return parsed_data
    
    def _extract_section(self, content: str, start_marker: str, end_marker: str) -> str:
        """
        Extract a section from filing content.
        
        Args:
            content: Filing content
            start_marker: Text marking the start of the section
            end_marker: Text marking the end of the section
            
        Returns:
            Extracted section text
        """
        try:
            start_idx = content.find(start_marker)
            if start_idx == -1:
                return ""
            
            # Start after the marker
            start_idx += len(start_marker)
            
            # Find end marker after start position
            end_idx = content.find(end_marker, start_idx)
            if end_idx == -1:
                # If end marker not found, take the rest of the content
                return content[start_idx:].strip()
            
            return content[start_idx:end_idx].strip()
            
        except Exception as e:
            logger.error(f"Error extracting section {start_marker}: {e}")
            return ""
    
    def _extract_financial_highlights(self, content: str) -> Dict:
        """
        Extract financial highlights from filing content.
        
        Args:
            content: Filing content
            
        Returns:
            Dictionary containing financial highlights
        """
        financial_data = {
            "revenue": None,
            "net_income": None,
            "eps": None,
            "assets": None,
            "liabilities": None,
            "equity": None,
            "cash": None,
            "operating_income": None
        }
        
        # Extract revenue
        revenue_patterns = [
            r'Total (?:net )?revenues?[^\n]*\$([0-9,]+)',
            r'Net sales[^\n]*\$([0-9,]+)',
            r'Revenue[^\n]*\$([0-9,]+)'
        ]
        
        for pattern in revenue_patterns:
            revenue_match = re.search(pattern, content, re.IGNORECASE)
            if revenue_match:
                financial_data["revenue"] = revenue_match.group(1).replace(',', '')
                break
        
        # Extract net income
        income_patterns = [
            r'Net income[^\n]*\$([0-9,]+)',
            r'Net earnings[^\n]*\$([0-9,]+)',
            r'Net profit[^\n]*\$([0-9,]+)'
        ]
        
        for pattern in income_patterns:
            income_match = re.search(pattern, content, re.IGNORECASE)
            if income_match:
                financial_data["net_income"] = income_match.group(1).replace(',', '')
                break
        
        # Extract EPS
        eps_match = re.search(r'Earnings per share[^\n]*\$([0-9.]+)', content, re.IGNORECASE)
        if eps_match:
            financial_data["eps"] = eps_match.group(1)
        
        # Extract assets
        assets_match = re.search(r'Total assets[^\n]*\$([0-9,]+)', content, re.IGNORECASE)
        if assets_match:
            financial_data["assets"] = assets_match.group(1).replace(',', '')
        
        # Extract liabilities
        liabilities_match = re.search(r'Total liabilities[^\n]*\$([0-9,]+)', content, re.IGNORECASE)
        if liabilities_match:
            financial_data["liabilities"] = liabilities_match.group(1).replace(',', '')
        
        # Extract equity
        equity_patterns = [
            r'Total (?:stockholders\'|shareholders\') equity[^\n]*\$([0-9,]+)',
            r'(?:Stockholders\'|Shareholders\') equity[^\n]*\$([0-9,]+)'
        ]
        
        for pattern in equity_patterns:
            equity_match = re.search(pattern, content, re.IGNORECASE)
            if equity_match:
                financial_data["equity"] = equity_match.group(1).replace(',', '')
                break
        
        # Extract cash
        cash_match = re.search(r'Cash and cash equivalents[^\n]*\$([0-9,]+)', content, re.IGNORECASE)
        if cash_match:
            financial_data["cash"] = cash_match.group(1).replace(',', '')
        
        # Extract operating income
        op_income_match = re.search(r'Operating income[^\n]*\$([0-9,]+)', content, re.IGNORECASE)
        if op_income_match:
            financial_data["operating_income"] = op_income_match.group(1).replace(',', '')
        
        return financial_data
    
    def get_company_filings_by_type(self, 
                                   ticker_or_cik: str, 
                                   form_types: List[str] = ["10-K", "10-Q", "8-K"],
                                   years: int = 3) -> Dict[str, List[Dict]]:
        """
        Get company filings organized by form type for a specified time period.
        
        Args:
            ticker_or_cik: Company ticker symbol or CIK number
            form_types: List of SEC form types to retrieve
            years: Number of years of filings to retrieve
            
        Returns:
            Dictionary mapping form types to lists of filing metadata
        """
        # Calculate start date
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")
        
        # Get all filings
        all_filings = self.search_filings(
            ticker_or_cik=ticker_or_cik,
            form_types=form_types,
            start_date=start_date,
            end_date=end_date,
            count=500  # Increased count to get more filings
        )
        
        # Organize by form type
        filings_by_type = {}
        for filing in all_filings:
            form_type = filing["form_type"]
            if form_type not in filings_by_type:
                filings_by_type[form_type] = []
            
            filings_by_type[form_type].append(filing)
        
        return filings_by_type
    
    def download_and_parse_latest_filings(self, 
                                         ticker_or_cik: str, 
                                         form_types: List[str] = ["10-K", "10-Q"]) -> Dict[str, Dict]:
        """
        Download and parse the latest filings of specified types.
        
        Args:
            ticker_or_cik: Company ticker symbol or CIK number
            form_types: List of SEC form types to retrieve
            
        Returns:
            Dictionary mapping form types to parsed filing data
        """
        # Get filings
        filings = self.search_filings(
            ticker_or_cik=ticker_or_cik,
            form_types=form_types,
            count=10  # Limit to recent filings
        )
        
        # Group by form type
        latest_by_type = {}
        for filing in filings:
            form_type = filing["form_type"]
            if form_type not in latest_by_type:
                latest_by_type[form_type] = filing
        
        # Download and parse latest filing of each type
        results = {}
        for form_type, filing in latest_by_type.items():
            try:
                # Download filing
                filing_path = self.download_filing(
                    accession_number=filing["accession_number"],
                    cik=filing["cik"]
                )
                
                # Parse filing
                parsed_data = self.parse_filing(filing_path, form_type)
                
                # Add to results
                results[form_type] = {
                    "metadata": filing,
                    "parsed_data": parsed_data
                }
                
            except Exception as e:
                logger.error(f"Error processing {form_type} filing: {e}")
        
        return results
    
    def batch_process_companies(self, 
                               tickers_or_ciks: List[str], 
                               form_types: List[str] = ["10-K", "10-Q"], 
                               max_workers: int = 5) -> Dict[str, Dict]:
        """
        Process multiple companies in parallel to download and parse their filings.
        
        Args:
            tickers_or_ciks: List of company ticker symbols or CIK numbers
            form_types: List of SEC form types to retrieve
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping company identifiers to their filing data
        """
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for each company
            future_to_company = {
                executor.submit(self.download_and_parse_latest_filings, ticker_or_cik, form_types): ticker_or_cik
                for ticker_or_cik in tickers_or_ciks
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_company):
                company = future_to_company[future]
                try:
                    data = future.result()
                    results[company] = data
                    logger.info(f"Completed processing for {company}")
                except Exception as e:
                    logger.error(f"Error processing {company}: {e}")
                    results[company] = {"status": "error", "error": str(e)}
        
        return results
    
    def compare_companies(self, 
                         tickers_or_ciks: List[str], 
                         form_type: str = "10-K",
                         metrics: List[str] = ["revenue", "net_income", "eps"]) -> pd.DataFrame:
        """
        Compare financial metrics across multiple companies.
        
        Args:
            tickers_or_ciks: List of company ticker symbols or CIK numbers
            form_type: SEC form type to use for comparison
            metrics: List of financial metrics to compare
            
        Returns:
            DataFrame with companies as rows and metrics as columns
        """
        # Process all companies
        all_data = self.batch_process_companies(tickers_or_ciks, [form_type])
        
        # Extract metrics for comparison
        comparison_data = []
        
        for company, data in all_data.items():
            if "status" in data and data["status"] == "error":
                continue
                
            company_data = {"company": company}
            
            # Get the filing data
            filing_data = data.get(form_type, {}).get("parsed_data", {})
            
            # Extract financial highlights
            financial_highlights = filing_data.get("financial_highlights", {})
            
            # Add metrics to comparison data
            for metric in metrics:
                company_data[metric] = financial_highlights.get(metric)
            
            comparison_data.append(company_data)
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Set company as index
        if not df.empty and "company" in df.columns:
            df.set_index("company", inplace=True)
        
        return df
