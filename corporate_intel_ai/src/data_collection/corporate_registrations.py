"""
Corporate Registrations Collection Module

This module handles the collection of corporate registration data from state business databases
and OpenCorporates. It provides functionality to search, download, and analyze corporate
registration information for competitive intelligence analysis.

NOTE: The approach implemented in this module may need to be modified based on:
1. API access limitations for certain state databases
2. Changes in data availability from OpenCorporates
3. Rate limiting and authentication requirements
4. Potential need for web scraping as a fallback for sources without APIs
5. Varying data formats across different jurisdictions

Alternative approaches to consider:
- Using paid API services that aggregate corporate registration data
- Implementing web scraping for specific high-value state databases
- Leveraging SEC data to supplement state registration information
"""

import os
import requests
import logging
import json
import re
import time
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Set
import pandas as pd
from bs4 import BeautifulSoup

# Configure logging
logger = logging.getLogger(__name__)

class CorporateRegistrationsCollector:
    """
    Class for collecting and processing corporate registration data.
    """
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None, output_dir: str = "data/corporate_registrations",
                 use_cache: bool = True, cache_expiry_days: int = 30):
        """
        Initialize the corporate registrations collector.
        
        Args:
            api_keys: Dictionary of API keys for different data sources (opencorporates, etc.)
            output_dir: Directory to store downloaded data
            use_cache: Whether to use caching for downloaded data
            cache_expiry_days: Number of days after which cache entries expire
        """
        self.api_keys = api_keys or {}
        self.output_dir = output_dir
        self.use_cache = use_cache
        self.cache_expiry_days = cache_expiry_days
        self.cache_dir = os.path.join(output_dir, "_cache")
        
        # API endpoints
        self.opencorporates_url = "https://api.opencorporates.com/v0.4"
        
        # State business database URLs
        # Note: These URLs may change and many states don't have public APIs
        self.state_db_urls = {
            "CA": "https://bizfileonline.sos.ca.gov/api/records",
            "DE": "https://icis.corp.delaware.gov/eCorp/api",
            "NY": "https://apps.dos.ny.gov/publicInquiry/api",
            "TX": "https://mycpa.cpa.state.tx.us/coa/api",
            "FL": "https://dos.myflorida.com/sunbiz/api",
            # Add more states as needed
        }
        
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
    
    def search_companies(self, 
                        company_name: Optional[str] = None,
                        registration_number: Optional[str] = None,
                        jurisdiction: Optional[str] = None,
                        officer_name: Optional[str] = None,
                        status: Optional[str] = None,
                        limit: int = 50) -> List[Dict]:
        """
        Search for companies based on various criteria.
        
        Args:
            company_name: Name of the company
            registration_number: Company registration number
            jurisdiction: Jurisdiction code (e.g., 'us_de' for Delaware, 'gb' for UK)
            officer_name: Name of an officer or director
            status: Company status (e.g., 'active', 'dissolved')
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing company data
        """
        logger.info(f"Searching for companies with criteria: name={company_name}, jurisdiction={jurisdiction}")
        
        # Create a cache key based on search parameters
        cache_params = f"{company_name}_{registration_number}_{jurisdiction}_{officer_name}_{status}_{limit}"
        cache_key = self._get_cache_key("company_search", cache_params)
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached company search results")
            return cached_result
        
        # Try multiple sources for company data
        results = []
        
        # Try OpenCorporates first
        try:
            opencorporates_results = self._search_opencorporates(
                company_name=company_name,
                registration_number=registration_number,
                jurisdiction=jurisdiction,
                officer_name=officer_name,
                status=status,
                limit=limit
            )
            if opencorporates_results:
                results.extend(opencorporates_results)
        except Exception as e:
            logger.warning(f"OpenCorporates search failed: {e}")
        
        # Try state business databases if jurisdiction is specified as a US state
        if jurisdiction and jurisdiction.startswith("us_"):
            state_code = jurisdiction.split("_")[1].upper()
            if state_code in self.state_db_urls:
                try:
                    state_results = self._search_state_database(
                        state_code=state_code,
                        company_name=company_name,
                        registration_number=registration_number,
                        officer_name=officer_name,
                        status=status,
                        limit=limit
                    )
                    if state_results:
                        results.extend(state_results)
                except Exception as e:
                    logger.warning(f"State database search failed for {state_code}: {e}")
        
        # Remove duplicates based on unique identifiers
        unique_results = {}
        for result in results:
            # Create a unique identifier based on company name and jurisdiction
            company = result.get("company_name", "")
            jurisdiction = result.get("jurisdiction_code", "")
            reg_number = result.get("company_number", "")
            unique_id = f"{company}_{jurisdiction}_{reg_number}"
            
            if unique_id not in unique_results:
                unique_results[unique_id] = result
        
        results = list(unique_results.values())
        
        # Limit results
        results = results[:limit]
        
        # Cache and return results
        self._save_to_cache(cache_key, results)
        return results
    
    def _search_opencorporates(self,
                              company_name: Optional[str] = None,
                              registration_number: Optional[str] = None,
                              jurisdiction: Optional[str] = None,
                              officer_name: Optional[str] = None,
                              status: Optional[str] = None,
                              limit: int = 50) -> List[Dict]:
        """
        Search for companies using OpenCorporates API.
        
        Args:
            company_name: Name of the company
            registration_number: Company registration number
            jurisdiction: Jurisdiction code (e.g., 'us_de' for Delaware, 'gb' for UK)
            officer_name: Name of an officer or director
            status: Company status (e.g., 'active', 'dissolved')
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing company data
        """
        if "opencorporates" not in self.api_keys:
            logger.warning("OpenCorporates API key not provided")
            return []
        
        api_key = self.api_keys["opencorporates"]
        
        # Construct query parameters
        params = {
            "api_token": api_key,
            "per_page": min(limit, 100)  # OpenCorporates limits to 100 per page
        }
        
        # Add search criteria
        if company_name:
            params["q"] = company_name
        
        if jurisdiction:
            params["jurisdiction_code"] = jurisdiction
        
        if status:
            params["status"] = status
        
        # If registration number is provided, we can do a direct lookup
        if registration_number and jurisdiction:
            try:
                url = f"{self.opencorporates_url}/companies/{jurisdiction}/{registration_number}"
                response = self._make_request(url, params=params)
                data = response.json()
                
                company = data.get("results", {}).get("company", {})
                if company:
                    return [self._format_opencorporates_company(company)]
                return []
            except Exception as e:
                logger.error(f"Error looking up company by registration number: {e}")
                return []
        
        # Otherwise, do a search
        try:
            url = f"{self.opencorporates_url}/companies/search"
            response = self._make_request(url, params=params)
            data = response.json()
            
            companies = data.get("results", {}).get("companies", [])
            
            results = []
            for company_data in companies:
                company = company_data.get("company", {})
                results.append(self._format_opencorporates_company(company))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching OpenCorporates: {e}")
            return []
    
    def _format_opencorporates_company(self, company: Dict) -> Dict:
        """
        Format OpenCorporates company data into a standardized format.
        
        Args:
            company: Company data from OpenCorporates API
            
        Returns:
            Formatted company data
        """
        return {
            "source": "opencorporates",
            "company_name": company.get("name"),
            "company_number": company.get("company_number"),
            "jurisdiction_code": company.get("jurisdiction_code"),
            "jurisdiction_name": company.get("jurisdiction_name"),
            "company_type": company.get("company_type"),
            "incorporation_date": company.get("incorporation_date"),
            "dissolution_date": company.get("dissolution_date"),
            "status": company.get("current_status"),
            "registry_url": company.get("registry_url"),
            "opencorporates_url": company.get("opencorporates_url"),
            "registered_address": company.get("registered_address"),
            "previous_names": company.get("previous_names", []),
            "industry_codes": company.get("industry_codes", []),
            "is_branch": company.get("branch_status") is not None
        }
    
    def _search_state_database(self,
                              state_code: str,
                              company_name: Optional[str] = None,
                              registration_number: Optional[str] = None,
                              officer_name: Optional[str] = None,
                              status: Optional[str] = None,
                              limit: int = 50) -> List[Dict]:
        """
        Search for companies using a state business database.
        
        Args:
            state_code: Two-letter state code (e.g., 'CA', 'DE')
            company_name: Name of the company
            registration_number: Company registration number
            officer_name: Name of an officer or director
            status: Company status (e.g., 'active', 'dissolved')
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing company data
        """
        if state_code not in self.state_db_urls:
            logger.warning(f"No API URL defined for state {state_code}")
            return []
        
        # State database APIs vary widely, so we need to handle each one differently
        # This is a simplified implementation that would need to be customized for each state
        
        base_url = self.state_db_urls[state_code]
        
        # Different states have different parameter names and formats
        params = {}
        
        if state_code == "CA":
            params["entityName"] = company_name if company_name else ""
            params["entityNumber"] = registration_number if registration_number else ""
            params["status"] = status if status else ""
            url = f"{base_url}/search"
        elif state_code == "DE":
            params["name"] = company_name if company_name else ""
            params["file_number"] = registration_number if registration_number else ""
            url = f"{base_url}/search"
        elif state_code == "NY":
            params["entityName"] = company_name if company_name else ""
            params["ID"] = registration_number if registration_number else ""
            url = f"{base_url}/search_corporations"
        elif state_code == "TX":
            params["name"] = company_name if company_name else ""
            params["number"] = registration_number if registration_number else ""
            url = f"{base_url}/search"
        elif state_code == "FL":
            params["name"] = company_name if company_name else ""
            params["document_number"] = registration_number if registration_number else ""
            url = f"{base_url}/search"
        else:
            logger.warning(f"Search implementation not available for state {state_code}")
            return []
        
        try:
            response = self._make_request(url, params=params)
            data = response.json()
            
            # Format varies by state, this is a simplified example
            results = []
            
            # Parse the response based on the state format
            if state_code == "CA":
                companies = data.get("results", [])
                for company in companies[:limit]:
                    results.append(self._format_state_company(company, state_code))
            elif state_code == "DE":
                companies = data.get("entities", [])
                for company in companies[:limit]:
                    results.append(self._format_state_company(company, state_code))
            # Add more state-specific parsing as needed
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching {state_code} state database: {e}")
            return []
    
    def _format_state_company(self, company: Dict, state_code: str) -> Dict:
        """
        Format state database company data into a standardized format.
        
        Args:
            company: Company data from state database
            state_code: Two-letter state code
            
        Returns:
            Formatted company data
        """
        # Format varies by state, this is a simplified example
        if state_code == "CA":
            return {
                "source": f"state_{state_code}",
                "company_name": company.get("entityName"),
                "company_number": company.get("entityNumber"),
                "jurisdiction_code": f"us_{state_code.lower()}",
                "jurisdiction_name": f"California, US",
                "company_type": company.get("entityType"),
                "incorporation_date": company.get("registrationDate"),
                "status": company.get("status"),
                "registry_url": company.get("detailsUrl")
            }
        elif state_code == "DE":
            return {
                "source": f"state_{state_code}",
                "company_name": company.get("name"),
                "company_number": company.get("fileNumber"),
                "jurisdiction_code": f"us_{state_code.lower()}",
                "jurisdiction_name": f"Delaware, US",
                "company_type": company.get("type"),
                "incorporation_date": company.get("incorporationDate"),
                "status": company.get("status"),
                "registry_url": company.get("detailsUrl")
            }
        # Add more state-specific formatting as needed
        
        # Default format if state is not specifically handled
        return {
            "source": f"state_{state_code}",
            "company_name": company.get("name", ""),
            "company_number": company.get("number", ""),
            "jurisdiction_code": f"us_{state_code.lower()}",
            "jurisdiction_name": f"{state_code}, US",
            "status": company.get("status", "")
        }
    
    def get_company_details(self, company_number: str, jurisdiction: str) -> Dict:
        """
        Get detailed information about a specific company.
        
        Args:
            company_number: Company registration number
            jurisdiction: Jurisdiction code (e.g., 'us_de' for Delaware, 'gb' for UK)
            
        Returns:
            Dictionary containing detailed company information
        """
        logger.info(f"Getting details for company {company_number} in {jurisdiction}")
        
        # Create a cache key
        cache_key = self._get_cache_key("company_details", f"{jurisdiction}_{company_number}")
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached company details")
            return cached_result
        
        # Try to get details from OpenCorporates first
        try:
            company = self._get_opencorporates_company_details(company_number, jurisdiction)
            if company:
                # Get additional data
                company["officers"] = self._get_company_officers(company_number, jurisdiction)
                company["filings"] = self._get_company_filings(company_number, jurisdiction)
                
                # Cache and return result
                self._save_to_cache(cache_key, company)
                return company
        except Exception as e:
            logger.warning(f"Error getting company details from OpenCorporates: {e}")
        
        # If OpenCorporates fails, try state database for US jurisdictions
        if jurisdiction.startswith("us_"):
            state_code = jurisdiction.split("_")[1].upper()
            try:
                company = self._get_state_company_details(company_number, state_code)
                if company:
                    # Cache and return result
                    self._save_to_cache(cache_key, company)
                    return company
            except Exception as e:
                logger.warning(f"Error getting company details from state database: {e}")
        
        # If all sources fail, return empty result
        return {}
    
    def _get_opencorporates_company_details(self, company_number: str, jurisdiction: str) -> Dict:
        """
        Get detailed company information from OpenCorporates.
        
        Args:
            company_number: Company registration number
            jurisdiction: Jurisdiction code
            
        Returns:
            Dictionary containing company details
        """
        if "opencorporates" not in self.api_keys:
            logger.warning("OpenCorporates API key not provided")
            return {}
        
        api_key = self.api_keys["opencorporates"]
        
        try:
            url = f"{self.opencorporates_url}/companies/{jurisdiction}/{company_number}"
            params = {"api_token": api_key}
            
            response = self._make_request(url, params=params)
            data = response.json()
            
            company = data.get("results", {}).get("company", {})
            if not company:
                return {}
                
            # Format the company data
            return self._format_opencorporates_company(company)
            
        except Exception as e:
            logger.error(f"Error getting company details from OpenCorporates: {e}")
            return {}
    
    def _get_state_company_details(self, company_number: str, state_code: str) -> Dict:
        """
        Get detailed company information from a state business database.
        
        Args:
            company_number: Company registration number
            state_code: Two-letter state code
            
        Returns:
            Dictionary containing company details
        """
        if state_code not in self.state_db_urls:
            logger.warning(f"No API URL defined for state {state_code}")
            return {}
        
        base_url = self.state_db_urls[state_code]
        
        # Different states have different endpoints and formats
        if state_code == "CA":
            url = f"{base_url}/detail/{company_number}"
        elif state_code == "DE":
            url = f"{base_url}/entity/{company_number}"
        elif state_code == "NY":
            url = f"{base_url}/corporation/{company_number}"
        elif state_code == "TX":
            url = f"{base_url}/entity/{company_number}"
        elif state_code == "FL":
            url = f"{base_url}/detail/{company_number}"
        else:
            logger.warning(f"Details endpoint not available for state {state_code}")
            return {}
        
        try:
            response = self._make_request(url)
            data = response.json()
            
            # Format varies by state, this is a simplified example
            if state_code == "CA":
                company = data.get("entity", {})
                result = self._format_state_company(company, state_code)
                
                # Add additional details that might be available
                result["registered_agent"] = data.get("agent", {}).get("name")
                result["registered_address"] = data.get("agent", {}).get("address")
                result["officers"] = data.get("officers", [])
                result["filings"] = data.get("filings", [])
                
                return result
            elif state_code == "DE":
                # Similar formatting for Delaware
                pass
            # Add more state-specific formatting as needed
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting company details from {state_code} state database: {e}")
            return {}
    
    def _get_company_officers(self, company_number: str, jurisdiction: str) -> List[Dict]:
        """
        Get officers and directors for a company from OpenCorporates.
        
        Args:
            company_number: Company registration number
            jurisdiction: Jurisdiction code
            
        Returns:
            List of dictionaries containing officer information
        """
        if "opencorporates" not in self.api_keys:
            logger.warning("OpenCorporates API key not provided")
            return []
        
        api_key = self.api_keys["opencorporates"]
        
        try:
            url = f"{self.opencorporates_url}/companies/{jurisdiction}/{company_number}/officers"
            params = {"api_token": api_key}
            
            response = self._make_request(url, params=params)
            data = response.json()
            
            officers_data = data.get("results", {}).get("officers", [])
            
            officers = []
            for officer_data in officers_data:
                officer = officer_data.get("officer", {})
                officers.append({
                    "name": officer.get("name"),
                    "position": officer.get("position"),
                    "start_date": officer.get("start_date"),
                    "end_date": officer.get("end_date"),
                    "nationality": officer.get("nationality"),
                    "occupation": officer.get("occupation"),
                    "address": officer.get("address")
                })
            
            return officers
            
        except Exception as e:
            logger.error(f"Error getting company officers from OpenCorporates: {e}")
            return []
    
    def _get_company_filings(self, company_number: str, jurisdiction: str) -> List[Dict]:
        """
        Get filing history for a company from OpenCorporates.
        
        Args:
            company_number: Company registration number
            jurisdiction: Jurisdiction code
            
        Returns:
            List of dictionaries containing filing information
        """
        if "opencorporates" not in self.api_keys:
            logger.warning("OpenCorporates API key not provided")
            return []
        
        api_key = self.api_keys["opencorporates"]
        
        try:
            url = f"{self.opencorporates_url}/companies/{jurisdiction}/{company_number}/filings"
            params = {"api_token": api_key}
            
            response = self._make_request(url, params=params)
            data = response.json()
            
            filings_data = data.get("results", {}).get("filings", [])
            
            filings = []
            for filing_data in filings_data:
                filing = filing_data.get("filing", {})
                filings.append({
                    "title": filing.get("title"),
                    "description": filing.get("description"),
                    "date": filing.get("date"),
                    "filing_type": filing.get("filing_type"),
                    "url": filing.get("opencorporates_url")
                })
            
            return filings
            
        except Exception as e:
            logger.error(f"Error getting company filings from OpenCorporates: {e}")
            return []

    def get_corporate_network(self, company_number: str, jurisdiction: str, max_depth: int = 2) -> Dict:
        """
        Get the corporate network (parent/subsidiary relationships) for a company.
        
        Args:
            company_number: Company registration number
            jurisdiction: Jurisdiction code
            max_depth: Maximum depth of relationships to retrieve
            
        Returns:
            Dictionary containing corporate network information
        """
        logger.info(f"Getting corporate network for company {company_number} in {jurisdiction}")
        
        # Create a cache key
        cache_key = self._get_cache_key("corporate_network", f"{jurisdiction}_{company_number}_{max_depth}")
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached corporate network")
            return cached_result
        
        # Get company details first
        company = self.get_company_details(company_number, jurisdiction)
        if not company:
            return {"error": "Company not found"}
        
        # Initialize network
        network = {
            "company": company,
            "parents": [],
            "subsidiaries": [],
            "related_companies": []
        }
        
        # Only OpenCorporates has corporate network data
        if "opencorporates" not in self.api_keys:
            logger.warning("OpenCorporates API key not provided")
            self._save_to_cache(cache_key, network)
            return network
        
        api_key = self.api_keys["opencorporates"]
        
        # Get parent companies
        try:
            url = f"{self.opencorporates_url}/companies/{jurisdiction}/{company_number}/network"
            params = {"api_token": api_key}
            
            response = self._make_request(url, params=params)
            data = response.json()
            
            relationships = data.get("results", {}).get("relationships", [])
            
            for relationship in relationships:
                rel_type = relationship.get("relationship_type")
                related_company = relationship.get("related_company", {})
                
                if rel_type == "parent":
                    network["parents"].append({
                        "company_name": related_company.get("name"),
                        "company_number": related_company.get("company_number"),
                        "jurisdiction_code": related_company.get("jurisdiction_code"),
                        "percentage_owned": relationship.get("percentage_owned"),
                        "relationship_date": relationship.get("start_date")
                    })
                elif rel_type == "subsidiary":
                    network["subsidiaries"].append({
                        "company_name": related_company.get("name"),
                        "company_number": related_company.get("company_number"),
                        "jurisdiction_code": related_company.get("jurisdiction_code"),
                        "percentage_owned": relationship.get("percentage_owned"),
                        "relationship_date": relationship.get("start_date")
                    })
                else:
                    network["related_companies"].append({
                        "company_name": related_company.get("name"),
                        "company_number": related_company.get("company_number"),
                        "jurisdiction_code": related_company.get("jurisdiction_code"),
                        "relationship_type": rel_type,
                        "relationship_date": relationship.get("start_date")
                    })
            
            # If max_depth > 1, recursively get relationships for parent and subsidiary companies
            if max_depth > 1:
                for parent in network["parents"]:
                    parent_number = parent.get("company_number")
                    parent_jurisdiction = parent.get("jurisdiction_code")
                    if parent_number and parent_jurisdiction:
                        try:
                            parent_network = self.get_corporate_network(
                                parent_number, 
                                parent_jurisdiction, 
                                max_depth - 1
                            )
                            parent["network"] = parent_network
                        except Exception as e:
                            logger.warning(f"Error getting parent network: {e}")
                
                for subsidiary in network["subsidiaries"]:
                    sub_number = subsidiary.get("company_number")
                    sub_jurisdiction = subsidiary.get("jurisdiction_code")
                    if sub_number and sub_jurisdiction:
                        try:
                            sub_network = self.get_corporate_network(
                                sub_number, 
                                sub_jurisdiction, 
                                max_depth - 1
                            )
                            subsidiary["network"] = sub_network
                        except Exception as e:
                            logger.warning(f"Error getting subsidiary network: {e}")
            
            # Cache and return result
            self._save_to_cache(cache_key, network)
            return network
            
        except Exception as e:
            logger.error(f"Error getting corporate network from OpenCorporates: {e}")
            self._save_to_cache(cache_key, network)
            return network
    
    def get_officer_companies(self, officer_name: str, jurisdiction: Optional[str] = None, limit: int = 50) -> Dict:
        """
        Get companies associated with a specific officer or director.
        
        Args:
            officer_name: Name of the officer or director
            jurisdiction: Optional jurisdiction code to filter results
            limit: Maximum number of results to return
            
        Returns:
            Dictionary containing officer information and associated companies
        """
        logger.info(f"Getting companies for officer {officer_name}")
        
        # Create a cache key
        cache_key = self._get_cache_key("officer_companies", f"{officer_name}_{jurisdiction}_{limit}")
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached officer companies")
            return cached_result
        
        # Only OpenCorporates has officer search
        if "opencorporates" not in self.api_keys:
            logger.warning("OpenCorporates API key not provided")
            return {"error": "OpenCorporates API key required"}
        
        api_key = self.api_keys["opencorporates"]
        
        try:
            url = f"{self.opencorporates_url}/officers/search"
            params = {
                "api_token": api_key,
                "q": officer_name,
                "per_page": min(limit, 100)
            }
            
            if jurisdiction:
                params["jurisdiction_code"] = jurisdiction
            
            response = self._make_request(url, params=params)
            data = response.json()
            
            officers_data = data.get("results", {}).get("officers", [])
            
            if not officers_data:
                return {
                    "officer_name": officer_name,
                    "companies": [],
                    "total_companies": 0
                }
            
            # Get the first matching officer
            officer = officers_data[0].get("officer", {})
            officer_id = officer.get("id")
            
            if not officer_id:
                return {
                    "officer_name": officer_name,
                    "companies": [],
                    "total_companies": 0
                }
            
            # Get companies for this officer
            url = f"{self.opencorporates_url}/officers/{officer_id}"
            response = self._make_request(url, params={"api_token": api_key})
            data = response.json()
            
            officer_data = data.get("results", {}).get("officer", {})
            companies_data = officer_data.get("companies", [])
            
            companies = []
            for company_data in companies_data:
                company = company_data.get("company", {})
                companies.append({
                    "company_name": company.get("name"),
                    "company_number": company.get("company_number"),
                    "jurisdiction_code": company.get("jurisdiction_code"),
                    "jurisdiction_name": company.get("jurisdiction_name"),
                    "position": company_data.get("position"),
                    "start_date": company_data.get("start_date"),
                    "end_date": company_data.get("end_date")
                })
            
            result = {
                "officer_name": officer_data.get("name", officer_name),
                "officer_id": officer_id,
                "companies": companies,
                "total_companies": len(companies)
            }
            
            # Cache and return result
            self._save_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting officer companies from OpenCorporates: {e}")
            return {
                "officer_name": officer_name,
                "error": str(e),
                "companies": [],
                "total_companies": 0
            }
    
    def detect_shell_companies(self, companies: List[Dict]) -> List[Dict]:
        """
        Analyze a list of companies to detect potential shell companies based on common indicators.
        
        Args:
            companies: List of company dictionaries from search_companies or batch_process
            
        Returns:
            List of companies with shell company indicators
        """
        logger.info(f"Analyzing {len(companies)} companies for shell company indicators")
        
        shell_indicators = []
        
        for company in companies:
            indicators = []
            score = 0
            
            # Check for common shell company indicators
            
            # 1. Registered in known tax havens
            jurisdiction = company.get("jurisdiction_code", "")
            tax_havens = ["bvi", "vg", "ky", "im", "je", "gg", "pa", "bs"]
            if any(haven in jurisdiction for haven in tax_havens):
                indicators.append("Registered in potential tax haven")
                score += 3
            
            # 2. No physical address or shared address with many companies
            address = company.get("registered_address", "")
            if not address or address.lower().find("p.o. box") >= 0:
                indicators.append("No physical address or P.O. Box only")
                score += 2
            
            # 3. Few or no officers/directors
            officers = company.get("officers", [])
            if len(officers) <= 1:
                indicators.append("One or no officers/directors")
                score += 2
            
            # 4. Recently incorporated with little activity
            incorporation_date = company.get("incorporation_date", "")
            filings = company.get("filings", [])
            if incorporation_date and len(filings) <= 1:
                indicators.append("Recently incorporated with minimal filing activity")
                score += 2
            
            # 5. Generic or suspicious company name
            name = company.get("company_name", "").lower()
            suspicious_terms = ["holding", "group", "international", "overseas", "consulting", "investment", "trading"]
            if any(term in name for term in suspicious_terms):
                indicators.append("Generic company name with common shell company terms")
                score += 1
            
            # Add to results if score is high enough
            if score >= 3:
                shell_indicators.append({
                    "company": company,
                    "shell_company_score": score,
                    "indicators": indicators
                })
        
        # Sort by score (highest first)
        shell_indicators.sort(key=lambda x: x["shell_company_score"], reverse=True)
        
        return shell_indicators
    
    def batch_process_companies(self, company_names: List[str], jurisdiction: Optional[str] = None) -> Dict[str, Dict]:
        """
        Batch process multiple companies to retrieve their registration information.
        
        Args:
            company_names: List of company names to process
            jurisdiction: Optional jurisdiction code to filter results
            
        Returns:
            Dictionary mapping company names to their registration information
        """
        logger.info(f"Batch processing {len(company_names)} companies")
        
        # Create a cache key
        cache_key = self._get_cache_key("batch_companies", f"{','.join(company_names)}_{jurisdiction}")
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached batch results")
            return cached_result
        
        results = {}
        
        for company_name in company_names:
            try:
                # Search for the company
                search_results = self.search_companies(
                    company_name=company_name,
                    jurisdiction=jurisdiction,
                    limit=1
                )
                
                if search_results:
                    # Get the first result
                    company = search_results[0]
                    company_number = company.get("company_number")
                    company_jurisdiction = company.get("jurisdiction_code")
                    
                    # Get detailed information
                    if company_number and company_jurisdiction:
                        details = self.get_company_details(company_number, company_jurisdiction)
                        results[company_name] = details
                    else:
                        results[company_name] = company
                else:
                    results[company_name] = {"error": "Company not found"}
            except Exception as e:
                logger.error(f"Error processing company {company_name}: {e}")
                results[company_name] = {"error": str(e)}
        
        # Cache and return results
        self._save_to_cache(cache_key, results)
        return results
    
    def export_to_csv(self, data: List[Dict], output_file: str) -> str:
        """
        Export company data to a CSV file.
        
        Args:
            data: List of company dictionaries
            output_file: Path to output CSV file
            
        Returns:
            Path to the created CSV file
        """
        logger.info(f"Exporting data to CSV: {output_file}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Write to CSV
        df.to_csv(output_file, index=False)
        
        return output_file
    
    def export_to_excel(self, data: Dict[str, List[Dict]], output_file: str) -> str:
        """
        Export multiple datasets to an Excel file with multiple sheets.
        
        Args:
            data: Dictionary mapping sheet names to lists of dictionaries
            output_file: Path to output Excel file
            
        Returns:
            Path to the created Excel file
        """
        logger.info(f"Exporting data to Excel: {output_file}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Create Excel writer
        with pd.ExcelWriter(output_file) as writer:
            for sheet_name, sheet_data in data.items():
                # Convert to DataFrame
                df = pd.DataFrame(sheet_data)
                
                # Write to Excel sheet
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return output_file
