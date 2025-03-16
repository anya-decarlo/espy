"""
Funding Data Collection Module

This module handles the collection of private equity and venture funding data from various sources.
It provides functionality to search, download, and analyze funding rounds, investors, and startups
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

# Configure logging
logger = logging.getLogger(__name__)

class FundingDataCollector:
    """
    Class for collecting and processing private equity and venture funding data.
    """
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None, output_dir: str = "data/funding",
                 use_cache: bool = True, cache_expiry_days: int = 30):
        """
        Initialize the funding data collector.
        
        Args:
            api_keys: Dictionary of API keys for different data sources (crunchbase, pitchbook, cbinsights)
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
        self.crunchbase_url = "https://api.crunchbase.com/api/v4"
        self.pitchbook_url = "https://api.pitchbook.com/v1"
        self.cbinsights_url = "https://api.cbinsights.com/v1"
        
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
    
    def search_funding_rounds(self, 
                             company_name: Optional[str] = None,
                             investor_name: Optional[str] = None,
                             industry: Optional[str] = None,
                             funding_type: Optional[str] = None,
                             min_amount: Optional[float] = None,
                             max_amount: Optional[float] = None,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             limit: int = 50) -> List[Dict]:
        """
        Search for funding rounds based on various criteria.
        
        Args:
            company_name: Name of the company receiving funding
            investor_name: Name of the investor
            industry: Industry or sector
            funding_type: Type of funding (seed, series_a, series_b, etc.)
            min_amount: Minimum funding amount in USD
            max_amount: Maximum funding amount in USD
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing funding round data
        """
        logger.info(f"Searching for funding rounds with criteria: company={company_name}, investor={investor_name}")
        
        # Create a cache key based on search parameters
        cache_params = f"{company_name}_{investor_name}_{industry}_{funding_type}_{min_amount}_{max_amount}_{start_date}_{end_date}_{limit}"
        cache_key = self._get_cache_key("funding_search", cache_params)
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached funding round search results")
            return cached_result
        
        # Try multiple sources for funding data
        results = []
        
        # Try Crunchbase first
        try:
            crunchbase_results = self._search_crunchbase_funding(
                company_name=company_name,
                investor_name=investor_name,
                industry=industry,
                funding_type=funding_type,
                min_amount=min_amount,
                max_amount=max_amount,
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
            if crunchbase_results:
                results.extend(crunchbase_results)
        except Exception as e:
            logger.warning(f"Crunchbase search failed: {e}")
        
        # Try PitchBook if we have an API key
        if "pitchbook" in self.api_keys:
            try:
                pitchbook_results = self._search_pitchbook_funding(
                    company_name=company_name,
                    investor_name=investor_name,
                    industry=industry,
                    funding_type=funding_type,
                    min_amount=min_amount,
                    max_amount=max_amount,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit
                )
                if pitchbook_results:
                    results.extend(pitchbook_results)
            except Exception as e:
                logger.warning(f"PitchBook search failed: {e}")
        
        # Try CB Insights if we have an API key
        if "cbinsights" in self.api_keys:
            try:
                cbinsights_results = self._search_cbinsights_funding(
                    company_name=company_name,
                    investor_name=investor_name,
                    industry=industry,
                    funding_type=funding_type,
                    min_amount=min_amount,
                    max_amount=max_amount,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit
                )
                if cbinsights_results:
                    results.extend(cbinsights_results)
            except Exception as e:
                logger.warning(f"CB Insights search failed: {e}")
        
        # Remove duplicates based on unique identifiers
        unique_results = {}
        for result in results:
            # Create a unique identifier based on company, date, and amount
            company = result.get("company_name", "")
            date = result.get("announced_date", "")
            amount = result.get("amount_usd", 0)
            unique_id = f"{company}_{date}_{amount}"
            
            if unique_id not in unique_results:
                unique_results[unique_id] = result
        
        results = list(unique_results.values())
        
        # Sort by date (most recent first)
        results.sort(key=lambda x: x.get("announced_date", ""), reverse=True)
        
        # Limit results
        results = results[:limit]
        
        # Cache and return results
        self._save_to_cache(cache_key, results)
        return results
    
    def _search_crunchbase_funding(self,
                                  company_name: Optional[str] = None,
                                  investor_name: Optional[str] = None,
                                  industry: Optional[str] = None,
                                  funding_type: Optional[str] = None,
                                  min_amount: Optional[float] = None,
                                  max_amount: Optional[float] = None,
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  limit: int = 50) -> List[Dict]:
        """
        Search for funding rounds using Crunchbase API.
        
        Args:
            company_name: Name of the company receiving funding
            investor_name: Name of the investor
            industry: Industry or sector
            funding_type: Type of funding (seed, series_a, series_b, etc.)
            min_amount: Minimum funding amount in USD
            max_amount: Maximum funding amount in USD
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing funding round data
        """
        if "crunchbase" not in self.api_keys:
            logger.warning("Crunchbase API key not provided")
            return []
        
        api_key = self.api_keys["crunchbase"]
        
        # Construct query parameters
        params = {
            "user_key": api_key,
            "limit": limit
        }
        
        # Build query
        query = {}
        
        # Add field-specific filters
        if company_name:
            query["organization_name"] = {"contains": company_name}
        
        if investor_name:
            query["investors"] = {"contains": investor_name}
        
        if industry:
            query["organization_industries"] = {"contains": industry}
        
        if funding_type:
            query["funding_type"] = {"eq": funding_type}
        
        if min_amount:
            query["money_raised_usd"] = query.get("money_raised_usd", {})
            query["money_raised_usd"]["gte"] = min_amount
        
        if max_amount:
            query["money_raised_usd"] = query.get("money_raised_usd", {})
            query["money_raised_usd"]["lte"] = max_amount
        
        if start_date:
            query["announced_on"] = query.get("announced_on", {})
            query["announced_on"]["gte"] = start_date
        
        if end_date:
            query["announced_on"] = query.get("announced_on", {})
            query["announced_on"]["lte"] = end_date
        
        # Add query to params
        if query:
            params["query"] = json.dumps(query)
        
        # Make the request
        try:
            url = f"{self.crunchbase_url}/searches/funding_rounds"
            response = self._make_request(url, params=params)
            data = response.json()
            
            # Extract funding round information
            results = []
            
            for item in data.get("data", {}).get("items", []):
                properties = item.get("properties", {})
                
                # Get organization data
                organization = item.get("relationships", {}).get("organization", {}).get("properties", {})
                
                # Get investor data
                investors = []
                for investor in item.get("relationships", {}).get("investors", {}).get("items", []):
                    investor_properties = investor.get("properties", {})
                    investors.append({
                        "name": investor_properties.get("name"),
                        "type": investor_properties.get("investor_type"),
                        "location": investor_properties.get("location_identifiers", [{}])[0].get("value") if investor_properties.get("location_identifiers") else None
                    })
                
                # Format the result
                result = {
                    "source": "crunchbase",
                    "funding_round_id": properties.get("uuid"),
                    "company_name": organization.get("name"),
                    "company_description": organization.get("short_description"),
                    "company_location": organization.get("location_identifiers", [{}])[0].get("value") if organization.get("location_identifiers") else None,
                    "company_industries": [industry.get("value") for industry in organization.get("industry_identifiers", [])],
                    "announced_date": properties.get("announced_on"),
                    "funding_type": properties.get("investment_type"),
                    "amount_usd": properties.get("money_raised_usd"),
                    "amount_local": properties.get("money_raised"),
                    "currency": properties.get("money_raised_currency_code"),
                    "post_money_valuation": properties.get("post_money_valuation_usd"),
                    "investors": investors,
                    "lead_investors": [investor.get("name") for investor in investors if investor.get("is_lead_investor")]
                }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Crunchbase: {e}")
            return []
    
    def _search_pitchbook_funding(self,
                                 company_name: Optional[str] = None,
                                 investor_name: Optional[str] = None,
                                 industry: Optional[str] = None,
                                 funding_type: Optional[str] = None,
                                 min_amount: Optional[float] = None,
                                 max_amount: Optional[float] = None,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 limit: int = 50) -> List[Dict]:
        """
        Search for funding rounds using PitchBook API.
        
        Args:
            company_name: Name of the company receiving funding
            investor_name: Name of the investor
            industry: Industry or sector
            funding_type: Type of funding (seed, series_a, series_b, etc.)
            min_amount: Minimum funding amount in USD
            max_amount: Maximum funding amount in USD
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing funding round data
        """
        if "pitchbook" not in self.api_keys:
            logger.warning("PitchBook API key not provided")
            return []
        
        api_key = self.api_keys["pitchbook"]
        
        # PitchBook API requires a different header format
        headers = {
            "X-API-Key": api_key
        }
        
        # Construct query parameters
        params = {
            "limit": limit
        }
        
        # Add field-specific filters
        if company_name:
            params["companyName"] = company_name
        
        if investor_name:
            params["investorName"] = investor_name
        
        if industry:
            params["industry"] = industry
        
        if funding_type:
            params["dealType"] = funding_type
        
        if min_amount:
            params["dealSizeMin"] = min_amount
        
        if max_amount:
            params["dealSizeMax"] = max_amount
        
        if start_date:
            params["announcedDateMin"] = start_date
        
        if end_date:
            params["announcedDateMax"] = end_date
        
        # Make the request
        try:
            url = f"{self.pitchbook_url}/deals"
            response = self._make_request(url, headers=headers, params=params)
            data = response.json()
            
            # Extract funding round information
            results = []
            
            for deal in data.get("deals", []):
                # Format the result
                result = {
                    "source": "pitchbook",
                    "funding_round_id": deal.get("dealId"),
                    "company_name": deal.get("companyName"),
                    "company_description": deal.get("companyDescription"),
                    "company_location": f"{deal.get('companyCity')}, {deal.get('companyState')}, {deal.get('companyCountry')}",
                    "company_industries": deal.get("industries", []),
                    "announced_date": deal.get("announcedDate"),
                    "funding_type": deal.get("dealType"),
                    "amount_usd": deal.get("dealSize"),
                    "currency": "USD",  # PitchBook typically normalizes to USD
                    "post_money_valuation": deal.get("postValuation"),
                    "investors": [{"name": investor} for investor in deal.get("investors", [])],
                    "lead_investors": [investor for investor in deal.get("leadInvestors", [])]
                }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching PitchBook: {e}")
            return []
    
    def _search_cbinsights_funding(self,
                                  company_name: Optional[str] = None,
                                  investor_name: Optional[str] = None,
                                  industry: Optional[str] = None,
                                  funding_type: Optional[str] = None,
                                  min_amount: Optional[float] = None,
                                  max_amount: Optional[float] = None,
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  limit: int = 50) -> List[Dict]:
        """
        Search for funding rounds using CB Insights API.
        
        Args:
            company_name: Name of the company receiving funding
            investor_name: Name of the investor
            industry: Industry or sector
            funding_type: Type of funding (seed, series_a, series_b, etc.)
            min_amount: Minimum funding amount in USD
            max_amount: Maximum funding amount in USD
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing funding round data
        """
        if "cbinsights" not in self.api_keys:
            logger.warning("CB Insights API key not provided")
            return []
        
        api_key = self.api_keys["cbinsights"]
        
        # CB Insights API requires a different header format
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        # Construct query parameters
        params = {
            "limit": limit
        }
        
        # Build query
        query = {}
        
        # Add field-specific filters
        if company_name:
            query["company_name"] = company_name
        
        if investor_name:
            query["investors"] = investor_name
        
        if industry:
            query["industry"] = industry
        
        if funding_type:
            query["funding_type"] = funding_type
        
        if min_amount:
            query["raised_min"] = min_amount
        
        if max_amount:
            query["raised_max"] = max_amount
        
        if start_date:
            query["date_min"] = start_date
        
        if end_date:
            query["date_max"] = end_date
        
        # Add query to params
        if query:
            params["query"] = json.dumps(query)
        
        # Make the request
        try:
            url = f"{self.cbinsights_url}/funding-rounds"
            response = self._make_request(url, headers=headers, params=params)
            data = response.json()
            
            # Extract funding round information
            results = []
            
            for round_data in data.get("data", []):
                # Format the result
                result = {
                    "source": "cbinsights",
                    "funding_round_id": round_data.get("id"),
                    "company_name": round_data.get("company", {}).get("name"),
                    "company_description": round_data.get("company", {}).get("description"),
                    "company_location": round_data.get("company", {}).get("location"),
                    "company_industries": round_data.get("company", {}).get("industries", []),
                    "announced_date": round_data.get("announced_date"),
                    "funding_type": round_data.get("funding_type"),
                    "amount_usd": round_data.get("raised_usd"),
                    "currency": round_data.get("raised_currency"),
                    "post_money_valuation": round_data.get("post_valuation_usd"),
                    "investors": [{"name": investor.get("name"), "type": investor.get("type")} for investor in round_data.get("investors", [])],
                    "lead_investors": [investor.get("name") for investor in round_data.get("investors", []) if investor.get("is_lead")]
                }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching CB Insights: {e}")
            return []
    
    def get_company_funding_history(self, company_name: str) -> Dict:
        """
        Get the complete funding history for a company.
        
        Args:
            company_name: Name of the company
            
        Returns:
            Dictionary containing company information and funding history
        """
        logger.info(f"Getting funding history for {company_name}")
        
        # Create a cache key
        cache_key = self._get_cache_key("company_funding", company_name)
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached funding history for {company_name}")
            return cached_result
        
        # Search for funding rounds for this company
        funding_rounds = self.search_funding_rounds(company_name=company_name, limit=100)
        
        # Sort rounds by date
        funding_rounds.sort(key=lambda x: x.get("announced_date", ""))
        
        # Extract company information from the most recent round
        company_info = {}
        if funding_rounds:
            latest_round = funding_rounds[-1]
            company_info = {
                "name": latest_round.get("company_name"),
                "description": latest_round.get("company_description"),
                "location": latest_round.get("company_location"),
                "industries": latest_round.get("company_industries")
            }
        
        # Calculate total funding
        total_funding = sum(round.get("amount_usd", 0) or 0 for round in funding_rounds)
        
        # Get latest valuation
        latest_valuation = None
        for round in reversed(funding_rounds):
            if round.get("post_money_valuation"):
                latest_valuation = round.get("post_money_valuation")
                break
        
        # Get all investors
        all_investors = set()
        for round in funding_rounds:
            for investor in round.get("investors", []):
                if isinstance(investor, dict):
                    all_investors.add(investor.get("name"))
                else:
                    all_investors.add(investor)
        
        # Compile result
        result = {
            "company": company_info,
            "total_funding_usd": total_funding,
            "latest_valuation_usd": latest_valuation,
            "funding_rounds": funding_rounds,
            "investors": list(all_investors)
        }
        
        # Cache and return result
        self._save_to_cache(cache_key, result)
        return result
    
    def get_investor_portfolio(self, investor_name: str) -> Dict:
        """
        Get the investment portfolio for an investor.
        
        Args:
            investor_name: Name of the investor
            
        Returns:
            Dictionary containing investor information and portfolio
        """
        logger.info(f"Getting investment portfolio for {investor_name}")
        
        # Create a cache key
        cache_key = self._get_cache_key("investor_portfolio", investor_name)
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached portfolio for {investor_name}")
            return cached_result
        
        # Search for funding rounds with this investor
        funding_rounds = self.search_funding_rounds(investor_name=investor_name, limit=100)
        
        # Extract portfolio companies
        portfolio = {}
        for round in funding_rounds:
            company_name = round.get("company_name")
            if not company_name:
                continue
                
            if company_name not in portfolio:
                portfolio[company_name] = {
                    "name": company_name,
                    "description": round.get("company_description"),
                    "location": round.get("company_location"),
                    "industries": round.get("company_industries"),
                    "funding_rounds": []
                }
                
            portfolio[company_name]["funding_rounds"].append({
                "date": round.get("announced_date"),
                "type": round.get("funding_type"),
                "amount_usd": round.get("amount_usd"),
                "is_lead": investor_name in round.get("lead_investors", [])
            })
        
        # Calculate portfolio metrics
        total_companies = len(portfolio)
        total_investments = sum(
            sum(round.get("amount_usd", 0) or 0 for round in company.get("funding_rounds", []))
            for company in portfolio.values()
        )
        
        # Count investments by industry
        industry_investments = {}
        for company in portfolio.values():
            for industry in company.get("industries", []):
                if industry not in industry_investments:
                    industry_investments[industry] = 0
                    
                industry_investments[industry] += sum(
                    round.get("amount_usd", 0) or 0 
                    for round in company.get("funding_rounds", [])
                )
        
        # Count investments by stage
        stage_investments = {}
        for company in portfolio.values():
            for round in company.get("funding_rounds", []):
                stage = round.get("type", "Unknown")
                if stage not in stage_investments:
                    stage_investments[stage] = 0
                    
                stage_investments[stage] += round.get("amount_usd", 0) or 0
        
        # Compile result
        result = {
            "investor_name": investor_name,
            "total_companies": total_companies,
            "total_investments_usd": total_investments,
            "portfolio_companies": list(portfolio.values()),
            "industry_investments": industry_investments,
            "stage_investments": stage_investments
        }
        
        # Cache and return result
        self._save_to_cache(cache_key, result)
        return result
    
    def get_industry_funding_trends(self, industry: str, start_date: Optional[str] = None, 
                                   end_date: Optional[str] = None, limit: int = 100) -> Dict:
        """
        Get funding trends for a specific industry.
        
        Args:
            industry: Industry or sector to analyze
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of funding rounds to analyze
            
        Returns:
            Dictionary containing industry funding trends
        """
        logger.info(f"Getting funding trends for industry: {industry}")
        
        # Create a cache key
        cache_params = f"{industry}_{start_date}_{end_date}_{limit}"
        cache_key = self._get_cache_key("industry_trends", cache_params)
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached industry trends for {industry}")
            return cached_result
        
        # Search for funding rounds in this industry
        funding_rounds = self.search_funding_rounds(
            industry=industry,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        # Group rounds by quarter
        quarters = {}
        for round in funding_rounds:
            date_str = round.get("announced_date")
            if not date_str:
                continue
                
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
                quarter = f"{date.year}Q{(date.month - 1) // 3 + 1}"
                
                if quarter not in quarters:
                    quarters[quarter] = {
                        "quarter": quarter,
                        "total_rounds": 0,
                        "total_funding": 0,
                        "avg_round_size": 0,
                        "rounds": []
                    }
                    
                quarters[quarter]["total_rounds"] += 1
                quarters[quarter]["total_funding"] += round.get("amount_usd", 0) or 0
                quarters[quarter]["rounds"].append(round)
                
            except Exception as e:
                logger.warning(f"Error parsing date {date_str}: {e}")
        
        # Calculate average round size for each quarter
        for quarter in quarters.values():
            if quarter["total_rounds"] > 0:
                quarter["avg_round_size"] = quarter["total_funding"] / quarter["total_rounds"]
        
        # Sort quarters chronologically
        sorted_quarters = sorted(quarters.values(), key=lambda x: x["quarter"])
        
        # Extract top companies by funding
        companies = {}
        for round in funding_rounds:
            company_name = round.get("company_name")
            if not company_name:
                continue
                
            if company_name not in companies:
                companies[company_name] = {
                    "name": company_name,
                    "total_funding": 0,
                    "rounds": []
                }
                
            companies[company_name]["total_funding"] += round.get("amount_usd", 0) or 0
            companies[company_name]["rounds"].append(round)
        
        # Sort companies by total funding
        top_companies = sorted(
            companies.values(), 
            key=lambda x: x["total_funding"], 
            reverse=True
        )[:10]  # Top 10 companies
        
        # Extract top investors
        investors = {}
        for round in funding_rounds:
            for investor in round.get("investors", []):
                investor_name = investor.get("name") if isinstance(investor, dict) else investor
                if not investor_name:
                    continue
                    
                if investor_name not in investors:
                    investors[investor_name] = {
                        "name": investor_name,
                        "total_investments": 0,
                        "companies_invested": set()
                    }
                    
                investors[investor_name]["total_investments"] += 1
                investors[investor_name]["companies_invested"].add(round.get("company_name", ""))
        
        # Convert sets to lists for JSON serialization
        for investor in investors.values():
            investor["companies_invested"] = list(investor["companies_invested"])
        
        # Sort investors by number of investments
        top_investors = sorted(
            investors.values(),
            key=lambda x: x["total_investments"],
            reverse=True
        )[:10]  # Top 10 investors
        
        # Compile result
        result = {
            "industry": industry,
            "total_funding": sum(quarter["total_funding"] for quarter in quarters.values()),
            "total_rounds": sum(quarter["total_rounds"] for quarter in quarters.values()),
            "quarterly_trends": sorted_quarters,
            "top_companies": top_companies,
            "top_investors": top_investors
        }
        
        # Cache and return result
        self._save_to_cache(cache_key, result)
        return result
    
    def get_competitor_funding(self, company_names: List[str]) -> Dict:
        """
        Compare funding data for a list of competitor companies.
        
        Args:
            company_names: List of company names to compare
            
        Returns:
            Dictionary containing comparative funding data
        """
        logger.info(f"Comparing funding data for competitors: {company_names}")
        
        # Create a cache key
        cache_key = self._get_cache_key("competitor_funding", "_".join(company_names))
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached competitor funding data")
            return cached_result
        
        # Get funding history for each company
        companies = {}
        for company_name in company_names:
            try:
                funding_history = self.get_company_funding_history(company_name)
                companies[company_name] = funding_history
            except Exception as e:
                logger.warning(f"Error getting funding history for {company_name}: {e}")
                companies[company_name] = {"error": str(e)}
        
        # Extract comparative metrics
        comparison = {
            "total_funding": {company: data.get("total_funding_usd", 0) for company, data in companies.items()},
            "latest_valuation": {company: data.get("latest_valuation_usd") for company, data in companies.items()},
            "funding_rounds_count": {company: len(data.get("funding_rounds", [])) for company, data in companies.items()},
            "investors_count": {company: len(data.get("investors", [])) for company, data in companies.items()},
            "latest_round": {company: self._get_latest_round(data.get("funding_rounds", [])) for company, data in companies.items()},
            "funding_timeline": self._create_funding_timeline(companies)
        }
        
        # Find common investors
        all_investors = {}
        for company, data in companies.items():
            for investor in data.get("investors", []):
                if investor not in all_investors:
                    all_investors[investor] = set()
                all_investors[investor].add(company)
        
        common_investors = {investor: list(companies) for investor, companies in all_investors.items() if len(companies) > 1}
        
        # Compile result
        result = {
            "companies": companies,
            "comparison": comparison,
            "common_investors": common_investors
        }
        
        # Cache and return result
        self._save_to_cache(cache_key, result)
        return result
    
    def _get_latest_round(self, funding_rounds: List[Dict]) -> Optional[Dict]:
        """
        Get the latest funding round from a list of rounds.
        
        Args:
            funding_rounds: List of funding round dictionaries
            
        Returns:
            Latest funding round or None if no rounds
        """
        if not funding_rounds:
            return None
            
        # Sort by date
        sorted_rounds = sorted(
            funding_rounds,
            key=lambda x: x.get("announced_date", ""),
            reverse=True
        )
        
        return sorted_rounds[0]
    
    def _create_funding_timeline(self, companies: Dict[str, Dict]) -> Dict:
        """
        Create a timeline of funding events for multiple companies.
        
        Args:
            companies: Dictionary of company funding data
            
        Returns:
            Dictionary with timeline data
        """
        timeline = []
        
        for company_name, data in companies.items():
            for round in data.get("funding_rounds", []):
                date_str = round.get("announced_date")
                if not date_str:
                    continue
                    
                timeline.append({
                    "date": date_str,
                    "company": company_name,
                    "event": f"{round.get('funding_type', 'Funding')} Round",
                    "amount_usd": round.get("amount_usd"),
                    "details": round
                })
        
        # Sort by date
        timeline.sort(key=lambda x: x["date"])
        
        return timeline
    
    def batch_process_companies(self, company_names: List[str]) -> Dict[str, Dict]:
        """
        Process funding data for multiple companies in batch.
        
        Args:
            company_names: List of company names to process
            
        Returns:
            Dictionary mapping company names to their funding data
        """
        logger.info(f"Batch processing funding data for {len(company_names)} companies")
        
        results = {}
        for company_name in company_names:
            try:
                results[company_name] = self.get_company_funding_history(company_name)
            except Exception as e:
                logger.error(f"Error processing {company_name}: {e}")
                results[company_name] = {"error": str(e)}
        
        return results
    
    def batch_process_investors(self, investor_names: List[str]) -> Dict[str, Dict]:
        """
        Process portfolio data for multiple investors in batch.
        
        Args:
            investor_names: List of investor names to process
            
        Returns:
            Dictionary mapping investor names to their portfolio data
        """
        logger.info(f"Batch processing portfolio data for {len(investor_names)} investors")
        
        results = {}
        for investor_name in investor_names:
            try:
                results[investor_name] = self.get_investor_portfolio(investor_name)
            except Exception as e:
                logger.error(f"Error processing {investor_name}: {e}")
                results[investor_name] = {"error": str(e)}
        
        return results
    
    def export_to_csv(self, data: List[Dict], output_file: str) -> str:
        """
        Export funding data to CSV format.
        
        Args:
            data: List of funding round dictionaries
            output_file: Path to output CSV file
            
        Returns:
            Path to the created CSV file
        """
        logger.info(f"Exporting funding data to {output_file}")
        
        if not data:
            logger.warning("No data to export")
            return None
        
        # Create a DataFrame from the data
        df = pd.DataFrame(data)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Export to CSV
        df.to_csv(output_file, index=False)
        
        return output_file
    
    def export_to_excel(self, data: Dict, output_file: str) -> str:
        """
        Export funding data to Excel format with multiple sheets.
        
        Args:
            data: Dictionary containing multiple data sets to export as sheets
            output_file: Path to output Excel file
            
        Returns:
            Path to the created Excel file
        """
        logger.info(f"Exporting funding data to Excel: {output_file}")
        
        if not data:
            logger.warning("No data to export")
            return None
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Create an Excel writer
        with pd.ExcelWriter(output_file) as writer:
            for sheet_name, sheet_data in data.items():
                # Skip non-list data
                if not isinstance(sheet_data, list):
                    continue
                    
                if not sheet_data:
                    continue
                
                # Create a DataFrame from the data
                df = pd.DataFrame(sheet_data)
                
                # Write to Excel
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return output_file
