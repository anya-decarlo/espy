#!/usr/bin/env python
"""
Example script demonstrating the usage of the FundingDataCollector module.

This script shows how to:
1. Initialize the funding data collector
2. Search for funding rounds based on various criteria
3. Get company funding history
4. Get investor portfolio information
5. Analyze industry funding trends
6. Compare competitor funding data
7. Batch process multiple companies and investors
8. Export data to CSV and Excel formats
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pprint import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from corporate_intel_ai.src.data_collection.funding_data import FundingDataCollector

def main():
    # Initialize the funding data collector
    # Note: You should replace these with your actual API keys
    api_keys = {
        "crunchbase": "YOUR_CRUNCHBASE_API_KEY",
        "pitchbook": "YOUR_PITCHBOOK_API_KEY",
        "cbinsights": "YOUR_CBINSIGHTS_API_KEY"
    }
    
    # For demo purposes, we'll use a mock API key
    # In a real scenario, you would use actual API keys
    collector = FundingDataCollector(
        api_keys=api_keys,
        output_dir="data/funding",
        use_cache=True
    )
    
    print("=" * 80)
    print("Corporate Intelligence Automation - Funding Data Example")
    print("=" * 80)
    
    # Example 1: Search for funding rounds
    print("\n1. Searching for recent funding rounds in AI industry")
    print("-" * 50)
    
    # Calculate date 6 months ago
    six_months_ago = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    
    try:
        ai_funding = collector.search_funding_rounds(
            industry="artificial intelligence",
            start_date=six_months_ago,
            min_amount=1000000,  # $1M minimum
            limit=5
        )
        
        print(f"Found {len(ai_funding)} recent AI funding rounds:")
        for i, round in enumerate(ai_funding, 1):
            print(f"\n{i}. {round.get('company_name')} - {round.get('funding_type')} Round")
            print(f"   Amount: ${round.get('amount_usd', 0):,.2f}")
            print(f"   Date: {round.get('announced_date')}")
            print(f"   Investors: {', '.join([i.get('name') if isinstance(i, dict) else i for i in round.get('investors', [])])}")
    except Exception as e:
        print(f"Error searching for funding rounds: {e}")
    
    # Example 2: Get company funding history
    print("\n2. Getting funding history for a specific company")
    print("-" * 50)
    
    try:
        company_history = collector.get_company_funding_history("OpenAI")
        
        print(f"Funding history for {company_history.get('company', {}).get('name', 'OpenAI')}:")
        print(f"Total funding: ${company_history.get('total_funding_usd', 0):,.2f}")
        print(f"Latest valuation: ${company_history.get('latest_valuation_usd', 0):,.2f}")
        print(f"Number of funding rounds: {len(company_history.get('funding_rounds', []))}")
        print(f"Number of investors: {len(company_history.get('investors', []))}")
        
        print("\nFunding rounds:")
        for i, round in enumerate(company_history.get('funding_rounds', []), 1):
            print(f"{i}. {round.get('funding_type')} - ${round.get('amount_usd', 0):,.2f} ({round.get('announced_date')})")
    except Exception as e:
        print(f"Error getting company funding history: {e}")
    
    # Example 3: Get investor portfolio
    print("\n3. Getting investor portfolio")
    print("-" * 50)
    
    try:
        investor_portfolio = collector.get_investor_portfolio("Sequoia Capital")
        
        print(f"Portfolio for {investor_portfolio.get('investor_name', 'Sequoia Capital')}:")
        print(f"Total companies: {investor_portfolio.get('total_companies', 0)}")
        print(f"Total investments: ${investor_portfolio.get('total_investments_usd', 0):,.2f}")
        
        print("\nTop portfolio companies:")
        for i, company in enumerate(investor_portfolio.get('portfolio_companies', [])[:5], 1):
            print(f"{i}. {company.get('name')} - {', '.join(company.get('industries', []))}")
            print(f"   Rounds: {len(company.get('funding_rounds', []))}")
    except Exception as e:
        print(f"Error getting investor portfolio: {e}")
    
    # Example 4: Industry funding trends
    print("\n4. Analyzing industry funding trends")
    print("-" * 50)
    
    try:
        industry_trends = collector.get_industry_funding_trends(
            industry="fintech",
            start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")  # Last year
        )
        
        print(f"Funding trends for {industry_trends.get('industry', 'fintech')} industry:")
        print(f"Total funding: ${industry_trends.get('total_funding', 0):,.2f}")
        print(f"Total rounds: {industry_trends.get('total_rounds', 0)}")
        
        print("\nQuarterly trends:")
        for quarter in industry_trends.get('quarterly_trends', []):
            print(f"{quarter.get('quarter')}: ${quarter.get('total_funding', 0):,.2f} ({quarter.get('total_rounds', 0)} rounds)")
        
        print("\nTop companies by funding:")
        for i, company in enumerate(industry_trends.get('top_companies', [])[:3], 1):
            print(f"{i}. {company.get('name')} - ${company.get('total_funding', 0):,.2f}")
    except Exception as e:
        print(f"Error analyzing industry trends: {e}")
    
    # Example 5: Competitor funding comparison
    print("\n5. Comparing competitor funding")
    print("-" * 50)
    
    try:
        competitors = ["Anthropic", "Cohere", "Mistral AI"]
        competitor_funding = collector.get_competitor_funding(competitors)
        
        print(f"Funding comparison for AI competitors:")
        
        # Total funding comparison
        print("\nTotal funding:")
        for company, amount in competitor_funding.get('comparison', {}).get('total_funding', {}).items():
            print(f"{company}: ${amount:,.2f}")
        
        # Latest rounds
        print("\nLatest funding rounds:")
        for company, round in competitor_funding.get('comparison', {}).get('latest_round', {}).items():
            if round:
                print(f"{company}: {round.get('funding_type')} - ${round.get('amount_usd', 0):,.2f} ({round.get('announced_date')})")
            else:
                print(f"{company}: No funding round data available")
        
        # Common investors
        print("\nCommon investors:")
        common_investors = competitor_funding.get('common_investors', {})
        if common_investors:
            for investor, companies in list(common_investors.items())[:3]:
                print(f"{investor}: {', '.join(companies)}")
        else:
            print("No common investors found")
    except Exception as e:
        print(f"Error comparing competitor funding: {e}")
    
    # Example 6: Batch processing
    print("\n6. Batch processing multiple companies")
    print("-" * 50)
    
    try:
        companies = ["Stripe", "Plaid", "Chime"]
        batch_results = collector.batch_process_companies(companies)
        
        print(f"Batch processed {len(batch_results)} companies:")
        for company, data in batch_results.items():
            if "error" in data:
                print(f"{company}: Error - {data['error']}")
            else:
                print(f"{company}: ${data.get('total_funding_usd', 0):,.2f} total funding, {len(data.get('funding_rounds', []))} rounds")
    except Exception as e:
        print(f"Error batch processing companies: {e}")
    
    # Example 7: Export data
    print("\n7. Exporting data to CSV and Excel")
    print("-" * 50)
    
    try:
        # Export funding rounds to CSV
        csv_file = collector.export_to_csv(
            ai_funding,
            "data/funding/ai_funding_rounds.csv"
        )
        print(f"Exported funding rounds to CSV: {csv_file}")
        
        # Export multiple datasets to Excel
        excel_data = {
            "AI Funding": ai_funding,
            "Fintech Quarterly": industry_trends.get('quarterly_trends', []),
            "Top Fintech Companies": industry_trends.get('top_companies', [])
        }
        
        excel_file = collector.export_to_excel(
            excel_data,
            "data/funding/funding_analysis.xlsx"
        )
        print(f"Exported funding analysis to Excel: {excel_file}")
    except Exception as e:
        print(f"Error exporting data: {e}")
    
    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
