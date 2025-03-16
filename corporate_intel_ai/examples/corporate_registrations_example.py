#!/usr/bin/env python
"""
Example script demonstrating the usage of the CorporateRegistrationsCollector module.

This script shows how to:
1. Initialize the corporate registrations collector
2. Search for companies based on various criteria
3. Get detailed company information
4. Explore corporate networks (parent/subsidiary relationships)
5. Find companies associated with specific officers
6. Detect potential shell companies
7. Batch process multiple companies
8. Export data to CSV and Excel formats
"""

import os
import sys
import json
from datetime import datetime
from pprint import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from corporate_intel_ai.src.data_collection.corporate_registrations import CorporateRegistrationsCollector

def main():
    # Initialize the corporate registrations collector
    # Note: You should replace these with your actual API keys
    api_keys = {
        "opencorporates": "YOUR_OPENCORPORATES_API_KEY"
    }
    
    # For demo purposes, we'll use a mock API key
    # In a real scenario, you would use actual API keys
    collector = CorporateRegistrationsCollector(
        api_keys=api_keys,
        output_dir="data/corporate_registrations",
        use_cache=True
    )
    
    print("=" * 80)
    print("Corporate Intelligence Automation - Corporate Registrations Example")
    print("=" * 80)
    
    # Example 1: Search for companies
    print("\n1. Searching for companies")
    print("-" * 50)
    
    try:
        # Search for a specific company
        companies = collector.search_companies(
            company_name="Apple Inc",
            jurisdiction="us_ca",  # California
            limit=3
        )
        
        print(f"Found {len(companies)} companies matching 'Apple Inc' in California:")
        for i, company in enumerate(companies, 1):
            print(f"\n{i}. {company.get('company_name')}")
            print(f"   Registration Number: {company.get('company_number')}")
            print(f"   Jurisdiction: {company.get('jurisdiction_name')}")
            print(f"   Status: {company.get('status')}")
            print(f"   Source: {company.get('source')}")
    except Exception as e:
        print(f"Error searching for companies: {e}")
    
    # Example 2: Get detailed company information
    print("\n2. Getting detailed company information")
    print("-" * 50)
    
    try:
        # Get details for a specific company
        # Note: In a real scenario, you would use actual company numbers and jurisdictions
        company_details = collector.get_company_details(
            company_number="C0806592",  # Example company number
            jurisdiction="us_ca"        # California
        )
        
        if company_details:
            print(f"Details for {company_details.get('company_name')}:")
            print(f"Registration Number: {company_details.get('company_number')}")
            print(f"Jurisdiction: {company_details.get('jurisdiction_name')}")
            print(f"Incorporation Date: {company_details.get('incorporation_date')}")
            print(f"Status: {company_details.get('status')}")
            
            # Print officer information if available
            officers = company_details.get('officers', [])
            if officers:
                print(f"\nOfficers ({len(officers)}):")
                for i, officer in enumerate(officers[:3], 1):
                    print(f"{i}. {officer.get('name')} - {officer.get('position')}")
                if len(officers) > 3:
                    print(f"   ... and {len(officers) - 3} more")
            
            # Print filing information if available
            filings = company_details.get('filings', [])
            if filings:
                print(f"\nRecent Filings ({len(filings)}):")
                for i, filing in enumerate(filings[:3], 1):
                    print(f"{i}. {filing.get('title')} - {filing.get('date')}")
                if len(filings) > 3:
                    print(f"   ... and {len(filings) - 3} more")
        else:
            print("Company details not found")
    except Exception as e:
        print(f"Error getting company details: {e}")
    
    # Example 3: Get corporate network
    print("\n3. Exploring corporate network")
    print("-" * 50)
    
    try:
        # Get corporate network for a specific company
        # Note: In a real scenario, you would use actual company numbers and jurisdictions
        network = collector.get_corporate_network(
            company_number="C0806592",  # Example company number
            jurisdiction="us_ca",       # California
            max_depth=1                 # Only immediate relationships
        )
        
        if network and "error" not in network:
            print(f"Corporate network for {network.get('company', {}).get('company_name')}:")
            
            # Print parent companies
            parents = network.get('parents', [])
            if parents:
                print(f"\nParent Companies ({len(parents)}):")
                for i, parent in enumerate(parents, 1):
                    print(f"{i}. {parent.get('company_name')} - {parent.get('jurisdiction_code')}")
                    print(f"   Ownership: {parent.get('percentage_owned', 'Unknown')}%")
            else:
                print("\nNo parent companies found")
            
            # Print subsidiary companies
            subsidiaries = network.get('subsidiaries', [])
            if subsidiaries:
                print(f"\nSubsidiaries ({len(subsidiaries)}):")
                for i, subsidiary in enumerate(subsidiaries, 1):
                    print(f"{i}. {subsidiary.get('company_name')} - {subsidiary.get('jurisdiction_code')}")
                    print(f"   Ownership: {subsidiary.get('percentage_owned', 'Unknown')}%")
            else:
                print("\nNo subsidiaries found")
        else:
            print("Corporate network not found or error occurred")
    except Exception as e:
        print(f"Error getting corporate network: {e}")
    
    # Example 4: Get officer companies
    print("\n4. Finding companies associated with an officer")
    print("-" * 50)
    
    try:
        # Get companies associated with a specific officer
        officer_companies = collector.get_officer_companies(
            officer_name="Tim Cook",
            limit=5
        )
        
        if officer_companies and "error" not in officer_companies:
            print(f"Companies associated with {officer_companies.get('officer_name')}:")
            print(f"Total companies: {officer_companies.get('total_companies')}")
            
            companies = officer_companies.get('companies', [])
            for i, company in enumerate(companies, 1):
                print(f"\n{i}. {company.get('company_name')} - {company.get('jurisdiction_name')}")
                print(f"   Position: {company.get('position')}")
                print(f"   Period: {company.get('start_date')} to {company.get('end_date') or 'Present'}")
        else:
            print(f"No companies found for officer or error occurred: {officer_companies.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"Error getting officer companies: {e}")
    
    # Example 5: Detect shell companies
    print("\n5. Detecting potential shell companies")
    print("-" * 50)
    
    try:
        # Search for companies in known tax havens
        tax_haven_companies = collector.search_companies(
            jurisdiction="vg",  # British Virgin Islands
            limit=10
        )
        
        # Analyze for shell company indicators
        shell_companies = collector.detect_shell_companies(tax_haven_companies)
        
        print(f"Found {len(shell_companies)} potential shell companies:")
        for i, shell in enumerate(shell_companies[:3], 1):
            company = shell.get('company', {})
            print(f"\n{i}. {company.get('company_name')} - {company.get('jurisdiction_name')}")
            print(f"   Shell Company Score: {shell.get('shell_company_score')}/10")
            print(f"   Indicators:")
            for indicator in shell.get('indicators', []):
                print(f"   - {indicator}")
        if len(shell_companies) > 3:
            print(f"   ... and {len(shell_companies) - 3} more")
    except Exception as e:
        print(f"Error detecting shell companies: {e}")
    
    # Example 6: Batch process companies
    print("\n6. Batch processing multiple companies")
    print("-" * 50)
    
    try:
        companies = ["Microsoft Corporation", "Amazon.com Inc", "Google LLC"]
        batch_results = collector.batch_process_companies(companies)
        
        print(f"Batch processed {len(batch_results)} companies:")
        for company, data in batch_results.items():
            if "error" in data:
                print(f"{company}: Error - {data['error']}")
            else:
                print(f"{company}: {data.get('company_number')} in {data.get('jurisdiction_name')}")
                print(f"   Status: {data.get('status')}")
                print(f"   Incorporation Date: {data.get('incorporation_date')}")
                print()
    except Exception as e:
        print(f"Error batch processing companies: {e}")
    
    # Example 7: Export data
    print("\n7. Exporting data to CSV and Excel")
    print("-" * 50)
    
    try:
        # Export company search results to CSV
        csv_file = collector.export_to_csv(
            companies,
            "data/corporate_registrations/company_search_results.csv"
        )
        print(f"Exported company search results to CSV: {csv_file}")
        
        # Export multiple datasets to Excel
        excel_data = {
            "Companies": companies,
            "Shell Companies": [s.get('company') for s in shell_companies],
            "Officer Companies": officer_companies.get('companies', [])
        }
        
        excel_file = collector.export_to_excel(
            excel_data,
            "data/corporate_registrations/corporate_analysis.xlsx"
        )
        print(f"Exported corporate analysis to Excel: {excel_file}")
    except Exception as e:
        print(f"Error exporting data: {e}")
    
    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)
    print("\nNOTE: This example uses placeholder API keys and company identifiers.")
    print("To use with real data, replace the API keys and company identifiers with actual values.")

if __name__ == "__main__":
    main()
