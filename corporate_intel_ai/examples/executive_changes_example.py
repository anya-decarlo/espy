#!/usr/bin/env python
"""
Example script demonstrating the usage of the ExecutiveChangesTracker module.

This script shows how to:
1. Initialize the executive changes tracker
2. Extract executive changes from SEC filings
3. Extract executive changes from company websites
4. Compile executive profiles
5. Analyze executive networks
6. Correlate executive changes with company performance
7. Generate timelines of leadership changes
8. Export data to CSV and Excel formats
"""

import os
import sys
import json
from datetime import datetime
from pprint import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from corporate_intel_ai.src.data_collection.executive_changes import ExecutiveChangesTracker

def main():
    # Initialize the executive changes tracker
    tracker = ExecutiveChangesTracker(
        output_dir="data/executive_changes",
        use_cache=True
    )
    
    print("=" * 80)
    print("Corporate Intelligence Automation - Executive Changes Example")
    print("=" * 80)
    
    # Example 1: Extract executive changes from SEC filings
    print("\n1. Extracting executive changes from SEC filings")
    print("-" * 50)
    
    try:
        # Get executive changes for a company
        company_ticker = "AAPL"  # Apple Inc.
        sec_changes = tracker.get_executive_changes_from_sec(
            ticker=company_ticker,
            start_date="2022-01-01",
            end_date="2023-01-01"
        )
        
        print(f"Found {len(sec_changes)} executive changes for {company_ticker} from SEC filings:")
        for i, change in enumerate(sec_changes[:3], 1):
            print(f"\n{i}. {change.get('name')} - {change.get('change_type').replace('_', ' ').title()}")
            print(f"   Previous Title: {change.get('previous_title') or 'N/A'}")
            print(f"   New Title: {change.get('new_title') or 'N/A'}")
            print(f"   Date: {change.get('detection_date')}")
        
        if len(sec_changes) > 3:
            print(f"   ... and {len(sec_changes) - 3} more")
    except Exception as e:
        print(f"Error extracting changes from SEC filings: {e}")
    
    # Example 2: Extract executive changes from company websites
    print("\n2. Extracting executive changes from company websites")
    print("-" * 50)
    
    try:
        # Get executive changes from company website
        company_name = "Apple"
        website_url = "https://www.apple.com"
        
        website_changes = tracker.get_executive_changes_from_website(
            company_name=company_name,
            website_url=website_url
        )
        
        print(f"Found {len(website_changes)} executive changes for {company_name} from website:")
        for i, change in enumerate(website_changes[:3], 1):
            print(f"\n{i}. {change.get('name')} - {change.get('change_type').replace('_', ' ').title()}")
            print(f"   Previous Title: {change.get('previous_title') or 'N/A'}")
            print(f"   New Title: {change.get('new_title') or 'N/A'}")
            print(f"   Date: {change.get('detection_date')}")
            print(f"   Source: {change.get('source')}")
        
        if len(website_changes) > 3:
            print(f"   ... and {len(website_changes) - 3} more")
    except Exception as e:
        print(f"Error extracting changes from website: {e}")
    
    # Example 3: Get executive changes from LinkedIn
    print("\n3. Extracting executive changes from LinkedIn")
    print("-" * 50)
    
    try:
        # Note: LinkedIn scraping is limited by their terms of service
        # This is a simulated example
        linkedin_changes = tracker.get_executive_changes_from_linkedin(
            company_name=company_name,
            limit=5
        )
        
        print(f"Found {len(linkedin_changes)} executive changes for {company_name} from LinkedIn:")
        for i, change in enumerate(linkedin_changes, 1):
            print(f"\n{i}. {change.get('name')} - {change.get('change_type').replace('_', ' ').title()}")
            print(f"   Previous Title: {change.get('previous_title') or 'N/A'}")
            print(f"   New Title: {change.get('new_title') or 'N/A'}")
            print(f"   Date: {change.get('detection_date')}")
            print(f"   Source: {change.get('source')}")
    except Exception as e:
        print(f"Error extracting changes from LinkedIn: {e}")
    
    # Example 4: Compile executive profile
    print("\n4. Compiling executive profile")
    print("-" * 50)
    
    try:
        # Get profile for an executive
        executive_name = "Tim Cook"
        profile = tracker.get_executive_profile(
            name=executive_name,
            company=company_name
        )
        
        print(f"Profile for {profile.get('name')}:")
        print(f"Current Company: {profile.get('current_company')}")
        print(f"Current Title: {profile.get('current_title')}")
        
        # Print education
        education = profile.get('education', [])
        if education:
            print("\nEducation:")
            for i, edu in enumerate(education, 1):
                print(f"{i}. {edu.get('degree')} in {edu.get('field')} from {edu.get('institution')} ({edu.get('year')})")
        
        # Print previous roles
        roles = profile.get('previous_roles', [])
        if roles:
            print("\nPrevious Roles:")
            for i, role in enumerate(roles, 1):
                print(f"{i}. {role.get('title')} at {role.get('company')} ({role.get('start_date')} to {role.get('end_date')})")
        
        # Print compensation
        compensation = profile.get('compensation')
        if compensation:
            print(f"\nCompensation ({compensation.get('year')}):")
            print(f"Salary: {compensation.get('salary')}")
            print(f"Bonus: {compensation.get('bonus')}")
            print(f"Stock Options: {compensation.get('stock_options')}")
            print(f"Total: {compensation.get('total')}")
    except Exception as e:
        print(f"Error compiling executive profile: {e}")
    
    # Example 5: Analyze executive network
    print("\n5. Analyzing executive network")
    print("-" * 50)
    
    try:
        # Analyze executive network
        network = tracker.analyze_executive_network(
            company_name=company_name,
            depth=2
        )
        
        print(f"Executive network for {network.get('company')}:")
        print(f"Executives: {', '.join(network.get('executives', []))}")
        
        # Print connections
        connections = network.get('connections', [])
        if connections:
            print(f"\nConnections ({len(connections)}):")
            for i, conn in enumerate(connections[:5], 1):
                print(f"{i}. {conn.get('source')} → {conn.get('target')} ({conn.get('type')})")
            
            if len(connections) > 5:
                print(f"   ... and {len(connections) - 5} more")
        
        # Print network metrics
        metrics = network.get('metrics', {})
        if metrics:
            print("\nNetwork Metrics:")
            
            # Degree centrality (most connected nodes)
            degree_centrality = metrics.get('degree_centrality', {})
            if degree_centrality:
                print("Top nodes by degree centrality (connectedness):")
                sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
                for i, (node, value) in enumerate(sorted_nodes[:3], 1):
                    print(f"{i}. {node}: {value}")
    except Exception as e:
        print(f"Error analyzing executive network: {e}")
    
    # Example 6: Correlate with company performance
    print("\n6. Correlating executive changes with company performance")
    print("-" * 50)
    
    try:
        # Correlate executive changes with performance
        correlation = tracker.correlate_with_performance(
            company_ticker=company_ticker,
            start_date="2022-01-01",
            end_date="2023-01-01"
        )
        
        print(f"Correlation analysis for {correlation.get('company')}:")
        print(f"Period: {correlation.get('period', {}).get('start_date')} to {correlation.get('period', {}).get('end_date')}")
        
        # Print summary
        summary = correlation.get('summary', {})
        if summary:
            print(f"\nSummary:")
            print(f"Total Changes: {summary.get('total_changes')}")
            print(f"Positive Impact: {summary.get('positive_impact')}")
            print(f"Negative Impact: {summary.get('negative_impact')}")
            print(f"Average Impact: {summary.get('average_impact')}%")
        
        # Print correlations
        correlations = correlation.get('correlations', [])
        if correlations:
            print(f"\nDetailed Correlations ({len(correlations)}):")
            for i, corr in enumerate(correlations[:3], 1):
                print(f"\n{i}. {corr.get('executive')} - {corr.get('change_type').replace('_', ' ').title()}")
                print(f"   Title: {corr.get('title')}")
                print(f"   Date: {corr.get('change_date')}")
                print(f"   Stock Change After: {corr.get('percent_change_after')}%")
                print(f"   Impact Assessment: {corr.get('impact_assessment').title()}")
            
            if len(correlations) > 3:
                print(f"   ... and {len(correlations) - 3} more")
    except Exception as e:
        print(f"Error correlating with performance: {e}")
    
    # Example 7: Generate timeline
    print("\n7. Generating executive changes timeline")
    print("-" * 50)
    
    try:
        # Generate timeline
        timeline = tracker.generate_timeline(
            company_ticker=company_ticker,
            start_date="2018-01-01",
            end_date="2023-01-01"
        )
        
        print(f"Executive changes timeline for {timeline.get('company')}:")
        print(f"Period: {timeline.get('period', {}).get('start_date')} to {timeline.get('period', {}).get('end_date')}")
        
        # Print summary
        summary = timeline.get('summary', {})
        if summary:
            print(f"\nSummary:")
            print(f"Total Changes: {summary.get('total_changes')}")
            print(f"Appointments: {summary.get('appointments')}")
            print(f"Departures: {summary.get('departures')}")
            print(f"Role Changes: {summary.get('role_changes')}")
        
        # Print timeline
        timeline_data = timeline.get('timeline', {})
        if timeline_data:
            print(f"\nTimeline by Month:")
            for i, (month, changes) in enumerate(sorted(timeline_data.items())[:5], 1):
                print(f"\n{i}. {month} ({len(changes)} changes):")
                for j, change in enumerate(changes[:2], 1):
                    print(f"   {j}. {change.get('name')} - {change.get('change_type').replace('_', ' ').title()}")
                
                if len(changes) > 2:
                    print(f"      ... and {len(changes) - 2} more")
            
            if len(timeline_data) > 5:
                print(f"\n   ... and {len(timeline_data) - 5} more months")
    except Exception as e:
        print(f"Error generating timeline: {e}")
    
    # Example 8: Export data
    print("\n8. Exporting data to CSV and Excel")
    print("-" * 50)
    
    try:
        # Combine all changes
        all_changes = sec_changes + website_changes + linkedin_changes
        
        # Export to CSV
        csv_file = tracker.export_to_csv(
            all_changes,
            "data/executive_changes/all_executive_changes.csv"
        )
        print(f"Exported all changes to CSV: {csv_file}")
        
        # Export to Excel with multiple sheets
        excel_data = {
            "SEC Changes": sec_changes,
            "Website Changes": website_changes,
            "LinkedIn Changes": linkedin_changes,
            "Performance Correlation": correlation.get('correlations', [])
        }
        
        excel_file = tracker.export_to_excel(
            excel_data,
            "data/executive_changes/executive_analysis.xlsx"
        )
        print(f"Exported executive analysis to Excel: {excel_file}")
    except Exception as e:
        print(f"Error exporting data: {e}")
    
    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)
    print("\nNOTE: This example uses simulated data for demonstration purposes.")
    print("In a real scenario, you would need to provide proper API keys and handle rate limiting.")

if __name__ == "__main__":
    main()
