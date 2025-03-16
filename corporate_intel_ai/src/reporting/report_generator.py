"""
Report Generator Module

This module generates comprehensive competitive intelligence reports based on
analyzed data from various sources. It creates structured reports with visualizations
for strategic decision-making.
"""

import logging
import os
from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import jinja2
import pdfkit
import json

# Configure logging
logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Class for generating competitive intelligence reports.
    """
    
    def __init__(self, output_dir: str = "reports", template_dir: str = "templates"):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory to store generated reports
            template_dir: Directory containing report templates
        """
        self.output_dir = output_dir
        self.template_dir = template_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up Jinja2 template environment
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    def generate_market_positioning_report(self, 
                                         company: str,
                                         analysis_data: Dict,
                                         output_format: str = "pdf") -> str:
        """
        Generate a market positioning report.
        
        Args:
            company: Company name or ticker
            analysis_data: Dictionary containing analysis data
            output_format: Output format (pdf, html, docx)
            
        Returns:
            Path to the generated report
        """
        logger.info(f"Generating market positioning report for {company}")
        
        # Format report date
        report_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create report filename
        filename = f"{company.lower().replace(' ', '_')}_market_positioning_{report_date}"
        
        # Prepare template context
        context = {
            "company": company,
            "report_date": report_date,
            "report_title": f"{company} Market Positioning Analysis",
            "analysis_data": analysis_data,
            "generated_by": "Corporate Intelligence Automation"
        }
        
        # Generate report based on format
        if output_format == "pdf":
            return self._generate_pdf_report("market_positioning_template.html", context, filename)
        elif output_format == "html":
            return self._generate_html_report("market_positioning_template.html", context, filename)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def generate_competitive_landscape_report(self,
                                            industry: str,
                                            companies: List[str],
                                            analysis_data: Dict,
                                            output_format: str = "pdf") -> str:
        """
        Generate a competitive landscape report.
        
        Args:
            industry: Industry name
            companies: List of company names or tickers
            analysis_data: Dictionary containing analysis data
            output_format: Output format (pdf, html, docx)
            
        Returns:
            Path to the generated report
        """
        logger.info(f"Generating competitive landscape report for {industry}")
        
        # Format report date
        report_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create report filename
        filename = f"{industry.lower().replace(' ', '_')}_competitive_landscape_{report_date}"
        
        # Prepare template context
        context = {
            "industry": industry,
            "companies": companies,
            "report_date": report_date,
            "report_title": f"{industry} Competitive Landscape Analysis",
            "analysis_data": analysis_data,
            "generated_by": "Corporate Intelligence Automation"
        }
        
        # Generate report based on format
        if output_format == "pdf":
            return self._generate_pdf_report("competitive_landscape_template.html", context, filename)
        elif output_format == "html":
            return self._generate_html_report("competitive_landscape_template.html", context, filename)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def generate_strategic_intelligence_report(self,
                                             company: str,
                                             analysis_data: Dict,
                                             output_format: str = "pdf") -> str:
        """
        Generate a strategic intelligence report.
        
        Args:
            company: Company name or ticker
            analysis_data: Dictionary containing analysis data
            output_format: Output format (pdf, html, docx)
            
        Returns:
            Path to the generated report
        """
        logger.info(f"Generating strategic intelligence report for {company}")
        
        # Format report date
        report_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create report filename
        filename = f"{company.lower().replace(' ', '_')}_strategic_intelligence_{report_date}"
        
        # Prepare template context
        context = {
            "company": company,
            "report_date": report_date,
            "report_title": f"{company} Strategic Intelligence Analysis",
            "analysis_data": analysis_data,
            "generated_by": "Corporate Intelligence Automation"
        }
        
        # Generate report based on format
        if output_format == "pdf":
            return self._generate_pdf_report("strategic_intelligence_template.html", context, filename)
        elif output_format == "html":
            return self._generate_html_report("strategic_intelligence_template.html", context, filename)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_html_report(self, template_name: str, context: Dict, filename: str) -> str:
        """
        Generate an HTML report using a template.
        
        Args:
            template_name: Name of the template file
            context: Dictionary containing template context
            filename: Base filename for the report
            
        Returns:
            Path to the generated HTML report
        """
        try:
            # Get template
            template = self.template_env.get_template(template_name)
            
            # Render template with context
            html_content = template.render(**context)
            
            # Save HTML to file
            output_path = os.path.join(self.output_dir, f"{filename}.html")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Generated HTML report: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            raise
    
    def _generate_pdf_report(self, template_name: str, context: Dict, filename: str) -> str:
        """
        Generate a PDF report using a template.
        
        Args:
            template_name: Name of the template file
            context: Dictionary containing template context
            filename: Base filename for the report
            
        Returns:
            Path to the generated PDF report
        """
        try:
            # First generate HTML
            html_path = self._generate_html_report(template_name, context, filename)
            
            # Convert HTML to PDF
            output_path = os.path.join(self.output_dir, f"{filename}.pdf")
            
            # Use pdfkit to convert HTML to PDF
            pdfkit_options = {
                'page-size': 'A4',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': 'UTF-8',
                'no-outline': None
            }
            
            pdfkit.from_file(html_path, output_path, options=pdfkit_options)
            
            logger.info(f"Generated PDF report: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            raise
    
    def generate_executive_summary(self, analysis_data: Dict, max_length: int = 500) -> str:
        """
        Generate an executive summary from analysis data.
        
        Args:
            analysis_data: Dictionary containing analysis data
            max_length: Maximum length of the executive summary in characters
            
        Returns:
            Executive summary text
        """
        logger.info("Generating executive summary")
        
        # TODO: Implement actual summary generation logic
        # This would extract key points from analysis data
        
        return "Executive summary placeholder"
    
    def export_data_to_json(self, data: Dict, filename: str) -> str:
        """
        Export analysis data to JSON format.
        
        Args:
            data: Dictionary containing data to export
            filename: Filename for the JSON file
            
        Returns:
            Path to the generated JSON file
        """
        output_path = os.path.join(self.output_dir, f"{filename}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported data to JSON: {output_path}")
        return output_path
