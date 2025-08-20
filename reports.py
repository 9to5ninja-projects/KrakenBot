"""
Reporting module for KrakenBot.
Generates automated reports on arbitrage opportunities and system performance.
"""
import os
import sys
import json
import csv
import datetime
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from health_monitor import get_monitor

class ReportGenerator:
    """Generates reports for KrakenBot."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.data_dir = config.DATA_DIR
        self.report_dir = config.DATA_DIR / "reports"
        self.report_dir.mkdir(exist_ok=True)
        
        # Health monitor
        self.health_monitor = get_monitor()
    
    def generate_daily_report(self, date: Optional[str] = None):
        """
        Generate a daily report for the specified date.
        
        Args:
            date: Date in YYYY-MM-DD format (default: yesterday)
            
        Returns:
            Path to the generated report
        """
        # Determine date
        if date:
            report_date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        else:
            report_date = datetime.date.today() - datetime.timedelta(days=1)
        
        date_str = report_date.strftime('%Y-%m-%d')
        logger.info(f"Generating daily report for {date_str}")
        
        # Prepare report data
        report_data = {
            'date': date_str,
            'generated_at': datetime.datetime.now().isoformat(),
            'opportunities': self._get_opportunities_for_date(report_date),
            'health': self._get_health_data_for_date(report_date),
            'summary': {}
        }
        
        # Calculate summary statistics
        self._calculate_summary_statistics(report_data)
        
        # Generate report file
        report_file = self.report_dir / f"daily_report_{date_str}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Daily report saved to {report_file}")
        
        # Generate HTML report
        html_file = self._generate_html_report(report_data, 'daily')
        
        return html_file
    
    def generate_weekly_report(self, end_date: Optional[str] = None):
        """
        Generate a weekly report ending on the specified date.
        
        Args:
            end_date: End date in YYYY-MM-DD format (default: yesterday)
            
        Returns:
            Path to the generated report
        """
        # Determine date range
        if end_date:
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
        else:
            end_date = datetime.date.today() - datetime.timedelta(days=1)
        
        start_date = end_date - datetime.timedelta(days=6)
        
        date_range = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
        logger.info(f"Generating weekly report for {date_range}")
        
        # Prepare report data
        report_data = {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'generated_at': datetime.datetime.now().isoformat(),
            'opportunities': [],
            'health': [],
            'summary': {},
            'daily_summaries': {}
        }
        
        # Collect data for each day in the range
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Get opportunities for this day
            daily_opportunities = self._get_opportunities_for_date(current_date)
            report_data['opportunities'].extend(daily_opportunities)
            
            # Get health data for this day
            daily_health = self._get_health_data_for_date(current_date)
            report_data['health'].extend(daily_health)
            
            # Calculate daily summary
            daily_summary = self._calculate_daily_summary(daily_opportunities)
            report_data['daily_summaries'][date_str] = daily_summary
            
            current_date += datetime.timedelta(days=1)
        
        # Calculate overall summary statistics
        self._calculate_summary_statistics(report_data)
        
        # Generate report file
        report_file = self.report_dir / f"weekly_report_{date_range}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Weekly report saved to {report_file}")
        
        # Generate HTML report
        html_file = self._generate_html_report(report_data, 'weekly')
        
        return html_file
    
    def generate_monthly_report(self, month: Optional[int] = None, year: Optional[int] = None):
        """
        Generate a monthly report for the specified month and year.
        
        Args:
            month: Month (1-12, default: last month)
            year: Year (default: current year or last year if last month is December)
            
        Returns:
            Path to the generated report
        """
        # Determine month and year
        today = datetime.date.today()
        
        if month is None:
            # Default to last month
            if today.month == 1:
                month = 12
                year = today.year - 1
            else:
                month = today.month - 1
                year = today.year
        
        if year is None:
            year = today.year
        
        # Validate month
        if not 1 <= month <= 12:
            raise ValueError(f"Invalid month: {month}")
        
        # Determine date range
        start_date = datetime.date(year, month, 1)
        
        # Calculate end date (last day of the month)
        if month == 12:
            end_date = datetime.date(year, 12, 31)
        else:
            end_date = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
        
        date_range = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
        logger.info(f"Generating monthly report for {datetime.date(year, month, 1).strftime('%B %Y')}")
        
        # Prepare report data
        report_data = {
            'month': month,
            'year': year,
            'month_name': datetime.date(year, month, 1).strftime('%B'),
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'generated_at': datetime.datetime.now().isoformat(),
            'opportunities': [],
            'health': [],
            'summary': {},
            'daily_summaries': {},
            'weekly_summaries': {}
        }
        
        # Collect data for each day in the month
        current_date = start_date
        current_week = 1
        week_start = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Get opportunities for this day
            daily_opportunities = self._get_opportunities_for_date(current_date)
            report_data['opportunities'].extend(daily_opportunities)
            
            # Get health data for this day
            daily_health = self._get_health_data_for_date(current_date)
            report_data['health'].extend(daily_health)
            
            # Calculate daily summary
            daily_summary = self._calculate_daily_summary(daily_opportunities)
            report_data['daily_summaries'][date_str] = daily_summary
            
            # Check if we need to end a week
            if current_date.weekday() == 6 or current_date == end_date:
                # Calculate weekly summary
                week_opportunities = [
                    op for op in report_data['opportunities']
                    if week_start <= datetime.datetime.fromisoformat(op['timestamp']).date() <= current_date
                ]
                
                week_range = f"{week_start.strftime('%Y-%m-%d')}_{current_date.strftime('%Y-%m-%d')}"
                weekly_summary = self._calculate_daily_summary(week_opportunities)  # Reuse the function
                report_data['weekly_summaries'][week_range] = weekly_summary
                
                # Start a new week
                current_week += 1
                week_start = current_date + datetime.timedelta(days=1)
            
            current_date += datetime.timedelta(days=1)
        
        # Calculate overall summary statistics
        self._calculate_summary_statistics(report_data)
        
        # Generate report file
        report_file = self.report_dir / f"monthly_report_{year}_{month:02d}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Monthly report saved to {report_file}")
        
        # Generate HTML report
        html_file = self._generate_html_report(report_data, 'monthly')
        
        return html_file
    
    def _get_opportunities_for_date(self, date: datetime.date) -> List[Dict[str, Any]]:
        """
        Get arbitrage opportunities for a specific date.
        
        Args:
            date: Date to get opportunities for
            
        Returns:
            List of opportunity dictionaries
        """
        opportunities_file = self.data_dir / "opportunities.csv"
        
        if not opportunities_file.exists():
            logger.warning(f"Opportunities file not found: {opportunities_file}")
            return []
        
        try:
            # Read CSV file
            df = pd.read_csv(opportunities_file)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by date
            date_start = datetime.datetime.combine(date, datetime.time.min)
            date_end = datetime.datetime.combine(date, datetime.time.max)
            
            df_filtered = df[(df['timestamp'] >= date_start) & (df['timestamp'] <= date_end)]
            
            # Convert to list of dictionaries
            opportunities = df_filtered.to_dict('records')
            
            # Convert timestamps to ISO format strings
            for op in opportunities:
                op['timestamp'] = op['timestamp'].isoformat()
            
            return opportunities
        
        except Exception as e:
            logger.error(f"Error reading opportunities file: {e}")
            return []
    
    def _get_health_data_for_date(self, date: datetime.date) -> List[Dict[str, Any]]:
        """
        Get health data for a specific date.
        
        Args:
            date: Date to get health data for
            
        Returns:
            List of health data dictionaries
        """
        health_file = self.data_dir / "health.json"
        
        if not health_file.exists():
            logger.warning(f"Health file not found: {health_file}")
            return []
        
        try:
            # Read JSON file
            with open(health_file, 'r') as f:
                health_data = json.load(f)
            
            # Filter by date if timestamp exists
            if 'timestamp' in health_data:
                health_timestamp = datetime.datetime.fromisoformat(health_data['timestamp']).date()
                
                if health_timestamp == date:
                    return [health_data]
            
            return []
        
        except Exception as e:
            logger.error(f"Error reading health file: {e}")
            return []
    
    def _calculate_daily_summary(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate summary statistics for a day's opportunities.
        
        Args:
            opportunities: List of opportunity dictionaries
            
        Returns:
            Dictionary of summary statistics
        """
        summary = {
            'total_opportunities': len(opportunities),
            'profitable_opportunities': 0,
            'total_profit': 0.0,
            'max_profit': 0.0,
            'avg_profit': 0.0,
            'paths': {}
        }
        
        if not opportunities:
            return summary
        
        # Calculate statistics
        profitable_ops = [op for op in opportunities if op.get('is_profitable', False)]
        summary['profitable_opportunities'] = len(profitable_ops)
        
        if profitable_ops:
            profits = [op.get('profit', 0.0) for op in profitable_ops]
            summary['total_profit'] = sum(profits)
            summary['max_profit'] = max(profits)
            summary['avg_profit'] = sum(profits) / len(profits)
        
        # Group by path
        for op in opportunities:
            path = op.get('path', '')
            if isinstance(path, list):
                path = '->'.join(path)
            
            if path not in summary['paths']:
                summary['paths'][path] = {
                    'total': 0,
                    'profitable': 0,
                    'total_profit': 0.0,
                    'max_profit': 0.0
                }
            
            summary['paths'][path]['total'] += 1
            
            if op.get('is_profitable', False):
                summary['paths'][path]['profitable'] += 1
                summary['paths'][path]['total_profit'] += op.get('profit', 0.0)
                summary['paths'][path]['max_profit'] = max(
                    summary['paths'][path]['max_profit'],
                    op.get('profit', 0.0)
                )
        
        return summary
    
    def _calculate_summary_statistics(self, report_data: Dict[str, Any]):
        """
        Calculate overall summary statistics for a report.
        
        Args:
            report_data: Report data dictionary
        """
        opportunities = report_data['opportunities']
        
        summary = {
            'total_opportunities': len(opportunities),
            'profitable_opportunities': 0,
            'total_profit': 0.0,
            'max_profit': 0.0,
            'avg_profit': 0.0,
            'win_rate': 0.0,
            'paths': {}
        }
        
        if not opportunities:
            report_data['summary'] = summary
            return
        
        # Calculate statistics
        profitable_ops = [op for op in opportunities if op.get('is_profitable', False)]
        summary['profitable_opportunities'] = len(profitable_ops)
        
        if profitable_ops:
            profits = [op.get('profit', 0.0) for op in profitable_ops]
            summary['total_profit'] = sum(profits)
            summary['max_profit'] = max(profits)
            summary['avg_profit'] = sum(profits) / len(profits)
        
        # Calculate win rate
        if summary['total_opportunities'] > 0:
            summary['win_rate'] = (summary['profitable_opportunities'] / summary['total_opportunities']) * 100
        
        # Group by path
        for op in opportunities:
            path = op.get('path', '')
            if isinstance(path, list):
                path = '->'.join(path)
            
            if path not in summary['paths']:
                summary['paths'][path] = {
                    'total': 0,
                    'profitable': 0,
                    'total_profit': 0.0,
                    'max_profit': 0.0,
                    'win_rate': 0.0
                }
            
            summary['paths'][path]['total'] += 1
            
            if op.get('is_profitable', False):
                summary['paths'][path]['profitable'] += 1
                summary['paths'][path]['total_profit'] += op.get('profit', 0.0)
                summary['paths'][path]['max_profit'] = max(
                    summary['paths'][path]['max_profit'],
                    op.get('profit', 0.0)
                )
        
        # Calculate win rate for each path
        for path in summary['paths']:
            if summary['paths'][path]['total'] > 0:
                summary['paths'][path]['win_rate'] = (
                    summary['paths'][path]['profitable'] / summary['paths'][path]['total']
                ) * 100
        
        report_data['summary'] = summary
    
    def _generate_html_report(self, report_data: Dict[str, Any], report_type: str) -> Path:
        """
        Generate an HTML report from report data.
        
        Args:
            report_data: Report data dictionary
            report_type: Type of report ('daily', 'weekly', or 'monthly')
            
        Returns:
            Path to the generated HTML file
        """
        # Determine report title and filename
        if report_type == 'daily':
            title = f"Daily Report - {report_data['date']}"
            filename = f"daily_report_{report_data['date']}.html"
        elif report_type == 'weekly':
            title = f"Weekly Report - {report_data['start_date']} to {report_data['end_date']}"
            filename = f"weekly_report_{report_data['start_date']}_{report_data['end_date']}.html"
        elif report_type == 'monthly':
            title = f"Monthly Report - {report_data['month_name']} {report_data['year']}"
            filename = f"monthly_report_{report_data['year']}_{report_data['month']:02d}.html"
        else:
            raise ValueError(f"Invalid report type: {report_type}")
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .header {{
                    background-color: #3498db;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    margin-bottom: 20px;
                    border-radius: 5px;
                }}
                .section {{
                    background-color: #f9f9f9;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .summary-box {{
                    display: inline-block;
                    background-color: #3498db;
                    color: white;
                    padding: 15px;
                    margin: 10px;
                    border-radius: 5px;
                    text-align: center;
                    min-width: 150px;
                }}
                .summary-box h3 {{
                    margin: 0;
                    color: white;
                }}
                .summary-box p {{
                    margin: 5px 0 0;
                    font-size: 24px;
                    font-weight: bold;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .profitable {{
                    color: green;
                    font-weight: bold;
                }}
                .not-profitable {{
                    color: red;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    padding: 20px;
                    color: #7f8c8d;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{title}</h1>
                    <p>Generated on {datetime.datetime.fromisoformat(report_data['generated_at']).strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>Summary</h2>
                    <div class="summary-box">
                        <h3>Total Opportunities</h3>
                        <p>{report_data['summary']['total_opportunities']}</p>
                    </div>
                    <div class="summary-box">
                        <h3>Profitable</h3>
                        <p>{report_data['summary']['profitable_opportunities']}</p>
                    </div>
                    <div class="summary-box">
                        <h3>Win Rate</h3>
                        <p>{report_data['summary']['win_rate']:.2f}%</p>
                    </div>
                    <div class="summary-box">
                        <h3>Total Profit</h3>
                        <p>${report_data['summary']['total_profit']:.2f}</p>
                    </div>
                    <div class="summary-box">
                        <h3>Avg Profit</h3>
                        <p>${report_data['summary']['avg_profit']:.2f}</p>
                    </div>
                    <div class="summary-box">
                        <h3>Max Profit</h3>
                        <p>${report_data['summary']['max_profit']:.2f}</p>
                    </div>
                </div>
        """
        
        # Add path summary
        html_content += """
                <div class="section">
                    <h2>Path Summary</h2>
                    <table>
                        <tr>
                            <th>Path</th>
                            <th>Total</th>
                            <th>Profitable</th>
                            <th>Win Rate</th>
                            <th>Total Profit</th>
                            <th>Max Profit</th>
                        </tr>
        """
        
        for path, stats in report_data['summary']['paths'].items():
            html_content += f"""
                        <tr>
                            <td>{path}</td>
                            <td>{stats['total']}</td>
                            <td>{stats['profitable']}</td>
                            <td>{stats['win_rate']:.2f}%</td>
                            <td>${stats['total_profit']:.2f}</td>
                            <td>${stats['max_profit']:.2f}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                </div>
        """
        
        # Add daily summaries for weekly and monthly reports
        if report_type in ['weekly', 'monthly']:
            html_content += """
                <div class="section">
                    <h2>Daily Summary</h2>
                    <table>
                        <tr>
                            <th>Date</th>
                            <th>Total</th>
                            <th>Profitable</th>
                            <th>Total Profit</th>
                            <th>Max Profit</th>
                        </tr>
            """
            
            for date, stats in sorted(report_data['daily_summaries'].items()):
                html_content += f"""
                        <tr>
                            <td>{date}</td>
                            <td>{stats['total_opportunities']}</td>
                            <td>{stats['profitable_opportunities']}</td>
                            <td>${stats['total_profit']:.2f}</td>
                            <td>${stats['max_profit']:.2f}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
        
        # Add weekly summaries for monthly reports
        if report_type == 'monthly' and 'weekly_summaries' in report_data:
            html_content += """
                <div class="section">
                    <h2>Weekly Summary</h2>
                    <table>
                        <tr>
                            <th>Week</th>
                            <th>Total</th>
                            <th>Profitable</th>
                            <th>Total Profit</th>
                            <th>Max Profit</th>
                        </tr>
            """
            
            for week_range, stats in sorted(report_data['weekly_summaries'].items()):
                html_content += f"""
                        <tr>
                            <td>{week_range}</td>
                            <td>{stats['total_opportunities']}</td>
                            <td>{stats['profitable_opportunities']}</td>
                            <td>${stats['total_profit']:.2f}</td>
                            <td>${stats['max_profit']:.2f}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
        
        # Add profitable opportunities
        profitable_ops = [op for op in report_data['opportunities'] if op.get('is_profitable', False)]
        
        if profitable_ops:
            html_content += """
                <div class="section">
                    <h2>Profitable Opportunities</h2>
                    <table>
                        <tr>
                            <th>Time</th>
                            <th>Path</th>
                            <th>Start Amount</th>
                            <th>End Amount</th>
                            <th>Profit</th>
                            <th>Profit %</th>
                        </tr>
            """
            
            for op in sorted(profitable_ops, key=lambda x: x.get('timestamp', ''), reverse=True):
                timestamp = op.get('timestamp', '')
                if timestamp:
                    try:
                        time_str = datetime.datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        time_str = timestamp
                else:
                    time_str = 'Unknown'
                
                path = op.get('path', '')
                if isinstance(path, list):
                    path = '->'.join(path)
                
                html_content += f"""
                        <tr>
                            <td>{time_str}</td>
                            <td>{path}</td>
                            <td>${op.get('start_amount', 0):.2f}</td>
                            <td>${op.get('end_amount', 0):.2f}</td>
                            <td class="profitable">${op.get('profit', 0):.2f}</td>
                            <td>{op.get('profit_percentage', 0):.4f}%</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
        
        # Add footer
        html_content += """
                <div class="footer">
                    <p>Generated by KrakenBot Reporting Module</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML file
        html_file = self.report_dir / filename
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {html_file}")
        
        return html_file


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="KrakenBot Reporting Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--type",
        choices=["daily", "weekly", "monthly"],
        default="daily",
        help="Type of report to generate"
    )
    
    parser.add_argument(
        "--date",
        type=str,
        help="Date for daily report (YYYY-MM-DD), end date for weekly report, or ignored for monthly report"
    )
    
    parser.add_argument(
        "--month",
        type=int,
        help="Month for monthly report (1-12)"
    )
    
    parser.add_argument(
        "--year",
        type=int,
        help="Year for monthly report"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add(
        config.LOG_DIR / "reports.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )
    logger.add(lambda msg: print(msg), level="INFO")
    
    # Parse arguments
    args = parse_args()
    
    # Print banner
    print("=" * 80)
    print("KrakenBot Reporting Tool".center(80))
    print("=" * 80)
    
    # Generate report
    report_generator = ReportGenerator()
    
    if args.type == "daily":
        report_file = report_generator.generate_daily_report(args.date)
        print(f"Daily report generated: {report_file}")
    
    elif args.type == "weekly":
        report_file = report_generator.generate_weekly_report(args.date)
        print(f"Weekly report generated: {report_file}")
    
    elif args.type == "monthly":
        report_file = report_generator.generate_monthly_report(args.month, args.year)
        print(f"Monthly report generated: {report_file}")
    
    print("=" * 80)