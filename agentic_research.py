"""
Market Research Agent - Mistral-7B via HuggingFace Inference Providers
Specialized agent for web search analysis and salary insights using Together AI
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from huggingface_hub import InferenceClient
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import json


@dataclass
class ResearchReport:
    """Structured research output"""
    salary_overview: str
    market_insights: str
    hiring_recommendations: str
    sources: List[str]
    raw_data: str


class Colors:
    """Pastel color scheme for terminal output"""
    RESET = '\033[0m'
    BLUE = '\033[38;5;117m'
    GREEN = '\033[38;5;114m'
    YELLOW = '\033[38;5;186m'
    RED = '\033[38;5;181m'
    PURPLE = '\033[38;5;183m'
    CYAN = '\033[38;5;152m'
    ORANGE = '\033[38;5;180m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


class MarketResearchAgent:
    """
    Mistral-7B-powered Market Research Agent via Inference Providers
    
    Specialized for:
    - Web search analysis using Tavily
    - Salary and compensation research
    - Market trend analysis
    - Hiring insights generation
    """
    
    def __init__(
        self,
        huggingface_api_key: str,
        tavily_api_key: str,
        model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        temperature: float = 0.3
    ):
        """Initialize Mistral-7B research agent via Inference Providers"""
        print(f"{Colors.CYAN}[Research Agent] Initializing Mistral-7B via Together AI...{Colors.RESET}")
        
        self.client = InferenceClient(token=huggingface_api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = 2048
        
        # Initialize Tavily search
        search_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_api_key)
        self.web_search = TavilySearchResults(
            api_wrapper=search_wrapper,
            max_results=5,
            search_depth="advanced",  # Use advanced for deeper research
            include_answer=True,
            include_raw_content=True
        )
        
        print(f"{Colors.GREEN}✓ Research agent ready (using Inference Providers){Colors.RESET}\n")
    
    def conduct_research(
        self,
        query: str,
        context: str = None
    ) -> ResearchReport:
        """
        Conduct comprehensive market research
        
        Args:
            query: Research query (e.g., "salary for senior ML engineers")
            context: Optional context from conversation history
        
        Returns:
            ResearchReport with structured insights
        """
        try:
            print(f"\n{Colors.CYAN}{'─'*100}{Colors.RESET}")
            print(f"{Colors.BOLD}MARKET RESEARCH & SALARY ANALYSIS{Colors.RESET}\n")
            
            # Step 1: Enhance query with context
            enhanced_query = self._enhance_query(query, context)
            print(f"{Colors.DIM}Enhanced query: {enhanced_query[:100]}...{Colors.RESET}")
            
            # Step 2: Web search
            print(f"{Colors.DIM}Searching web for current market data...{Colors.RESET}")
            search_results = self.web_search.run(enhanced_query)
            
            # Extract sources
            sources = self._extract_sources(search_results)
            print(f"{Colors.GREEN}✓ Found {len(sources)} sources{Colors.RESET}\n")
            
            # Step 3: Analyze with Mistral-7B
            print(f"{Colors.DIM}Mistral-7B analyzing search results...{Colors.RESET}")
            report = self._analyze_results(query, search_results)
            
            print(f"{Colors.GREEN}✓ Analysis complete{Colors.RESET}")
            print(f"{Colors.CYAN}{'─'*100}{Colors.RESET}\n")
            
            return ResearchReport(
                salary_overview=report.get("salary_overview", ""),
                market_insights=report.get("market_insights", ""),
                hiring_recommendations=report.get("hiring_recommendations", ""),
                sources=sources,
                raw_data=str(search_results)
            )
            
        except Exception as e:
            print(f"{Colors.RED}Research error: {str(e)}{Colors.RESET}")
            return ResearchReport(
                salary_overview=f"Error retrieving salary data: {str(e)}",
                market_insights="Unable to complete market analysis",
                hiring_recommendations="Please try again with a more specific query",
                sources=[],
                raw_data=""
            )
    
    def _enhance_query(self, query: str, context: str = None) -> str:
        """Enhance query with context and specificity"""
        if context and ("this role" in query.lower() or "this position" in query.lower()):
            return f"{query} for {context[:200]}"
        
        # Add year for recency
        if "2025" not in query and "2024" not in query:
            return f"{query} 2025"
        
        return query
    
    def _extract_sources(self, search_results) -> List[str]:
        """Extract URLs from search results"""
        sources = []
        
        if isinstance(search_results, list):
            for item in search_results:
                if isinstance(item, dict) and 'url' in item:
                    sources.append(item['url'])
        elif isinstance(search_results, str):
            # Try to parse JSON string
            try:
                data = json.loads(search_results)
                if isinstance(data, list):
                    sources = [item.get('url') for item in data if 'url' in item]
            except:
                pass
        
        return sources
    
    def _analyze_results(self, query: str, search_results) -> Dict[str, str]:
        """Use Mistral-7B to analyze search results and generate structured report"""
        
        messages = [
            {
                "role": "user",
                "content": f"""You are an expert HR market analyst specializing in compensation research and hiring trends.

Analyze the web search results below and provide a structured market report in JSON format.

QUERY: {query}

SEARCH RESULTS:
{str(search_results)[:4000]}

Provide your analysis in VALID JSON format ONLY (no markdown, no other text):
{{
    "salary_data": [
        {{"level": "Junior", "experience": "1-3 years", "salary_range": "$80K - $100K", "location": "United States"}},
        {{"level": "Mid-Level", "experience": "4-6 years", "salary_range": "$100K - $140K", "location": "United States"}},
        {{"level": "Senior", "experience": "7-10 years", "salary_range": "$140K - $180K", "location": "United States"}},
        {{"level": "Lead/Staff", "experience": "10+ years", "salary_range": "$180K - $250K", "location": "United States"}}
    ],
    "market_insights": "• Insight 1 about demand trends\\n• Insight 2 about in-demand skills\\n• Insight 3 about market conditions\\n• Insight 4 about remote work impact",
    "hiring_recommendations": "1. Recommendation for competitive positioning\\n2. Budget planning considerations\\n3. Key factors for attracting talent\\n4. Timing and market conditions"
}}

CRITICAL REQUIREMENTS FOR SALARY DATA:
- Extract specific salary ranges by experience level (Junior, Mid-Level, Senior, Lead/Staff)
- Use consistent currency format: $XXK or $XXXK
- Include years of experience for each level
- Add geographic variations if mentioned in results (US, Europe, etc.)
- If multiple locations found, create separate entries
- Focus on 2024-2025 data only

CRITICAL REQUIREMENTS FOR INSIGHTS:
- Use bullet points (•) in market_insights
- Keep each insight concise (1-2 sentences)
- Focus on actionable information
- Include demand trends, skills, and growth indicators

CRITICAL REQUIREMENTS FOR RECOMMENDATIONS:
- Use numbered format (1., 2., 3.)
- Be specific and actionable
- Include budget guidance and hiring strategy"""
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            analysis = self._parse_json_response(content)
            
            if not analysis:
                # Fallback to text parsing
                return self._fallback_analysis(content)
            
            # Format salary data into a table
            if "salary_data" in analysis:
                analysis["salary_overview"] = self._format_salary_table(analysis["salary_data"])
                del analysis["salary_data"]
            else:
                analysis["salary_overview"] = "No salary data available in search results"
            
            return analysis
            
        except Exception as e:
            print(f"{Colors.YELLOW}⚠️  Mistral-7B API error: {str(e)[:150]}{Colors.RESET}")
            return {
                "salary_overview": "Unable to retrieve salary data due to API error",
                "market_insights": "Market analysis unavailable - API connection failed",
                "hiring_recommendations": "Please try again or refine your query"
            }
    
    def _parse_json_response(self, response: str) -> Dict[str, str]:
        """Parse JSON from Mistral-7B response"""
        try:
            # Find JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            
            return {}
        except:
            return {}
    
    def _format_salary_table(self, salary_data: List[Dict[str, str]]) -> str:
        """Format salary data into a beautiful ASCII table"""
        if not salary_data:
            return "No salary data available"
        
        # Table structure
        table = []
        
        # Header
        table.append("┌─────────────────────┬─────────────────────┬─────────────────────┬─────────────────────┐")
        table.append("│ LEVEL               │ EXPERIENCE          │ SALARY RANGE        │ LOCATION            │")
        table.append("├─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤")
        
        # Data rows
        for entry in salary_data:
            level = entry.get("level", "N/A")[:19].ljust(19)
            experience = entry.get("experience", "N/A")[:19].ljust(19)
            salary = entry.get("salary_range", "N/A")[:19].ljust(19)
            location = entry.get("location", "N/A")[:19].ljust(19)
            
            table.append(f"│ {level} │ {experience} │ {salary} │ {location} │")
        
        # Footer
        table.append("└─────────────────────┴─────────────────────┴─────────────────────┴─────────────────────┘")
        
        return "\n".join(table)
    
    def _fallback_analysis(self, content: str) -> Dict[str, str]:
        """Fallback text parsing if JSON fails"""
        return {
            "salary_overview": self._extract_section(content, "salary", "market"),
            "market_insights": self._extract_section(content, "market", "hiring"),
            "hiring_recommendations": self._extract_section(content, "hiring", "end")
        }
    
    def _extract_section(self, text: str, start_marker: str, end_marker: str) -> str:
        """Extract section from text based on markers"""
        text_lower = text.lower()
        start_pos = text_lower.find(start_marker)
        end_pos = text_lower.find(end_marker, start_pos) if end_marker != "end" else len(text)
        
        if start_pos != -1:
            section = text[start_pos:end_pos].strip()
            return section[:500] if len(section) > 500 else section
        
        return "Analysis unavailable"
    
    def format_report(self, report: ResearchReport) -> str:
        """Format research report with beautiful styling"""
        
        try:
            output = []
            
            # Header
            output.append(f"\n{Colors.BLUE}{'='*100}{Colors.RESET}")
            output.append(f"{Colors.BOLD}{Colors.PURPLE}MARKET RESEARCH REPORT{Colors.RESET}")
            output.append(f"{Colors.BLUE}{'='*100}{Colors.RESET}\n")
            
            # Salary Overview
            output.append(f"{Colors.BOLD}{Colors.CYAN}SALARY OVERVIEW{Colors.RESET}")
            output.append(f"{Colors.CYAN}{'─' * 100}{Colors.RESET}")
            
            # Check if it's a table format
            salary_text = self._format_multiline(report.salary_overview)
            if '┌' in salary_text or '│' in salary_text:
                # It's a table - add padding for centering
                table_lines = salary_text.split('\n')
                for line in table_lines:
                    output.append(f"  {line}")
            else:
                output.append(salary_text)
            
            output.append("")
            
            # Market Insights
            output.append(f"\n{Colors.BOLD}{Colors.GREEN}MARKET INSIGHTS{Colors.RESET}")
            output.append(f"{Colors.GREEN}{'─' * 60}{Colors.RESET}")
            output.append(self._format_multiline(report.market_insights))
            output.append("")
            
            # Hiring Recommendations
            output.append(f"\n{Colors.BOLD}{Colors.ORANGE}HIRING RECOMMENDATIONS{Colors.RESET}")
            output.append(f"{Colors.ORANGE}{'─' * 60}{Colors.RESET}")
            output.append(self._format_multiline(report.hiring_recommendations))
            output.append("")
            
            # Sources
            if report.sources:
                output.append(f"\n{Colors.PURPLE}{'='*100}{Colors.RESET}")
                output.append(f"{Colors.BOLD}{Colors.CYAN}SOURCES & REFERENCES{Colors.RESET}")
                output.append(f"{Colors.PURPLE}{'='*100}{Colors.RESET}\n")
                for idx, url in enumerate(report.sources, 1):
                    output.append(f"  {Colors.BOLD}[{idx}]{Colors.RESET} {Colors.CYAN}{url}{Colors.RESET}")
            
            output.append(f"\n{Colors.BLUE}{'='*100}{Colors.RESET}\n")
            
            return "\n".join(output)
            
        except Exception as e:
            # Fallback formatting if there's any error
            return f"\n{Colors.RED}Error formatting report: {str(e)}{Colors.RESET}\n\nRaw data:\n{str(report)}\n"
    
    def _format_multiline(self, text) -> str:
        """Format text with proper indentation and bullet points"""
        if not text:
            return "  No data available"
        
        # Handle list inputs (convert to string)
        if isinstance(text, list):
            text = '\n'.join(str(item) for item in text)
        
        # Convert to string if not already
        text = str(text)
        
        # Check if text contains table characters (don't add indentation for tables)
        is_table = any(char in text for char in ['┌', '│', '└', '├', '┤', '┬', '┴', '┼'])
        
        lines = text.split('\n')
        formatted = []
        
        for line in lines:
            # For tables, don't strip or add indentation
            if is_table and any(char in line for char in ['┌', '│', '└', '├', '┤', '┬', '┴', '┼']):
                formatted.append(line)
                continue
            
            line = line.strip()
            if not line:
                continue
            
            # Format bullet points
            if line.startswith('-') or line.startswith('•'):
                formatted.append(f"  {Colors.YELLOW}•{Colors.RESET} {line[1:].strip()}")
            # Format numbered lists
            elif len(line) > 2 and line[0].isdigit() and line[1] in '.):':
                formatted.append(f"  {Colors.BOLD}{line[:2]}{Colors.RESET} {line[2:].strip()}")
            # Format headers
            elif line.endswith(':') and len(line) < 60:
                formatted.append(f"\n  {Colors.BOLD}{line}{Colors.RESET}")
            # Regular text
            else:
                formatted.append(f"  {line}")
        
        return '\n'.join(formatted) if formatted else "  No data available"


def create_research_agent(
    huggingface_api_key: str,
    tavily_api_key: str,
    model: str = "mistralai/Mistral-7B-Instruct-v0.3"
) -> MarketResearchAgent:
    """Factory function to create research agent instance"""
    return MarketResearchAgent(
        huggingface_api_key=huggingface_api_key,
        tavily_api_key=tavily_api_key,
        model=model
    )


# Test the agent
if __name__ == "__main__":
    print(f"\n{Colors.CYAN}{'='*100}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.PURPLE}MISTRAL-7B MARKET RESEARCH AGENT - TEST MODE{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*100}{Colors.RESET}\n")
    
    # Initialize agent
    agent = create_research_agent(
        huggingface_api_key="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",
        tavily_api_key="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"
    )
    
    # Test query
    test_query = "salary for senior machine learning engineers with 5+ years experience"
    print(f"{Colors.BOLD}Test Query:{Colors.RESET} {test_query}\n")
    
    # Conduct research
    report = agent.conduct_research(test_query)
    
    # Display formatted report
    print(agent.format_report(report))
    
    print(f"{Colors.GREEN}✓ Test complete{Colors.RESET}")