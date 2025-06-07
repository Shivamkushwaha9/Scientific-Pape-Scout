import asyncio
import json
import logging
import os
from typing import Dict, Any, List, AsyncGenerator
from dotenv import load_dotenv
from .llm_providers import LLMProviderFactory
from .mcp_client import MCPClient

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class ScientificPaperAgent:
    def __init__(self):
        self.setup_llm_provider()
        self.mcp_client = MCPClient()
        self.conversation_history = []
        
    def setup_llm_provider(self):
        """Initialize the LLM provider"""
        provider_name = os.getenv("LLM_PROVIDER", "openai")
        model = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
        
        api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY")
        }
        
        api_key = api_keys.get(provider_name)
        if not api_key:
            raise ValueError(f"API key for {provider_name} not found in environment")
        
        self.llm_provider = LLMProviderFactory.create_provider(provider_name, api_key, model)
        logger.info(f"Initialized LLM provider: {provider_name} with model {model}")
    
    async def initialize(self):
        """Initialize the agent and start MCP servers"""
        try:
            # Start MCP servers
            await self.mcp_client.start_server(
                "paper_search", 
                "servers/paper_search/server.py",
                int(os.getenv("PAPER_SEARCH_PORT", 8001))
            )
            
            await self.mcp_client.start_server(
                "pdf_summarize",
                "servers/pdf_summarize/server.py", 
                int(os.getenv("PDF_SUMMARIZE_PORT", 8002))
            )
            
            logger.info("Scientific Paper Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    async def process_message(self, user_message: str) -> AsyncGenerator[str, None]:
        """Process a user message and stream the response"""
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Create system prompt with available tools
            system_prompt = self._create_system_prompt()
            
            # Prepare messages for LLM
            messages = [{"role": "system", "content": system_prompt}] + self.conversation_history
            
            # Generate response with tool calling capability
            async for chunk in self._generate_with_tools(messages):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            yield f"Error: {str(e)}"
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt with tool descriptions"""
        return """You are a Scientific Paper Scout, an AI assistant specialized in discovering and summarizing recent research papers. You have access to the following tools:

1. search_papers(query: str, max_results: int = 10) - Search for academic papers on ArXiv
2. summarize_pdf(pdf_url: str, max_length: int = 200) - Download and summarize a PDF document

When a user asks about research papers:
1. Use search_papers to find relevant papers
2. Present the results in a clear, organized way
3. If the user wants more details, use summarize_pdf to provide summaries
4. Always cite your sources and provide paper URLs

Be helpful, accurate, and focused on scientific research. Explain complex concepts clearly."""
    
    # agent/host.py - Updated _generate_with_tools method
    async def _generate_with_tools(self, messages: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
        """Generate response with tool calling capability"""
        # First, check if we need to call any tools
        tool_calls = await self._determine_tool_calls(messages)
        
        if tool_calls:
            # Execute tool calls
            tool_results = []
            for tool_call in tool_calls:
                yield f"\n🔧 Calling {tool_call['server']}.{tool_call['tool']} with {tool_call['args']}\n"
                
                try:
                    result = await self.mcp_client.call_tool(
                        tool_call['server'],
                        tool_call['tool'], 
                        tool_call['args']
                    )
                    tool_results.append({
                        "tool": tool_call['tool'],
                        "result": result
                    })
                    
                    yield f"✅ Tool call completed successfully\n"
                    
                except Exception as e:
                    yield f"❌ Tool call failed: {str(e)}\n"
                    tool_results.append({
                        "tool": tool_call['tool'],
                        "result": {"success": False, "error": str(e)}
                    })
            
            # Add tool results to context and generate response
            if tool_results:
                tool_context = self._format_tool_results(tool_results)
                
                # Create new messages with tool context
                context_messages = messages + [{
                    "role": "system", 
                    "content": f"Here are the results from the tools you called:\n\n{tool_context}\n\nNow provide a helpful response based on these results."
                }]
                
                yield f"\n📝 Analyzing results and generating response...\n\n"
                
                # Generate response with tool context
                response_content = ""
                async for chunk in self.llm_provider.stream_generate(context_messages):
                    yield chunk
                    response_content += chunk
                
                # Add to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_content
                })
        else:
            # No tools needed, generate direct response
            yield f"\n📝 Generating response...\n\n"
            
            response_content = ""
            async for chunk in self.llm_provider.stream_generate(messages):
                yield chunk
                response_content += chunk
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response_content
            })

    async def _determine_tool_calls(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Determine what tools to call based on the conversation"""
        user_message = messages[-1]["content"].lower()
        tool_calls = []
        
        # Enhanced keyword detection
        search_keywords = ["search", "find", "papers", "research", "arxiv", "quantum", "machine learning", "ai", "deep learning"]
        summarize_keywords = ["summarize", "summary", "abstract", "explain"]
        
        if any(keyword in user_message for keyword in search_keywords):
            # Extract search query more intelligently
            query = self._extract_search_query(messages[-1]["content"])
            if query:
                tool_calls.append({
                    "server": "paper_search",
                    "tool": "search_papers", 
                    "args": {"query": query, "max_results": 5}
                })
        
        if any(keyword in user_message for keyword in summarize_keywords):
            # Look for PDF URLs in the conversation
            pdf_urls = self._extract_pdf_urls(messages)
            for url in pdf_urls:
                tool_calls.append({
                    "server": "pdf_summarize",
                    "tool": "summarize_pdf",
                    "args": {"pdf_url": url, "max_length": 200}
                })
        
        return tool_calls

    def _extract_search_query(self, message: str) -> str:
        """Extract search query from user message"""  
        import re
        
        # Remove tool call syntax if present
        message = re.sub(r'search_papers\([^)]*\)', '', message)
        
        # Clean the message
        words = message.split()
        
        # Remove common words but keep important terms
        stop_words = {"search", "find", "for", "papers", "about", "on", "research", "in", "the", "a", "an"}
        query_words = [w for w in words if w.lower() not in stop_words and len(w) > 1]
        
        # Join and clean
        query = " ".join(query_words[:6])  # Limit to 6 words
        query = re.sub(r'[^\w\s]', '', query)  # Remove special characters
        
        return query.strip()
    
    def _extract_pdf_urls(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract PDF URLs from conversation"""
        urls = []
        for message in messages:
            content = message["content"]
            # Look for arxiv PDF URLs
            import re
            pdf_pattern = r'https?://[^\s]+\.pdf'
            found_urls = re.findall(pdf_pattern, content)
            urls.extend(found_urls)
        return urls
    
    def _format_tool_results(self, tool_results: List[Dict[str, Any]]) -> str:
        """Format tool results for the LLM context"""
        formatted = ""
        for result in tool_results:
            tool_name = result["tool"]
            data = result["result"]
            
            if data.get("success"):
                if tool_name == "search_papers":
                    papers = data.get("results", [])
                    formatted += f"\nFound {len(papers)} papers:\n"
                    for i, paper in enumerate(papers[:3], 1):  # Limit to top 3
                        formatted += f"{i}. {paper['title']}\n"
                        formatted += f"   Authors: {', '.join(paper['authors'][:3])}\n"
                        formatted += f"   URL: {paper['pdf_url']}\n"
                        formatted += f"   Summary: {paper['summary'][:200]}...\n\n"
                
                elif tool_name == "summarize_pdf":
                    summary = data.get("summary", "")
                    formatted += f"\nPDF Summary:\n{summary}\n\n"
            else:
                formatted += f"\nError with {tool_name}: {data.get('error', 'Unknown error')}\n"
        
        return formatted
    
    def get_tool_call_log(self) -> List[Dict[str, Any]]:
        """Get the tool call log"""
        return self.mcp_client.get_call_log()
    
    async def shutdown(self):
        """Shutdown the agent"""
        await self.mcp_client.shutdown()
        logger.info("Scientific Paper Agent shutdown complete")