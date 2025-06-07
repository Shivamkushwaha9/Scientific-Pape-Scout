import asyncio
import json
import logging
from typing import Dict, Any, List
import subprocess
import sys
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self):
        self.servers = {}
        self.call_log = []
    
    async def start_server(self, name: str, server_path: str, port: int):
        """Start an MCP server subprocess"""
        try:
            logger.info(f"Starting MCP server: {name}")
            # For this implementation, we'll import the servers directly
            # In a production MCP setup, this would use proper MCP protocol
            self.servers[name] = {
                "status": "running",
                "port": port
            }
            logger.info(f"MCP server {name} started on port {port}")
            
        except Exception as e:
            logger.error(f"Failed to start MCP server {name}: {e}")
            raise
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on an MCP server"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Calling tool {tool_name} on server {server_name} with args: {arguments}")
            
            # Direct tool calls - in production this would use MCP protocol
            if server_name == "paper_search" and tool_name == "search_papers":
                from servers.paper_search.arxiv_client import ArxivClient
                client = ArxivClient()
                query = arguments.get("query", "")
                max_results = arguments.get("max_results", 10)
                
                results = await client.search_papers(query, max_results)
                response_data = {
                    "success": True,
                    "results": results,
                    "count": len(results)
                }
                
            elif server_name == "pdf_summarize" and tool_name == "summarize_pdf":
                from servers.pdf_summarize.pdf_processor import PDFProcessor
                from agent.llm_providers import LLMProviderFactory
                
                processor = PDFProcessor()
                pdf_url = arguments.get("pdf_url")
                max_length = arguments.get("max_length", 200)
                
                # Extract text
                text = await processor.extract_text_from_url(pdf_url)
                if not text:
                    raise Exception("Failed to extract text from PDF")
                
                # Truncate if too long
                if len(text) > 10000:
                    text = text[:10000] + "..."
                
                # Setup LLM for summarization
                provider_name = os.getenv("LLM_PROVIDER", "openai")
                model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
                
                api_keys = {
                    "openai": os.getenv("OPENAI_API_KEY"),
                    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
                    "gemini": os.getenv("GEMINI_API_KEY")
                }
                
                api_key = api_keys.get(provider_name)
                if not api_key:
                    raise ValueError(f"API key for {provider_name} not found")
                
                llm_provider = LLMProviderFactory.create_provider(provider_name, api_key, model)
                
                messages = [
                    {
                        "role": "system",
                        "content": f"You are a research assistant. Summarize the following academic paper in approximately {max_length} words. Focus on the main contributions, methodology, and key findings."
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize this paper:\n\n{text}"
                    }
                ]
                
                llm_response = await llm_provider.generate(messages, max_tokens=max_length * 2)
                
                response_data = {
                    "success": True,
                    "summary": llm_response.content,
                    "model_used": llm_response.model,
                    "usage": llm_response.usage
                }
                
            else:
                raise ValueError(f"Unknown server/tool: {server_name}/{tool_name}")
            
            # Log the call
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds()
            
            self.call_log.append({
                "timestamp": start_time.isoformat(),
                "server": server_name,
                "tool": tool_name,
                "arguments": arguments,
                "latency_seconds": latency,
                "success": response_data.get("success", False)
            })
            
            logger.info(f"Tool call completed in {latency:.2f}s")
            return response_data
            
        except Exception as e:
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds()
            
            self.call_log.append({
                "timestamp": start_time.isoformat(),
                "server": server_name,
                "tool": tool_name,
                "arguments": arguments,
                "latency_seconds": latency,
                "success": False,
                "error": str(e)
            })
            
            logger.error(f"Tool call failed after {latency:.2f}s: {e}")
            raise
    
    def get_call_log(self) -> List[Dict[str, Any]]:
        """Get the log of all tool calls"""
        return self.call_log.copy()
    
    async def shutdown(self):
        """Shutdown all MCP servers"""
        for name, server_info in self.servers.items():
            try:
                logger.info(f"Shutdown MCP server: {name}")
            except Exception as e:
                logger.error(f"Error shutting down server {name}: {e}")