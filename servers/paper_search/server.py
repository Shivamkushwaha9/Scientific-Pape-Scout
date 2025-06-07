import asyncio
import json
import logging
from mcp.server import Server
from mcp.types import Tool, TextContent
from .arxiv_client import ArxivClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperSearchServer:
    def __init__(self):
        self.server = Server("paper-search")
        self.arxiv_client = ArxivClient()
        self.setup_tools()
    
    def setup_tools(self):
        """Register the paper search tool"""
        @self.server.list_tools()
        async def list_tools():
            return [
                Tool(
                    name="search_papers",
                    description="Search for academic papers on ArXiv",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for papers"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 50
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict):
            if name == "search_papers":
                try:
                    query = arguments.get("query")
                    max_results = arguments.get("max_results", 10)
                    
                    logger.info(f"Searching papers: query='{query}', max_results={max_results}")
                    
                    results = await self.arxiv_client.search_papers(query, max_results)
                    
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps({
                                "success": True,
                                "results": results,
                                "count": len(results)
                            }, indent=2)
                        )
                    ]
                    
                except Exception as e:
                    logger.error(f"Error in search_papers: {e}")
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps({
                                "success": False,
                                "error": str(e)
                            })
                        )
                    ]
            else:
                raise ValueError(f"Unknown tool: {name}")
    
    async def run(self, port: int = 8001):
        """Run the MCP server"""
        logger.info(f"Starting Paper Search MCP server on port {port}")
        # This would typically use stdio transport in a real MCP setup
        # For this example, we'll use a simple async server
        await self.server.run()

if __name__ == "__main__":
    import os
    server = PaperSearchServer()
    port = int(os.getenv("PAPER_SEARCH_PORT", 8001))
    asyncio.run(server.run(port))