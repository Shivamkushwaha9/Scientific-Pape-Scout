import asyncio
import json
import logging
import os
from mcp.server import Server
from mcp.types import Tool, TextContent
from .pdf_processor import PDFProcessor
from agent.llm_providers import LLMProviderFactory

logger = logging.getLogger(__name__)

class PDFSummarizeServer:
    def __init__(self):
        self.server = Server("pdf-summarize")
        self.pdf_processor = PDFProcessor()
        self.setup_llm_provider()
        self.setup_tools()
    
    def setup_llm_provider(self):
        """Initialize the LLM provider for summarization"""
        provider_name = os.getenv("LLM_PROVIDER", "openai")
        model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        
        api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY")
        }
        
        api_key = api_keys.get(provider_name)
        if not api_key:
            raise ValueError(f"API key for {provider_name} not found in environment")
        
        self.llm_provider = LLMProviderFactory.create_provider(provider_name, api_key, model)
    
    def setup_tools(self):
        """Register the PDF summarization tool"""
        @self.server.list_tools()
        async def list_tools():
            return [
                Tool(
                    name="summarize_pdf",
                    description="Download and summarize a PDF document",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "pdf_url": {
                                "type": "string",
                                "description": "URL of the PDF to summarize"
                            },
                            "max_length": {
                                "type": "integer",
                                "description": "Maximum length of summary in words",
                                "default": 200,
                                "minimum": 50,
                                "maximum": 1000
                            }
                        },
                        "required": ["pdf_url"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict):
            if name == "summarize_pdf":
                try:
                    pdf_url = arguments.get("pdf_url")
                    max_length = arguments.get("max_length", 200)
                    
                    logger.info(f"Summarizing PDF: {pdf_url}")
                    
                    # Extract text from PDF
                    text = await self.pdf_processor.extract_text_from_url(pdf_url)
                    if not text:
                        raise Exception("Failed to extract text from PDF")
                    
                    # Truncate text if too long (to avoid token limits)
                    if len(text) > 10000:
                        text = text[:10000] + "..."
                    
                    # Generate summary
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
                    
                    response = await self.llm_provider.generate(messages, max_tokens=max_length * 2)
                    
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps({
                                "success": True,
                                "summary": response.content,
                                "model_used": response.model,
                                "usage": response.usage
                            }, indent=2)
                        )
                    ]
                    
                except Exception as e:
                    logger.error(f"Error in summarize_pdf: {e}")
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
    
    async def run(self, port: int = 8002):
        """Run the MCP server"""
        logger.info(f"Starting PDF Summarize MCP server on port {port}")
        await self.server.run()

if __name__ == "__main__":
    server = PDFSummarizeServer()
    port = int(os.getenv("PDF_SUMMARIZE_PORT", 8002))
    asyncio.run(server.run(port))