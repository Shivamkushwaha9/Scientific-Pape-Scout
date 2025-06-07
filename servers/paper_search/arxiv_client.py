import arxiv
from typing import List, Dict, Any
import asyncio
import logging

logger = logging.getLogger(__name__)

class ArxivClient:
    def __init__(self):
        self.client = arxiv.Client()
    
    async def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for papers on ArXiv"""
        try:
            # Run the search in a thread pool to avoid blocking
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            results = []
            loop = asyncio.get_event_loop()
            
            # Fetch results asynchronously
            papers = await loop.run_in_executor(None, list, self.client.results(search))
            
            for paper in papers:
                result = {
                    "id": paper.entry_id,
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "summary": paper.summary,
                    "published": paper.published.isoformat(),
                    "updated": paper.updated.isoformat() if paper.updated else None,
                    "categories": paper.categories,
                    "pdf_url": paper.pdf_url,
                    "links": [{"href": link.href, "title": link.title} for link in paper.links]
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} papers for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
            raise