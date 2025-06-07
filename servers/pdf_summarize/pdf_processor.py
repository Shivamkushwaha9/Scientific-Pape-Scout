import requests
from PyPDF2 import PdfReader
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Scientific Paper Scout Bot)'
        })
    
    async def extract_text_from_url(self, pdf_url: str) -> Optional[str]:
        """Download PDF from URL and extract text"""
        try:
            logger.info(f"Downloading PDF from: {pdf_url}")
            
            # Download PDF
            response = self.session.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            # Extract text
            pdf_file = io.BytesIO(response.content)
            reader = PdfReader(pdf_file)
            
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_url}: {e}")
            return None

