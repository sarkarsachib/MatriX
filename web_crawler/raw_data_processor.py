import re
import json
from bs4 import BeautifulSoup
from typing import List, Dict, Any

class RawDataProcessor:
    def __init__(self):
        self.processed_data = []
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove HTML entities
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        
        return text
    
    def extract_metadata(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from raw scraped data"""
        metadata = {
            'source_url': raw_data.get('url', ''),
            'timestamp': raw_data.get('timestamp', ''),
            'content_type': raw_data.get('content_type', 'text'),
            'language': self.detect_language(raw_data.get('content', '')),
            'word_count': len(raw_data.get('content', '').split()) if raw_data.get('content') else 0
        }
        return metadata
    
    def detect_language(self, text: str) -> str:
        """Simple language detection (can be enhanced with proper libraries)"""
        # Basic heuristic - can be replaced with proper language detection
        if not text:
            return 'unknown'
        
        # Simple English detection
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        words = text.lower().split()[:50]  # Check first 50 words
        english_count = sum(1 for word in words if word in english_words)
        
        if english_count > len(words) * 0.1:  # If more than 10% are common English words
            return 'english'
        else:
            return 'unknown'
    
    def process_batch(self, raw_data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of raw data"""
        processed_batch = []
        
        for item in raw_data_batch:
            processed_item = {
                'content': self.clean_text(item.get('content', '')),
                'metadata': self.extract_metadata(item),
                'processed_timestamp': item.get('timestamp', ''),
                'quality_score': self.calculate_quality_score(item)
            }
            processed_batch.append(processed_item)
        
        return processed_batch
    
    def calculate_quality_score(self, item: Dict[str, Any]) -> float:
        """Calculate quality score for the content (0.0 to 1.0)"""
        content = item.get('content', '')
        if not content:
            return 0.0
        
        score = 0.0
        
        # Length factor (prefer medium-length content)
        length = len(content)
        if 100 <= length <= 5000:
            score += 0.3
        elif 50 <= length < 100 or 5000 < length <= 10000:
            score += 0.2
        elif length > 10000:
            score += 0.1
        
        # Sentence structure (presence of punctuation)
        if re.search(r'[.!?]', content):
            score += 0.2
        
        # Word diversity (unique words / total words)
        words = content.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio * 0.3
        
        # Spam detection (repeated patterns)
        if not re.search(r'(.)\1{10,}', content):  # No character repeated 10+ times
            score += 0.2
        
        return min(score, 1.0)
    
    def save_processed_data(self, data: List[Dict[str, Any]], filename: str):
        """Save processed data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_processed_data(self, filename: str) -> List[Dict[str, Any]]:
        """Load processed data from JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

# Example usage
if __name__ == "__main__":
    processor = RawDataProcessor()
    
    # Sample raw data
    sample_data = [
        {
            'content': 'This is a sample text with some meaningful content. It has multiple sentences.',
            'url': 'https://example.com/page1',
            'timestamp': '2024-01-01T12:00:00Z',
            'content_type': 'text'
        },
        {
            'content': 'Another piece of content that needs processing and cleaning.',
            'url': 'https://example.com/page2',
            'timestamp': '2024-01-01T12:05:00Z',
            'content_type': 'text'
        }
    ]
    
    processed = processor.process_batch(sample_data)
    processor.save_processed_data(processed, 'processed_data.json')
    print("Sample data processed and saved.")

