import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import List, Dict
import numpy as np
from pathlib import Path
import json
import re

class MultilingualNLPProcessor:
    """Lightweight NLP processor without heavy dependencies"""
    
    def __init__(self, device: str = None):
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load multilingual sentence transformer for embeddings
        print("Loading sentence transformer...")
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.embedder.to(self.device)
        
        # Load sentiment analysis pipeline (multilingual)
        print("Loading sentiment analyzer...")
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=0 if self.device == "cuda" else -1
        )
        
        print("NLP Processor initialized (lightweight mode - no NER)")
        
    def extract_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text using multilingual model"""
        embeddings = self.embedder.encode(
            texts,
            convert_to_tensor=False,
            show_progress_bar=True
        )
        return embeddings
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text (returns 1-5 star rating)"""
        try:
            result = self.sentiment_analyzer(text[:512])[0]  # Limit text length
            # Convert to normalized score
            stars = int(result['label'].split()[0])
            normalized_score = (stars - 1) / 4  # Normalize to 0-1 range
            
            return {
                'stars': stars,
                'score': normalized_score,
                'label': 'positive' if stars > 3 else 'negative' if stars < 3 else 'neutral'
            }
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {'stars': 3, 'score': 0.5, 'label': 'neutral'}
    
    def extract_entities_simple(self, text: str) -> List[Dict]:
        """Simple rule-based entity extraction"""
        entities = []
        
        # Common Turkish/English patterns for organizations
        org_patterns = [
            r'\b[A-ZÇĞIÖŞÜ][a-zçğıöşü]+\s(?:Bankası|Bank|Bakanlığı|Ministry|Şirketi|Company|Corporation)\b',
            r'\b(?:Merkez|Central)\s(?:Bankası|Bank)\b',
            r'\bT\.C\.\s[A-ZÇĞIÖŞÜ][a-zçğıöşü\s]+Bakanlığı\b'
        ]
        
        # Common patterns for locations
        loc_patterns = [
            r'\b(?:İstanbul|Ankara|İzmir|Antalya|Bursa|London|Paris|New York|Tokyo)\b',
            r'\bTürkiye\b',
            r'\b[A-ZÇĞIÖŞÜ][a-zçğıöşü]+\s(?:Üniversitesi|University)\b'
        ]
        
        # Person patterns (simplified)
        person_patterns = [
            r'\b[A-ZÇĞIÖŞÜ][a-zçğıöşü]+\s[A-ZÇĞIÖŞÜ][a-zçğıöşü]+\b'
        ]
        
        # Extract organizations
        for pattern in org_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'type': 'ORG',
                    'score': 0.9
                })
        
        # Extract locations
        for pattern in loc_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'type': 'LOC',
                    'score': 0.85
                })
        
        return entities[:10]  # Limit to top 10
    
    def extract_topics(self, text: str, entities: List[Dict]) -> List[str]:
        """Extract main topics from text based on entities and keywords"""
        topics = set()
        
        # Add entity types as topics
        for ent in entities:
            if ent['score'] > 0.7:
                topics.add(ent['type'].lower())
        
        # Common topic keywords (multilingual support)
        topic_keywords = {
            'economy': ['ekonomi', 'economy', 'financial', 'finans', 'bank', 'banka', 'dolar', 'dollar', 'enflasyon', 'inflation', 'borsa', 'stock'],
            'politics': ['politik', 'politics', 'government', 'hükümet', 'seçim', 'election', 'bakanlı', 'ministry'],
            'technology': ['teknoloji', 'technology', 'dijital', 'digital', 'internet', 'yapay zeka', 'artificial intelligence', 'ai', 'siber', 'cyber'],
            'sports': ['spor', 'sports', 'futbol', 'football', 'basketball', 'maç', 'match', 'takım', 'team'],
            'health': ['sağlık', 'health', 'medical', 'tıp', 'hastane', 'hospital', 'doktor', 'doctor'],
            'energy': ['enerji', 'energy', 'elektrik', 'electric', 'güneş', 'solar', 'rüzgar', 'wind'],
            'education': ['eğitim', 'education', 'okul', 'school', 'üniversite', 'university', 'öğrenci', 'student'],
            'transportation': ['ulaşım', 'transportation', 'metro', 'otobüs', 'bus', 'trafik', 'traffic'],
            'environment': ['çevre', 'environment', 'iklim', 'climate', 'karbon', 'carbon'],
            'finance': ['finans', 'finance', 'yatırım', 'investment', 'kredi', 'credit', 'fintech']
        }
        
        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.add(topic)
        
        return list(topics)
    
    def process_article(self, article: Dict) -> Dict:
        """Process a single article with full NLP pipeline"""
        text = article.get('title', '') + ' ' + article.get('content', '')
        
        # Generate embedding
        embedding = self.extract_embeddings([text])[0]
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(text)
        
        # Extract entities (simple version)
        entities = self.extract_entities_simple(text)
        
        # Extract topics
        topics = self.extract_topics(text, entities)
        
        return {
            'id': article.get('id'),
            'title': article.get('title'),
            'content': article.get('content'),
            'source': article.get('source'),
            'published_at': article.get('published_at'),
            'url': article.get('url'),
            'embedding': embedding.tolist(),
            'sentiment': sentiment,
            'entities': entities,
            'topics': topics
        }
    
    def process_batch(self, articles: List[Dict], output_path: str) -> List[Dict]:
        """Process a batch of articles"""
        processed = []
        
        for idx, article in enumerate(articles):
            print(f"Processing article {idx + 1}/{len(articles)}")
            try:
                processed_article = self.process_article(article)
                processed.append(processed_article)
            except Exception as e:
                print(f"Error processing article {article.get('id')}: {e}")
        
        # Save processed data
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(processed)} processed articles to {output_path}")
        return processed

# Example usage
if __name__ == "__main__":
    processor = MultilingualNLPProcessor()
    
    # Sample article for testing
    sample_articles = [
        {
            "id": 1,
            "title": "Merkez Bankası faiz kararını açıkladı",
            "content": "Türkiye Cumhuriyet Merkez Bankası bugün politika faizinde değişiklik yapmadı.",
            "source": "Example News",
            "published_at": "2025-11-16T10:00:00",
            "url": "https://example.com/news1"
        }
    ]
    
    processed = processor.process_batch(sample_articles, 'data/processed/processed_news.json')
