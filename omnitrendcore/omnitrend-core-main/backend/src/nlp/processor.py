import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import spacy
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
import json

class MultilingualNLPProcessor:
    """Processes news articles with multilingual NLP capabilities"""
    
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
        
        # Load NER model (multilingual)
        print("Loading NER model...")
        try:
            # Try a more stable NER model
            self.ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            print(f"Warning: Could not load NER model: {e}")
            print("NER will be disabled")
            self.ner_pipeline = None
        
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
            normalized_score = float((stars - 1) / 4)  # Normalize to 0-1 range
            
            return {
                'stars': int(stars),
                'score': normalized_score,
                'label': 'positive' if stars > 3 else 'negative' if stars < 3 else 'neutral'
            }
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {'stars': 3, 'score': 0.5, 'label': 'neutral'}
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text"""
        try:
            entities = self.ner_pipeline(text[:512])  # Limit text length
            return [
                {
                    'text': ent['word'],
                    'type': ent['entity_group'],
                    'score': float(ent['score'])
                }
                for ent in entities
            ]
        except Exception as e:
            print(f"NER error: {e}")
            return []
    
    def extract_topics(self, text: str, entities: List[Dict]) -> List[str]:
        """Extract main topics from text based on entities"""
        topics = set()
        
        # Add entity types as topics
        for ent in entities:
            if ent['score'] > 0.7:  # High confidence entities
                topics.add(ent['type'].lower())
                topics.add(ent['text'].lower())
        
        # Common topic keywords (multilingual support)
        topic_keywords = {
            'economy': ['ekonomi', 'economy', 'financial', 'finans', 'bank', 'banka'],
            'politics': ['politik', 'politics', 'government', 'hükümet', 'seçim', 'election'],
            'technology': ['teknoloji', 'technology', 'dijital', 'digital', 'internet'],
            'sports': ['spor', 'sports', 'futbol', 'football', 'basketball'],
            'health': ['sağlık', 'health', 'medical', 'tıp', 'hastane', 'hospital']
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
        
        # Extract entities
        entities = self.extract_entities(text)
        
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
    
    # Load raw articles
    # with open('data/raw/news.json', 'r', encoding='utf-8') as f:
    #     articles = json.load(f)
    
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
