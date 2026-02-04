import torch
import numpy as np
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from tqdm import tqdm

class NewsGraphBuilder:
    """Builds temporal graph from processed news articles"""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.graph = nx.DiGraph()
        self.node_features = {}
        self.edge_features = {}
        self.timestamps = {}
        
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    
    def add_article_node(self, article: Dict):
        """Add article as a node to the graph"""
        node_id = article['id']
        
        # Add node with metadata
        self.graph.add_node(
            node_id,
            title=article['title'],
            source=article['source'],
            published_at=article['published_at'],
            url=article.get('url', ''),
            topics=article.get('topics', []),
            sentiment_score=article['sentiment']['score'],
            sentiment_label=article['sentiment']['label']
        )
        
        # Store node features (embedding + sentiment)
        embedding = np.array(article['embedding'])
        sentiment_feat = np.array([article['sentiment']['score']])
        
        self.node_features[node_id] = np.concatenate([embedding, sentiment_feat])
        
        # Store timestamp
        try:
            timestamp = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
            self.timestamps[node_id] = timestamp.timestamp()
        except:
            self.timestamps[node_id] = datetime.now().timestamp()
    
    def add_similarity_edges(self, articles: List[Dict]):
        """Add edges between similar articles using vectorized computation"""
        print("Computing similarity edges...", flush=True)
        
        # Stack embeddings into matrix
        embeddings = np.array([art['embedding'] for art in articles])
        ids = [art['id'] for art in articles]
        n = len(articles)
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings_norm = embeddings / norms
        
        # Compute full similarity matrix in batches to manage memory
        batch_size = 1000
        edge_count = 0
        
        pbar = tqdm(range(0, n, batch_size), desc="Similarity edges", unit="batch", file=sys.stdout, dynamic_ncols=True)
        for i in pbar:
            end_i = min(i + batch_size, n)
            sim_batch = embeddings_norm[i:end_i] @ embeddings_norm.T
            
            for local_i, global_i in enumerate(range(i, end_i)):
                start_j = max(global_i + 1, 0)
                for j in range(start_j, n):
                    sim = sim_batch[local_i, j]
                    if sim >= self.similarity_threshold:
                        if self.timestamps[ids[global_i]] <= self.timestamps[ids[j]]:
                            src, tgt = ids[global_i], ids[j]
                        else:
                            src, tgt = ids[j], ids[global_i]
                        
                        self.graph.add_edge(src, tgt)
                        self.edge_features[(src, tgt)] = {
                            'similarity': float(sim),
                            'time_diff': abs(self.timestamps[src] - self.timestamps[tgt])
                        }
                        edge_count += 1
            pbar.set_postfix(edges=edge_count)
        print(f"Added {edge_count} similarity edges", flush=True)
    
    def add_topic_edges(self, articles: List[Dict], max_edges_per_article: int = 10):
        """Add edges between articles sharing topics (limited to prevent memory explosion)"""
        print("Computing topic edges...", flush=True)
        
        # Group articles by topic
        topic_map = {}
        for art in articles:
            for topic in art.get('topics', []):
                if topic not in topic_map:
                    topic_map[topic] = []
                topic_map[topic].append(art['id'])
        
        # Track edges per article to limit connections
        article_edge_count = {}
        edge_count = 0
        skipped = 0
        
        for topic, node_ids in tqdm(topic_map.items(), desc="Topic edges", unit="topic", file=sys.stdout, dynamic_ncols=True):
            # Skip very large topic groups (generic topics like "business")
            if len(node_ids) > 500:
                skipped += 1
                continue
                
            for i in range(len(node_ids)):
                id1 = node_ids[i]
                # Limit edges per article
                if article_edge_count.get(id1, 0) >= max_edges_per_article:
                    continue
                    
                for j in range(i + 1, min(i + 20, len(node_ids))):  # Only check nearby articles
                    id2 = node_ids[j]
                    if article_edge_count.get(id2, 0) >= max_edges_per_article:
                        continue
                    
                    if not self.graph.has_edge(id1, id2) and not self.graph.has_edge(id2, id1):
                        if self.timestamps[id1] <= self.timestamps[id2]:
                            src, tgt = id1, id2
                        else:
                            src, tgt = id2, id1
                        
                        self.graph.add_edge(src, tgt)
                        if (src, tgt) not in self.edge_features:
                            self.edge_features[(src, tgt)] = {
                                'similarity': 0.5,
                                'time_diff': abs(self.timestamps[src] - self.timestamps[tgt]),
                                'shared_topic': topic
                            }
                            edge_count += 1
                            article_edge_count[id1] = article_edge_count.get(id1, 0) + 1
                            article_edge_count[id2] = article_edge_count.get(id2, 0) + 1
        
        print(f"Added {edge_count} topic edges (skipped {skipped} large topics)", flush=True)
    
    def compute_centrality_scores(self):
        """Compute centrality scores for nodes"""
        print("Computing PageRank centrality...")
        try:
            pagerank = nx.pagerank(self.graph, alpha=0.85)
            for node_id, score in pagerank.items():
                self.graph.nodes[node_id]['pagerank'] = float(score)
        except Exception as e:
            print(f"Error computing PageRank: {e}")

    def detect_communities(self):
        """Detect communities using Louvain algorithm"""
        print("Detecting communities...")
        try:
            # Convert to undirected for community detection
            undirected = self.graph.to_undirected()
            communities = nx.community.louvain_communities(undirected)
            
            for idx, community in enumerate(communities):
                for node_id in community:
                    self.graph.nodes[node_id]['community_id'] = int(idx)
                    
            print(f"Detected {len(communities)} communities")
        except Exception as e:
            print(f"Error detecting communities: {e}")
            # Fallback: assign all to community 0
            for node_id in self.graph.nodes():
                self.graph.nodes[node_id]['community_id'] = 0

    def build_graph(self, articles: List[Dict]) -> Tuple[nx.DiGraph, Dict, Dict, Dict]:
        """Build complete temporal graph from articles"""
        print(f"Building graph from {len(articles)} articles...")
        
        # Filter and fix articles with None IDs
        valid_articles = []
        for idx, article in enumerate(articles):
            if article.get('id') is None:
                article['id'] = f"auto_{idx}"  # Generate ID if missing
            valid_articles.append(article)
        
        articles = valid_articles
        print(f"Valid articles after ID check: {len(articles)}")
        
        # Add nodes
        for article in articles:
            self.add_article_node(article)
        
        print(f"Added {self.graph.number_of_nodes()} nodes")
        
        # Add similarity edges
        self.add_similarity_edges(articles)
        
        # Add topic edges
        self.add_topic_edges(articles)
        
        print(f"Added {self.graph.number_of_edges()} edges")
        
        # Compute advanced metrics
        self.compute_centrality_scores()
        self.detect_communities()
        
        return self.graph, self.node_features, self.edge_features, self.timestamps
    
    def save_graph(self, output_dir: str = "data/processed"):
        """Save graph and features to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save graph structure
        graph_data = nx.node_link_data(self.graph)
        with open(output_dir / "graph_structure.json", 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        # Save node features
        node_feat_dict = {str(k): v.tolist() for k, v in self.node_features.items()}
        with open(output_dir / "node_features.json", 'w') as f:
            json.dump(node_feat_dict, f)
        
        # Save edge features
        edge_feat_dict = {f"{k[0]}-{k[1]}": v for k, v in self.edge_features.items()}
        with open(output_dir / "edge_features.json", 'w') as f:
            json.dump(edge_feat_dict, f)
        
        # Save timestamps
        with open(output_dir / "timestamps.json", 'w') as f:
            json.dump(self.timestamps, f)
        
        print(f"Graph saved to {output_dir}")
    
    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            'density': nx.density(self.graph),
            'num_connected_components': nx.number_weakly_connected_components(self.graph),
            'num_communities': len(set(nx.get_node_attributes(self.graph, 'community_id').values()))
        }

# Example usage
if __name__ == "__main__":
    # Load processed articles
    with open('data/processed/processed_news.json', 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    # Build graph
    builder = NewsGraphBuilder(similarity_threshold=0.6)
    graph, node_features, edge_features, timestamps = builder.build_graph(articles)
    
    # Save graph
    builder.save_graph()
    
    # Print statistics
    stats = builder.get_statistics()
    print("\nGraph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
