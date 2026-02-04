import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class TrendAnalyzer:
    """Analyzes news trends using trained TGN model"""
    
    def __init__(self, model, device: str = None):
        self.model = model
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
    
    def analyze_trends(
        self,
        data,
        processed_articles: List[Dict],
        top_k: int = 10
    ) -> Dict:
        """
        Analyze trends in news articles
        
        Returns:
            Dictionary containing:
                - trending_topics: Top trending topics
                - trending_articles: Articles with highest trend scores
                - topic_evolution: How topics evolved over time
                - sentiment_trends: Sentiment analysis per topic
        """
        with torch.no_grad():
            # Check graph size - use CPU for large graphs to avoid OOM
            num_edges = data.edge_index.shape[1] if hasattr(data, 'edge_index') else 0
            num_nodes = data.x.shape[0]
            
            # For very large graphs, sample edges to reduce memory
            MAX_EDGES_GPU = 500000  # 500k edges max for GPU
            MAX_EDGES_CPU = 2000000  # 2M edges max even for CPU
            
            if num_edges > MAX_EDGES_CPU:
                # Sample a subset of edges
                print(f"Graph too large ({num_edges:,} edges), sampling to {MAX_EDGES_CPU:,} edges...")
                perm = torch.randperm(num_edges)[:MAX_EDGES_CPU]
                data.edge_index = data.edge_index[:, perm]
                if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                    data.edge_attr = data.edge_attr[perm]
                if hasattr(data, 'time_diffs') and data.time_diffs is not None:
                    data.time_diffs = data.time_diffs[perm]
                num_edges = MAX_EDGES_CPU
            
            if num_edges > MAX_EDGES_GPU:
                # Use CPU for inference
                print(f"Using CPU for inference (graph has {num_edges:,} edges)...")
                inference_device = torch.device('cpu')
                self.model.to(inference_device)
                data = data.to(inference_device)
            else:
                inference_device = self.device
                data = data.to(inference_device)
            
            # Get model predictions
            output = self.model(
                data.x,
                data.edge_index,
                time_diffs=data.time_diffs if hasattr(data, 'time_diffs') else None
            )
            
            embeddings = output['embeddings'].cpu().numpy()
            trend_scores = output['trend_score'].cpu().numpy()
            trend_classes = output['trend_class'].argmax(dim=1).cpu().numpy()
            
            # Move model back to original device
            self.model.to(self.device)
        
        # Limit articles to match number of graph nodes
        num_nodes = data.x.shape[0]
        processed_articles = processed_articles[:num_nodes]
        
        # Extract topics from articles
        node_topics = [art.get('topics', []) for art in processed_articles]
        
        
        trending_topics = self._detect_trending_topics(
            trend_scores,
            node_topics,
            number_of_top_topics=top_k
        )
        
        trending_articles = self._find_trending_articles(
            processed_articles,
            trend_scores,
            number_of_top_articles=top_k
        )
        
        # Analyze topic evolution
        topic_evolution = self._analyze_topic_evolution(
            processed_articles,
            trend_scores
        )
        
        # Analyze sentiment trends
        sentiment_trends = self._analyze_sentiment_trends(
            processed_articles,
            trend_scores
        )
        
        return {
            'trending_topics': trending_topics,
            'trending_articles': trending_articles,
            'topic_evolution': topic_evolution,
            'sentiment_trends': sentiment_trends,
            'trend_dynamics': self._calculate_trend_dynamics(topic_evolution),
            'controversy_scores': self._calculate_controversy_scores(sentiment_trends),
            'embeddings': embeddings,
            'trend_scores': trend_scores.flatten(),
            'trend_classes': trend_classes
        }
    
    def _detect_trending_topics(
        self,
        article_trend_scores: np.ndarray,
        article_topics_list: List[List[str]],
        number_of_top_topics: int
    ) -> List[Tuple[str, float, int]]:
        topic_to_scores_mapping = defaultdict(list)
        
        for article_index, topics_for_article in enumerate(article_topics_list):
            current_article_score = article_trend_scores[article_index].item()
            for topic_name in topics_for_article:
                topic_to_scores_mapping[topic_name].append(current_article_score)
        
        topic_statistics_list = []
        for topic_name, score_list in topic_to_scores_mapping.items():
            topic_statistics_list.append({
                'topic': topic_name,
                'avg_score': np.mean(score_list),
                'max_score': np.max(score_list),
                'count': len(score_list),
                'total_score': np.sum(score_list)
            })
        
        topic_statistics_list.sort(key=lambda x: x['total_score'], reverse=True)
        
        return [
            (stat['topic'], stat['avg_score'], stat['count'])
            for stat in topic_statistics_list[:number_of_top_topics]
        ]
    
    def _find_trending_articles(
        self,
        processed_articles_list: List[Dict],
        article_trend_scores: np.ndarray,
        number_of_top_articles: int
    ) -> List[Dict]:
        articles_with_trend_scores = []
        for article_index, article_data in enumerate(processed_articles_list):
            articles_with_trend_scores.append({
                'id': article_data['id'],
                'title': article_data['title'],
                'source': article_data['source'],
                'published_at': article_data['published_at'],
                'topics': article_data.get('topics', []),
                'sentiment': article_data['sentiment'],
                'trend_score': float(article_trend_scores[article_index])
            })
        
        articles_with_trend_scores.sort(key=lambda x: x['trend_score'], reverse=True)
        
        return articles_with_trend_scores[:number_of_top_articles]
    
    def _analyze_topic_evolution(
        self,
        processed_articles_list: List[Dict],
        article_trend_scores: np.ndarray
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        topic_to_timeline_mapping = defaultdict(list)
        
        for article_index, article_data in enumerate(processed_articles_list):
            try:
                article_timestamp = datetime.fromisoformat(article_data['published_at'].replace('Z', '+00:00'))
                # Convert to naive datetime to avoid comparison issues
                if article_timestamp.tzinfo is not None:
                    article_timestamp = article_timestamp.replace(tzinfo=None)
                article_trend_score = float(article_trend_scores[article_index])
                
                for topic_name in article_data.get('topics', []):
                    topic_to_timeline_mapping[topic_name].append((article_timestamp, article_trend_score))
            except:
                continue
        
        for topic_name in topic_to_timeline_mapping:
            try:
                topic_to_timeline_mapping[topic_name].sort(key=lambda x: x[0])
            except TypeError:
                # If sorting fails, skip this topic
                pass
        
        return dict(topic_to_timeline_mapping)
    
    def _analyze_sentiment_trends(
        self,
        processed_articles_list: List[Dict],
        article_trend_scores: np.ndarray
    ) -> Dict[str, Dict]:
        topic_to_sentiment_data = defaultdict(lambda: {'scores': [], 'labels': []})
        
        for article_index, article_data in enumerate(processed_articles_list):
            for topic_name in article_data.get('topics', []):
                article_sentiment = article_data['sentiment']
                topic_to_sentiment_data[topic_name]['scores'].append(article_sentiment['score'])
                topic_to_sentiment_data[topic_name]['labels'].append(article_sentiment['label'])
        
        sentiment_statistics_by_topic = {}
        for topic_name, sentiment_data in topic_to_sentiment_data.items():
            total_labels_count = len(sentiment_data['labels'])
            sentiment_statistics_by_topic[topic_name] = {
                'avg_sentiment': np.mean(sentiment_data['scores']),
                'sentiment_std': np.std(sentiment_data['scores']),
                'positive_ratio': sentiment_data['labels'].count('positive') / total_labels_count,
                'negative_ratio': sentiment_data['labels'].count('negative') / total_labels_count,
                'neutral_ratio': sentiment_data['labels'].count('neutral') / total_labels_count
            }
        
        return sentiment_statistics_by_topic
    
    def _calculate_trend_dynamics(
        self,
        topic_evolution: Dict[str, List[Tuple[datetime, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate velocity and acceleration of trends"""
        dynamics = {}
        
        for topic, timeline in topic_evolution.items():
            if len(timeline) < 3:
                dynamics[topic] = {'velocity': 0.0, 'acceleration': 0.0}
                continue
            
            # Sort by time
            timeline.sort(key=lambda x: x[0])
            
            # Get last 3 points
            t3, s3 = timeline[-1] # Current
            t2, s2 = timeline[-2] # Previous
            t1, s1 = timeline[-3] # Before previous
            
            # Time diffs in hours
            dt2 = (t3 - t2).total_seconds() / 3600
            dt1 = (t2 - t1).total_seconds() / 3600
            
            if dt2 == 0 or dt1 == 0:
                dynamics[topic] = {'velocity': 0.0, 'acceleration': 0.0}
                continue
                
            # Velocity: dS/dt
            v2 = (s3 - s2) / dt2
            v1 = (s2 - s1) / dt1
            
            # Acceleration: dV/dt
            acc = (v2 - v1) / dt2
            
            dynamics[topic] = {
                'velocity': float(v2),
                'acceleration': float(acc)
            }
            
        return dynamics

    def _calculate_controversy_scores(
        self,
        sentiment_trends: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Calculate controversy score based on sentiment variance
        Formula: variance * log(N)
        High variance means polarized opinions (controversial)
        """
        controversy_scores = {}
        
        for topic, stats in sentiment_trends.items():
            # Variance of sentiment scores
            variance = stats['sentiment_std'] ** 2
            
            # Number of articles (inferred from ratios, but we need raw count)
            # For approximation, we use variance directly as the core metric
            # If we had count, we would multiply by log(count)
            
            controversy_scores[topic] = float(variance)
            
        return controversy_scores
    
    def visualize_trends(
        self,
        analysis_results: Dict,
        visualization_output_directory: str = "visualizations"
    ):
        output_directory_path = Path(visualization_output_directory)
        output_directory_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self._plot_trending_topics(
                analysis_results['trending_topics'],
                output_directory_path / "trending_topics.png"
            )
        except Exception:
            pass
        
        try:
            self._plot_topic_evolution(
                analysis_results['topic_evolution'],
                output_directory_path / "topic_evolution.png"
            )
        except Exception:
            pass
        
        try:
            self._plot_sentiment_trends(
                analysis_results['sentiment_trends'],
                output_directory_path / "sentiment_trends.png"
            )
        except Exception:
            pass
        
        try:
            self._plot_score_distribution(
                analysis_results['trend_scores'],
                output_directory_path / "score_distribution.png"
            )
        except Exception:
            pass
    
    def _plot_trending_topics(self, trending_topics_list: List, output_file_path: Path):
        if not trending_topics_list:
            return
        
        topic_names = [topic_data[0] for topic_data in trending_topics_list]
        topic_scores = [topic_data[1] for topic_data in trending_topics_list]
        article_counts = [topic_data[2] for topic_data in trending_topics_list]
        
        plt.figure(figsize=(12, 6))
        bar_plot = plt.barh(topic_names, topic_scores)
        
        if max(topic_scores) > min(topic_scores):
            normalized_topic_scores = [(score - min(topic_scores)) / (max(topic_scores) - min(topic_scores)) for score in topic_scores]
        else:
            normalized_topic_scores = [0.5] * len(topic_scores)
        
        color_values = plt.cm.RdYlGn(np.array(normalized_topic_scores))
        for bar, color in zip(bar_plot, color_values):
            bar.set_color(color)
        
        for bar_index, (bar, article_count) in enumerate(zip(bar_plot, article_counts)):
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                    f' n={article_count}', va='center', fontsize=9, color='black')
        
        plt.xlabel('Average Trend Score', fontsize=12)
        plt.title('Top Trending Topics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_topic_evolution(self, topic_evolution_data: Dict, output_file_path: Path):
        if not topic_evolution_data:
            return
        
        plt.figure(figsize=(14, 8))
        
        topics_sorted_by_data_points = sorted(
            topic_evolution_data.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:5]
        
        number_of_plotted_topics = 0
        for topic_name, timeline_data in topics_sorted_by_data_points:
            if len(timeline_data) > 1:
                timestamps_list = [data_point[0] for data_point in timeline_data]
                trend_scores_list = [data_point[1] for data_point in timeline_data]
                plt.plot(timestamps_list, trend_scores_list, marker='o', label=topic_name, linewidth=2, markersize=8)
                number_of_plotted_topics += 1
        
        if number_of_plotted_topics == 0:
            for topic_name, timeline_data in topics_sorted_by_data_points[:10]:
                if len(timeline_data) > 0:
                    timestamps_list = [data_point[0] for data_point in timeline_data]
                    trend_scores_list = [data_point[1] for data_point in timeline_data]
                    plt.scatter(timestamps_list, trend_scores_list, label=topic_name, s=100, alpha=0.7)
        
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Trend Score', fontsize=12)
        plt.title('Topic Evolution Over Time', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_file_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_sentiment_trends(self, sentiment_trends_by_topic: Dict, output_file_path: Path):
        if not sentiment_trends_by_topic:
            return
        
        top_topics_by_sentiment = sorted(
            sentiment_trends_by_topic.items(),
            key=lambda x: x[1]['positive_ratio'] + x[1]['negative_ratio'],
            reverse=True
        )[:10]
        
        if not top_topics_by_sentiment:
            return
        
        topic_names_list = [topic_item[0] for topic_item in top_topics_by_sentiment]
        positive_ratios_list = [topic_item[1]['positive_ratio'] for topic_item in top_topics_by_sentiment]
        neutral_ratios_list = [topic_item[1]['neutral_ratio'] for topic_item in top_topics_by_sentiment]
        negative_ratios_list = [topic_item[1]['negative_ratio'] for topic_item in top_topics_by_sentiment]
        
        figure, axes = plt.subplots(figsize=(12, 6))
        
        x_positions = np.arange(len(topic_names_list))
        bar_width = 0.25
        
        axes.bar(x_positions - bar_width, positive_ratios_list, bar_width, label='Positive', color='#2ecc71', alpha=0.8)
        axes.bar(x_positions, neutral_ratios_list, bar_width, label='Neutral', color='#95a5a6', alpha=0.8)
        axes.bar(x_positions + bar_width, negative_ratios_list, bar_width, label='Negative', color='#e74c3c', alpha=0.8)
        
        axes.set_xlabel('Topics', fontsize=12)
        axes.set_ylabel('Ratio', fontsize=12)
        axes.set_title('Sentiment Distribution by Topic', fontsize=14, fontweight='bold')
        axes.set_xticks(x_positions)
        axes.set_xticklabels(topic_names_list, rotation=45, ha='right')
        axes.legend(fontsize=10)
        axes.grid(True, alpha=0.3, axis='y')
        axes.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plt.savefig(output_file_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_score_distribution(self, trend_scores_array: np.ndarray, output_file_path: Path):
        plt.figure(figsize=(10, 6))
        
        number_of_samples = len(trend_scores_array)
        number_of_bins = min(50, max(5, number_of_samples // 2))
        
        plt.hist(trend_scores_array, bins=number_of_bins, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(trend_scores_array.mean(), color='red', linestyle='--', 
                   label=f'Mean: {trend_scores_array.mean():.3f}', linewidth=2)
        plt.axvline(np.median(trend_scores_array), color='green', linestyle='--',
                   label=f'Median: {np.median(trend_scores_array):.3f}', linewidth=2)
        
        plt.xlabel('Trend Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Distribution of Trend Scores (n={number_of_samples})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_report(self, analysis_results: Dict, output_path: str):
        """Save analysis report to JSON"""
        # Convert numpy arrays to lists
        report = {
            'trending_topics': analysis_results['trending_topics'],
            'trending_articles': analysis_results['trending_articles'],
            'sentiment_trends': analysis_results['sentiment_trends'],
            'trend_dynamics': analysis_results.get('trend_dynamics', {}),
            'controversy_scores': analysis_results.get('controversy_scores', {}),
            'statistics': {
                'total_articles': len(analysis_results['trend_scores']),
                'avg_trend_score': float(np.mean(analysis_results['trend_scores'])),
                'max_trend_score': float(np.max(analysis_results['trend_scores'])),
                'high_trend_count': int(np.sum(analysis_results['trend_classes'] == 2)),
                'medium_trend_count': int(np.sum(analysis_results['trend_classes'] == 1)),
                'low_trend_count': int(np.sum(analysis_results['trend_classes'] == 0))
            },
            'generated_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"Report saved to {output_path}")

# Example usage
if __name__ == "__main__":
    from model import NewsTGN
    from train import TGNTrainer
    
    # Load model
    model = NewsTGN(input_dim=385, hidden_dim=256)
    trainer = TGNTrainer(model)
    trainer.load_checkpoint(Path("models/best_model.pt"))
    
    # Load data
    data = trainer.load_graph_data("data/processed")
    
    with open('data/processed/processed_news.json', 'r') as f:
        articles = json.load(f)
    
    # Analyze trends
    analyzer = TrendAnalyzer(model)
    results = analyzer.analyze_trends(data, articles, top_k=10)
    
    # Visualize
    analyzer.visualize_trends(results, "visualizations")
    
    # Save report
    analyzer.save_report(results, "trend_analysis_report.json")
    
    print("\nTop Trending Topics:")
    for topic, score, count in results['trending_topics']:
        print(f"  {topic}: {score:.3f} (n={count})")
