import logging
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DiversityMetrics:
    """Diversity metrics"""
    total_articles: int
    unique_sources: int
    source_distribution: Dict[str, int]
    shannon_entropy: float
    gini_coefficient: float
    coverage_score: float
    reliability_weighted_score: float
    temporal_distribution: Dict[str, int]
    category_distribution: Dict[str, int]

class SourceDiversityAnalyzer:
    """Analyzes news source diversity"""
    
    def __init__(self, source_configs: List):
        self.source_configs = {src.name: src for src in source_configs}
        logger.info(f"Initialized diversity analyzer with {len(source_configs)} sources")
    
    def analyze_collection(self, articles: List[Dict]) -> DiversityMetrics:
        """Analyze diversity of collected articles"""
        if not articles:
            logger.warning("No articles to analyze")
            return self._empty_metrics()
        
        # Source distribution
        source_counts = Counter(article.get('source') for article in articles)
        unique_sources = len(source_counts)
        total_articles = len(articles)
        
        # Shannon Entropy (diversity measure)
        shannon_entropy = self._calculate_shannon_entropy(source_counts, total_articles)
        
        # Gini Coefficient (inequality measure)
        gini_coefficient = self._calculate_gini_coefficient(list(source_counts.values()))
        
        # Coverage Score
        coverage_score = unique_sources / len(self.source_configs) if self.source_configs else 0
        
        # Reliability-weighted score
        reliability_score = self._calculate_reliability_weighted_score(source_counts)
        
        # Temporal distribution
        temporal_dist = self._analyze_temporal_distribution(articles)
        
        # Category distribution
        category_dist = self._analyze_category_distribution(articles)
        
        metrics = DiversityMetrics(
            total_articles=total_articles,
            unique_sources=unique_sources,
            source_distribution=dict(source_counts),
            shannon_entropy=shannon_entropy,
            gini_coefficient=gini_coefficient,
            coverage_score=coverage_score,
            reliability_weighted_score=reliability_score,
            temporal_distribution=temporal_dist,
            category_distribution=category_dist
        )
        
        logger.info(f"Diversity analysis complete: {unique_sources} sources, "
                   f"Shannon={shannon_entropy:.3f}, Gini={gini_coefficient:.3f}")
        
        return metrics
    
    def _calculate_shannon_entropy(self, counts: Counter, total: int) -> float:
        """Calculate Shannon Entropy (higher = more diversity)"""
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _calculate_gini_coefficient(self, values: List[int]) -> float:
        """Calculate Gini coefficient (lower = more equal distribution)"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        if cumsum[-1] == 0:
            return 0.0
        
        gini = (2 * sum((i + 1) * val for i, val in enumerate(sorted_values))) / (n * cumsum[-1])
        gini = gini - (n + 1) / n
        
        return gini
    
    def _calculate_reliability_weighted_score(self, source_counts: Counter) -> float:
        """Calculate weighted score based on reliability scores"""
        if not source_counts:
            return 0.0
        
        total_weighted = 0.0
        total_count = sum(source_counts.values())
        
        for source, count in source_counts.items():
            if source in self.source_configs:
                reliability = self.source_configs[source].reliability_score
                total_weighted += (count / total_count) * reliability
        
        return total_weighted
    
    def _analyze_temporal_distribution(self, articles: List[Dict]) -> Dict[str, int]:
        """Analyze temporal distribution"""
        temporal_counts = defaultdict(int)
        
        for article in articles:
            pub_date = article.get('published_at')
            if pub_date:
                try:
                    if isinstance(pub_date, str):
                        date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    else:
                        date = pub_date
                    
                    month_key = date.strftime('%Y-%m')
                    temporal_counts[month_key] += 1
                except Exception as e:
                    logger.debug(f"Could not parse date: {pub_date}")
        
        return dict(sorted(temporal_counts.items()))
    
    def _analyze_category_distribution(self, articles: List[Dict]) -> Dict[str, int]:
        """Analyze category distribution"""
        category_counts = defaultdict(int)
        
        for article in articles:
            source = article.get('source')
            if source in self.source_configs:
                category = self.source_configs[source].category
                category_counts[category] += 1
        
        return dict(category_counts)
    
    def _empty_metrics(self) -> DiversityMetrics:
        """Return empty metrics object"""
        return DiversityMetrics(
            total_articles=0,
            unique_sources=0,
            source_distribution={},
            shannon_entropy=0.0,
            gini_coefficient=0.0,
            coverage_score=0.0,
            reliability_weighted_score=0.0,
            temporal_distribution={},
            category_distribution={}
        )
    
    def generate_report(self, metrics: DiversityMetrics) -> str:
        """Generate diversity report"""
        report = []
        report.append("=" * 60)
        report.append("SOURCE DIVERSITY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"\nTotal Articles: {metrics.total_articles}")
        report.append(f"Unique Sources: {metrics.unique_sources}")
        report.append(f"Source Coverage Score: {metrics.coverage_score:.2%}")
        report.append(f"\nDiversity Metrics:")
        report.append(f"  - Shannon Entropy: {metrics.shannon_entropy:.3f} (max: {np.log2(metrics.unique_sources) if metrics.unique_sources > 0 else 0:.3f})")
        report.append(f"  - Gini Coefficient: {metrics.gini_coefficient:.3f} (0=equal, 1=unequal)")
        report.append(f"  - Reliability Score: {metrics.reliability_weighted_score:.3f}")
        
        report.append(f"\nSource Distribution:")
        sorted_sources = sorted(metrics.source_distribution.items(), 
                               key=lambda x: x[1], reverse=True)
        for source, count in sorted_sources:
            percentage = (count / metrics.total_articles * 100) if metrics.total_articles > 0 else 0
            reliability = self.source_configs.get(source, type('obj', (), {'reliability_score': 0})).reliability_score
            report.append(f"  - {source}: {count} ({percentage:.1f}%) [Reliability: {reliability:.2f}]")
        
        if metrics.category_distribution:
            report.append(f"\nCategory Distribution:")
            for category, count in sorted(metrics.category_distribution.items(), 
                                         key=lambda x: x[1], reverse=True):
                percentage = (count / metrics.total_articles * 100) if metrics.total_articles > 0 else 0
                report.append(f"  - {category}: {count} ({percentage:.1f}%)")
        
        if metrics.temporal_distribution:
            report.append(f"\nTemporal Distribution (Monthly):")
            for month, count in sorted(metrics.temporal_distribution.items()):
                report.append(f"  - {month}: {count} articles")
        
        report.append("\n" + "=" * 60)
        
        # Recommendations
        report.append("\nRECOMMENDATIONS:")
        if metrics.gini_coefficient > 0.5:
            report.append("  ⚠ Source distribution is very imbalanced. Collect more from underrepresented sources.")
        if metrics.coverage_score < 0.5:
            report.append("  ⚠ Less than half of available sources are used. Enable more sources.")
        if metrics.reliability_weighted_score < 0.8:
            report.append("  ⚠ Overall reliability score is low. Focus on more reliable sources.")
        if len(metrics.source_distribution) < 3:
            report.append("  ⚠ Very few sources used. Increase source diversity.")
        
        if (metrics.gini_coefficient <= 0.3 and metrics.coverage_score >= 0.7 and 
            metrics.reliability_weighted_score >= 0.85):
            report.append("  ✓ Excellent diversity and quality balance!")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def check_quality_thresholds(self, metrics: DiversityMetrics) -> Tuple[bool, List[str]]:
        """Check quality thresholds"""
        issues = []
        
        if metrics.unique_sources < 3:
            issues.append("Too few sources used (min: 3)")
        
        if metrics.gini_coefficient > 0.6:
            issues.append("Source distribution is very imbalanced (Gini > 0.6)")
        
        if metrics.coverage_score < 0.4:
            issues.append("Source coverage is too low (< 40%)")
        
        if metrics.reliability_weighted_score < 0.75:
            issues.append("Reliability score is insufficient (< 0.75)")
        
        if metrics.total_articles < 100:
            issues.append("Total article count is too low (< 100)")
        
        passed = len(issues) == 0
        return passed, issues