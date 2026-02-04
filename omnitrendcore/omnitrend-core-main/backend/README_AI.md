# News Trend Analysis with Temporal Graph Networks (TGN)

A comprehensive system for analyzing news trends using multilingual NLP and Temporal Graph Networks. This project processes news articles, builds temporal graphs, and identifies trending topics using state-of-the-art deep learning techniques.

## 🌟 Features

- **Multilingual NLP Processing**: Supports Turkish, English, and other languages
- **Named Entity Recognition**: Extracts people, organizations, and locations
- **Sentiment Analysis**: Analyzes sentiment with 5-star rating
- **Topic Extraction**: Automatically identifies article topics
- **Temporal Graph Network**: Models relationships between articles over time
- **Trend Detection**: Identifies trending topics and influential articles
- **Advanced Metrics**: PageRank, Community Detection, Trend Velocity, Controversy Score
- **Benchmark Comparison**: Compare against 6+ baseline methods (TF-IDF, MLP, GCN, etc.)
- **Visualization**: Generates comprehensive trend analysis visualizations

## 💡 Key Technical Contributions

This project advances standard trend detection by implementing **Advanced Temporal Modeling**:

1.  **Time2Vec Encoding**:
    *   Replaces simple time timestamps with learnable vector representations.
    *   Captures **periodic patterns** (e.g., daily/weekly news cycles) and linear trends simultaneously.

2.  **LSTM Temporal Memory**:
    *   A recurrent memory module that tracks **topic evolution** over time.
    *   Allows the model to maintain a "state" for each node, understanding the history of a trend rather than just its current snapshot.

3.  **Exponential Decay Attention**:
    *   Time-aware graph attention that naturally prioritizes recent information.
    *   Uses a learnable decay factor to weigh historical data appropriately.

## 📊 Advanced Metrics

### Graph-Level Metrics
| Metric | Description |
|--------|-------------|
| **PageRank Centrality** | Identifies influential articles in the network |
| **Community Detection** | Groups related articles using Louvain algorithm |
| **Attention Entropy** | Measures model certainty (interpretability) |

### Trend Dynamics
| Metric | Formula | Description |
|--------|---------|-------------|
| **Trend Velocity** | `dS/dt` | Rate of trend score change |
| **Trend Acceleration** | `d²S/dt²` | Viral potential indicator |
| **Controversy Score** | `Var(sentiment)` | Detects polarizing topics |

## 🔬 Benchmark Comparison

Compare the proposed TGN against multiple baselines:

| Method | Type | Description |
|--------|------|-------------|
| TF-IDF | Traditional | Cosine similarity on keyword vectors |
| Recency | Heuristic | Newer articles rank higher |
| Degree | Graph | Node degree as trend indicator |
| MLP | Neural | Embeddings only (no graph) |
| Static GCN | GNN | Graph structure, no temporal features |
| Standard TGN | GNN | Basic temporal modeling |
| **Proposed (Ours)** | GNN | Time2Vec + Hawkes + Neural ODE |

Run benchmarks with:
```bash
python run_benchmarks.py
```

## 🧮 Advanced Mathematical Implementations

Beyond standard temporal modeling, the project includes cutting-edge techniques:

### Neural Hawkes Process Attention
**Purpose**: Model self-exciting news bursts (viral trends)

**Math**: $\lambda(t) = \mu + \sum \alpha \exp(-\beta \cdot \Delta t)$
- Learns base intensity ($\mu$), excitation strength ($\alpha$), and decay rate ($\beta$)
- Captures how one news article triggers cascading follow-ups

**Usage**: Set `use_advanced_temporal=True` in model config

### Neural ODE Memory  
**Purpose**: Continuous-time hidden state evolution

**Math**: $\frac{dh(t)}{dt} = f_\theta(h(t), t)$ solved via Euler integration
- Handles irregular publication times naturally (5 minutes vs 5 days)
- No external dependencies (custom solver)

**File**: `src/tgn/layers.py`

## 📁 Project Structure

```
.
├── data/
│   ├── raw/              # Raw news data (JSON/CSV)
│   └── processed/        # NLP-processed data and graphs
├── src/
│   ├── ingestion/        # News collection modules
│   ├── nlp/              # Multilingual NLP processing
│   ├── graph/            # Graph construction (PageRank, Communities)
│   ├── tgn/              # TGN model and encoder
│   ├── benchmarks/       # Baseline models and evaluation
│   └── utils/            # Helper functions and analyzers
├── models/               # Trained model checkpoints
├── visualizations/       # Generated plots and reports
├── main.py               # Main pipeline script
├── run_benchmarks.py     # Benchmark comparison script
└── requirements.txt      # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd news-trend-analysis

# Create conda environment (optional but recommended)
conda create -n news-tgn python=3.10
conda activate news-tgn

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install torch-geometric
pip install sentence-transformers
pip install pandas numpy tqdm matplotlib seaborn networkx requests transformers
```

### 2. Prepare Your Data

Place your news articles in `data/raw/` as JSON or CSV files:

```json
[
  {
    "id": 1,
    "title": "Article title",
    "content": "Article content...",
    "source": "News Source",
    "published_at": "2025-11-16T10:00:00",
    "url": "https://example.com/article"
  }
]
```

### 3. Run the Pipeline

```bash
# Run full pipeline
python main.py --step all --num-epochs 100

# Or run individual steps
python main.py --step collect   # Collect news
python main.py --step process   # NLP processing
python main.py --step build     # Build graph
python main.py --step train     # Train TGN
python main.py --step analyze   # Analyze trends
```

### 4. View Results

Results will be saved in:
- `visualizations/` - Trend plots and charts
- `trend_analysis_report.json` - Detailed analysis report
- `models/` - Trained model checkpoints

## 🔧 Configuration Options

```bash
# Model parameters
python main.py --hidden-dim 256 --num-layers 3 --num-heads 4

# Training parameters
python main.py --num-epochs 100 --learning-rate 0.001

# Graph building
python main.py --similarity-threshold 0.6

# Analysis
python main.py --top-k 10  # Top 10 trending topics
```

## 📊 Pipeline Stages

### Stage 1: Data Collection
- Fetches news from various sources
- Supports multiple formats (JSON, CSV)
- Saves raw data to `data/raw/`

### Stage 2: NLP Processing
- **Embeddings**: Multilingual sentence transformers (384-dim)
- **Sentiment**: 5-star rating with polarity
- **NER**: Extracts entities (person, organization, location)
- **Topics**: Automatic topic extraction
- Saves processed data to `data/processed/`

### Stage 3: Graph Building
- Creates temporal directed graph
- **Nodes**: Articles with features (embeddings + sentiment)
- **Edges**: Similarity and topic-based connections
- **Temporal**: Time-aware edge weighting
- Saves graph structure and features

### Stage 4: TGN Training
- **Architecture**: Multi-layer Graph Attention Networks
- **Temporal Attention**: Time-aware edge weighting
- **Self-supervised**: Pseudo-labels from graph structure
- **Loss**: Classification + regression (trend scores)
- Saves best model checkpoint

### Stage 5: Trend Analysis
- Detects trending topics
- Identifies influential articles
- Analyzes topic evolution over time
- Sentiment trends per topic
- Generates visualizations

## 📈 Model Architecture

```
Input Features (385-dim)
    ↓
[Input Projection]
    ↓
[Temporal Attention Module]
    ↓
[GAT Layer 1] → LayerNorm → ReLU → Dropout
    ↓
[GAT Layer 2] → LayerNorm → ReLU → Dropout
    ↓
[GAT Layer 3] → LayerNorm → ReLU → Dropout
    ↓
[Trend Predictor]
    ↓
Output: {embeddings, trend_class, trend_score}
```

## 🎯 Use Cases

1. **Media Monitoring**: Track emerging news trends in real-time
2. **Content Strategy**: Identify trending topics for content creation
3. **Public Opinion**: Analyze sentiment shifts over time
4. **Research**: Study information propagation patterns
5. **News Recommendation**: Recommend trending content to users

## 🔍 Example Output

```
Top 10 Trending Topics:
  1. economy: score=0.842, articles=45
  2. politics: score=0.789, articles=38
  3. technology: score=0.756, articles=31
  4. health: score=0.723, articles=27
  5. sports: score=0.698, articles=22

Top 10 Trending Articles:
  1. "Merkez Bankası faiz kararını açıkladı..."
     Score: 0.923, Topics: economy, politics, financial
  2. "Yeni teknoloji düzenlemeleri yolda..."
     Score: 0.891, Topics: technology, politics, digital
```

## 🛠️ Advanced Usage

### Custom News Sources

Edit `src/ingestion/fetch_news.py` to add your news API:

```python
# Add your API integration
API_KEY = "your_api_key"
articles = collector.fetch_from_api(
    f"https://newsapi.org/v2/top-headlines?country=tr&apiKey={API_KEY}"
)
```

### Adjust Model Parameters

```python
# In main.py or directly in code
model = NewsTGN(
    input_dim=385,
    hidden_dim=512,      # Increase capacity
    num_layers=4,        # Deeper network
    num_heads=8,         # More attention heads
    dropout=0.2          # More regularization
)
```

### Custom Trend Detection

```python
# In src/utils/trend_analyzer.py
def custom_trend_score(self, article):
    # Implement your own trend scoring logic
    score = (
        article['sentiment_score'] * 0.3 +
        article['entity_count'] * 0.3 +
        article['degree_centrality'] * 0.4
    )
    return score
```

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{news_tgn_2025,
  author = {Your Name},
  title = {News Trend Analysis with Temporal Graph Networks},
  year = {2025},
  url = {https://github.com/yourusername/news-tgn}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch Geometric team for graph neural network tools
- Hugging Face for transformer models
- Sentence-Transformers for multilingual embeddings

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is a research/educational project. For production use, consider additional optimizations, error handling, and security measures.
