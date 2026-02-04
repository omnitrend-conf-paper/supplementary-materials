# Temel kullanım (son 90 gün, JSON output)
python main_collector.py

# Paralel toplama
python main_collector.py --parallel

# CSV formatında kaydet
python main_collector.py --format csv

# Hedef sayı belirt
python main_collector.py --target 1000

# Son 30 günü topla
python main_collector.py --days 30

# Tüm seçenekler birlikte
python main_collector.py --parallel --format json --target 5000 --days 60