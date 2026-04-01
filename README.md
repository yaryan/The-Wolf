readme_text = r"""# The-Wolf

**The WOLF** (**W**eighted **O**bservation of **L**atent **F**inance) is a news-aware multimodal AI pipeline for short-term financial market prediction. It combines **historical stock price data** with **financial news sentiment** to generate next-day stock price forecasts.

Financial markets are influenced not only by past price movements, but also by real-world events such as earnings reports, analyst actions, regulatory developments, product launches, and macroeconomic news. Many traditional forecasting systems rely mainly on numerical time-series data and miss predictive signals contained in unstructured text. WOLF is designed to bridge that gap by aligning market news with stock price history and using both sources in a unified modeling workflow.

In this project, stock OHLCV data and company-specific news articles are processed together. News articles are filtered by ticker, aligned to trading dates, scored with **FinBERT** for sentiment, and converted into structured context that complements price-based technical signals such as returns, volatility, and RSI. These inputs are then transformed into instruction-style training examples for a language model, which is fine-tuned to predict the **next trading-day adjusted close price**.

The implementation is built in **Python** using **PyTorch**, **Hugging Face Transformers**, **PEFT/LoRA**, **Datasets**, and common data science libraries such as **Pandas** and **NumPy**. Training for the experiments in this repository was run on an **NVIDIA H100 GPU**.

---

## Dataset

This project expects two datasets:

1. **Stock price CSV files**  
   Format:
   - `stock_prices/AAPL.csv`
   - `stock_prices/AMZN.csv`

2. **News CSV files**  
   Format:
   - `news/AAPL_news.csv`
   - `news/AMZN_news.csv`

**Dataset link:**  
[Google Drive Dataset Folder](https://drive.google.com/drive/folders/1Ibr0TcWWV7pH-9d8eZadGEYgVo5C4SPB?usp=sharing)

---

## Project Structure

### Google Colab / Google Drive layout

```bash
/content/drive/MyDrive/FOA_Data/
├── stock_prices/
│   ├── AAPL.csv
│   ├── AMZN.csv
│   └── ...
├── news/
│   ├── AAPL_news.csv
│   ├── AMZN_news.csv
│   └── ...
├── llm_forecast_data/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   └── test_predictions.csv
└── llm_forecaster_model_multi/
```
---

### TESTING
Run testing.ipynb using the dataset link, which contains the models, you can change the following path, if need be:

```DATA_DIR = "/content/drive/MyDrive/FOA_Data/llm_forecast_data"```

```TEST_JSONL = os.path.join(DATA_DIR, "test.jsonl")```


```MODEL_DIR = "/content/drive/MyDrive/FOA_Data/llm_forecaster_model_multi"```


