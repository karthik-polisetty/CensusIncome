# 🎯 Census Income Classification

Predict whether an individual earns >$50K/year using demographic and work-related data, powered by Logistic Regression.

---

## 📌 Overview
This project uses the **UCI Adult Income dataset** to train, evaluate, and deploy a machine learning model as a web application.

---

## 🚀 Features
✅ End-to-end ML pipeline (data ingestion → transformation → training → prediction)  
✅ Flask web app for real-time predictions  
✅ Modular, production-ready codebase  
✅ Docker support for deployment  
✅ Artifacts & logs for traceability  

---

## 📂 Project Structure
```text
CensusIncome/
│
├── app.py                  # Flask web app
├── src/                    # Modular ML code
│   ├── components/         # data_ingestion, transformation, training
│   ├── pipeline/           # training & prediction pipelines
│   ├── utils.py, logger.py
├── notebooks/              # EDA & model-building
├── artifacts/              # saved models, datasets
├── templates/              # HTML templates
├── requirements.txt        # Python dependencies
├── Dockerfile              # Containerization
├── README.md
```

---

## 🛠️ How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
python src/pipeline/training_pipeline.py
```

### Run the Web App
```bash
python app.py
```
Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Using Docker
```bash
docker build -t census-income .
docker run -p 5000:5000 census-income
```

---

## 📊 Dataset
- [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- Features: age, education, marital status, occupation, hours-per-week, etc.

---

## 📈 Results
| Metric       | Value               |
|--------------|--------------------|
| Best Model   | Logistic Regression |
| Accuracy     | 83.16%             |

---

## 📖 Key Insights
- Education and occupation significantly impact income level.
- Model can help in policy decisions, HR processes, and demographic analysis.

---

## 📄 License
This project is for educational and demonstration purposes.
