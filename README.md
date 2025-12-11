# Trading API

This is a FastAPI project that helps store forex price data, save trade history, analyze the market using indicators like EMA and RSI, and use machine learning to create automatic trading signals with suggested Take Profit (TP) and Stop Loss (SL).

---

## Overview

This API can:

- Save and retrieve price data from the database  
- Save and retrieve trade records  
- Calculate basic trading indicators (EMA9, EMA21, RSI)  
- Use a machine learning model to guess the next price  
- Suggest whether to **buy**, **sell**, or **hold**  
- Automatically generate TP and SL levels  
- Connect to a PostgreSQL database for storage  
- Log all actions and errors for easy debugging  

In short:  
**You send price data → the system learns → it gives you a trading signal with TP and SL.**

---

## Tech Stack Used

- **Python** — main language  
- **FastAPI** — API framework  
- **PostgreSQL** — database  
- **SQLAlchemy** — ORM for database operations  
- **scikit-learn** — machine learning model  
- **joblib** — save/load trained model  
- **Pandas & NumPy** — data processing  
- **ta library** — computes trading indicators  
- **Pydantic** — request/response validation  

---
## How to Run the Application Locally

Bundle all
```angular2html
 pip install -r requirements.txt
```


run the application
```angular2html
uvicorn app.main:app --reload --port 8000

```