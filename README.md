# Universal Agentic AI Data Analyst

This Streamlit application provides:
- **Data ingestion**: Supports CSV, Excel, PDF, HTML, XML, and SQLite.
- **Data cleaning**: Handles duplicates, missing values, and date parsing.
- **Domain detection**: Automatically identifies Sales, Marketing, HR, or Generic datasets.
- **Boardroom storytelling**: Generates executive-style reports with totals, averages, YoY growth, ROI, and attrition metrics.
- **Adaptive visualizations**: Monthly/quarterly/yearly trends, ROI scatter plots, and distributions.
- **LLM Q&A**: Asks 5 numeric questions (averages, totals, YoY comparisons) and answers them using a basic LLM layer.

---

## 📊 Dataset Schemas

### HR Dataset (employee_data.csv)
| Column Name        | Description                                |
|--------------------|--------------------------------------------|
| Employee ID        | Unique identifier for each employee        |
| Name               | Employee name                              |
| Department         | Department name (e.g., HR, IT, Sales)      |
| Role               | Job title                                  |
| Salary             | Monthly/annual salary                      |
| Gender             | Male/Female/Other                          |
| Age                | Employee age                               |
| Tenure             | Years in company                           |
| Attrition          | Whether employee left (Yes/No)             |
| Satisfaction Score | Survey score (1–5)                         |

### Sales Dataset (sales_data.csv)
| Column Name   | Description                                    |
|---------------|------------------------------------------------|
| Order ID      | Unique order identifier                        |
| Order Date    | Date of order                                  |
| Customer ID   | Unique customer identifier                     |
| Product       | Product name                                   |
| Category      | Product category                               |
| Region        | Sales region                                   |
| Quantity      | Units sold                                     |
| Unit Price    | Price per unit                                 |
| Total Price   | Quantity × Unit Price                          |
| Profit        | Profit per order                               |

### Marketing Dataset (marketing_data.csv)
| Column Name   | Description                                    |
|---------------|------------------------------------------------|
| Campaign      | Campaign name                                  |
| Channel       | Marketing channel (Email, Social, Ads)         |
| Impressions   | Number of times ad was shown                   |
| Clicks        | Number of clicks                               |
| Conversions   | Number of successful conversions               |
| Spend         | Campaign spend                                 |
| Revenue       | Revenue generated                              |
| ROI           | Return on investment (Revenue ÷ Spend)         |

### Retail Dataset (retail_data.csv)
| Column Name   | Description                                    |
|---------------|------------------------------------------------|
| Transaction ID| Unique transaction identifier                  |
| Date          | Transaction date                               |
| Store         | Store location                                 |
| Product       | Product name                                   |
| Category      | Product category                               |
| Quantity      | Units sold                                     |
| Price         | Price per unit                                 |
| Sales         | Total sales amount                             |
| Discount      | Discount applied                               |
| Net Sales     | Sales − Discount                               |

### Ecommerce Dataset (ecommerce_data.csv)
| Column Name       | Description                                |
|-------------------|--------------------------------------------|
| Order ID          | Unique order identifier                    |
| Order Date        | Date of order                              |
| Customer ID       | Unique customer identifier                 |
| Customer Segment  | Segment (e.g., New, Returning, VIP)        |
| Product           | Product name                               |
| Category          | Product category                           |
| Quantity          | Units ordered                              |
| Order Value       | Total order value                          |
| Payment Method    | Credit Card, PayPal, etc.                  |
| Delivery Time     | Days taken for delivery                    |
| Return Status     | Returned (Yes/No)                          |

---

## 🚀 Deployment on Streamlit Cloud

1. Push this repository to GitHub.
2. Connect it to [Streamlit Cloud](https://streamlit.io/cloud).
3. Set the entry point to `master_pipeline.py`.
4. Add secrets (optional) for API keys:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ANTHROPIC_API_KEY = "..."
   COHERE_API_KEY = "..."
   LLM_PROVIDER = "huggingface"

