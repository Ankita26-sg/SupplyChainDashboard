# 📦 Supply Chain Analytics Dashboard

This project is an **interactive Supply Chain Analytics Dashboard** built using **Python and Streamlit**.  
It analyzes retail supply chain performance and provides insights into revenue, customers, logistics, supplier reliability, and delivery risks.

---

## 🧠 Key Research Questions
| Research Question | Page in Dashboard | Insight |
|---|---|---|
| **RQ1:** Which products / customer segments generate the highest revenue & profit? | Product Analysis | Identifies top revenue drivers |
| **RQ2:** Which suppliers perform best or worst? | Supplier Analysis | Examines lead time & defect indicators |
| **RQ3:** What factors influence late deliveries and shipping delays? | Logistics Analysis | Shows relationship between shipping time & risk |
| **RQ4:** How do shipping modes impact cost, profit, and delivery experience? | Logistics Analysis | Compares standard vs express shipping |
| **RQ5:** Can we predict if an order is likely to be returned/cancelled? | Diagnostic Analytics | Machine learning classification model |

---

## 🛠️ Technology Stack

| Tool / Library | Use Case |
|---|---|
| **Python** | Core programming |
| **Pandas / NumPy** | Data cleaning & transformation |
| **Plotly** | Interactive visualizations |
| **Streamlit** | Dashboard UI |
| **Scikit-learn** | Predictive modeling |
| **Git & GitHub** | Version control + sharing |

---

## 🚀 How to Run the Dashboard Locally

### 1️⃣ Clone this repository:
```bash
git clone https://github.com/Ankita26-sg/SupplyChainDashboard.git
cd SupplyChainDashboard
### 2️⃣ Install Required Libraries:
pip install -r requirements.txt
### If you ever need to regenerate the requirements file:
pip freeze > requirements.txt
### 3️⃣ Run the Streamlit Dashboard:
streamlit run Dashboard.py

## 📂 Project Structure
SupplyChainDashboard/
│
├── Dashboard.py                   # Main Streamlit dashboard application
├── DataCoSupplyChainDataset.csv   # Dataset used for analysis
├── requirements.txt               # Python dependencies list
└── README.md                      # Project documentation

## 📊 Dashboard Sections Overview
| Dashboard Page            | Purpose                                                      |
| ------------------------- | ------------------------------------------------------------ |
| **Home**                  | Project introduction & dataset upload option                 |
| **Dashboard (KPIs)**      | High-level revenue, profit & delivery performance indicators |
| **Data Overview**         | Data preview, missing values & statistical insights          |
| **Product Analysis**      | Top revenue products & category performance (RQ1)            |
| **Customer Segmentation** | Sales patterns across customer groups (RQ4)                  |
| **Supplier Analysis**     | Supplier lead time & quality performance (RQ2)               |
| **Logistics Analysis**    | Shipping time, delivery delays & risk patterns (RQ3)         |
| **Diagnostic Analytics**  | Predictive model to identify return/cancellation risk        |

## 👩‍💻 Author
Name: Ankita Singh
University: MBS Education
Course: Advanced Business Statistics
Year: 2025

Feel free to reach out for academic collaboration.

##📝 License
This project is for educational and academic use only.
Not intended for commercial use
