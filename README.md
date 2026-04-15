# 🚺 SafeHer: Women Safety Analysis and Safe Route Recommendation System

SafeHer is a data-driven web application designed to enhance women’s safety by analyzing crime data, identifying high-risk zones, and recommending safer routes using intelligent risk analysis.

---
 
## 📌 Project Overview

The system analyzes crime datasets and transforms raw data into meaningful safety insights.
It uses spatial and temporal analysis to identify unsafe areas and provide users with safer navigation options. 
The application also includes explainable AI features to help users understand why certain areas or routes are considered risky.

---

## 🎯 Key Features

- 🔍 **Crime Data Analysis** – Processes crime datasets to detect unsafe zones
- 🗺️ **Risk Heatmap Visualization** – Displays high, medium, and low-risk areas
- ⏱️ **Time-Based Risk Analysis** – Identifies high-risk periods (day/night)
- 📊 **Safety Score Calculation** – Assigns risk scores to locations
- 🧭 **Safe Route Finder** – Suggests routes with minimal exposure to risk
- 💡 **Explainable AI** – Provides reasons behind unsafe routes and zones
- 🌐 **Web Dashboard** – User-friendly interface for interaction

---

## 🏗️ System Architecture

The system follows a structured pipeline:

1. Load Crime Dataset (CSV)
2. Data Cleaning & Preprocessing
3. Risk Score Calculation
4. Time-Based Filtering
5. Heatmap Generation (using Folium)
6. Safe Route Generation
7. Explainable AI (Risk Explanation)
8. Display on Web Dashboard

---

## 🛠️ Technologies Used

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python (Flask)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Folium (Heatmaps)
- **Machine Learning (basic):** Risk scoring logic

---

## 📂 Project Structure
SafeHer/
│── app.py
│── templates/
│── static/
│── data/
│── models/
│── utils/
│── requirements.txt
│── README.md


## 📊 Dataset

- Crime dataset used is in CSV format  
- Contains location coordinates, crime type, and time information  
- Currently uses sample/collected data due to limited availability of official datasets  

---

## 📸 Screenshots

<img width="975" height="460" alt="image" src="https://github.com/user-attachments/assets/1c5d69e3-be72-4359-bf8d-09ae5e770d75" />
<img width="975" height="542" alt="image" src="https://github.com/user-attachments/assets/04b4c891-a0f0-4993-8be0-407227e9ee80" />
<img width="975" height="460" alt="image" src="https://github.com/user-attachments/assets/24f0292e-53c8-4693-91f8-54b22996492b" />
<img width="975" height="463" alt="image" src="https://github.com/user-attachments/assets/d7fd09ac-8bc7-4d4c-a149-5a9a615a2d92" />
<img width="975" height="438" alt="image" src="https://github.com/user-attachments/assets/b21ffb65-5521-49a0-bd42-1ef52a8330c9" />
<img width="975" height="463" alt="image" src="https://github.com/user-attachments/assets/e79327f9-c397-4ce3-99fe-5d07c3891da8" />
<img width="975" height="456" alt="image" src="https://github.com/user-attachments/assets/4dc0fd9a-5792-445f-9a93-193e7d818cae" />
<img width="576" height="1262" alt="image" src="https://github.com/user-attachments/assets/44e3078f-dcf3-4d1c-8400-252dc3c9a5e6" />


## ⚠️ Limitations

- Does not support real-time crime data  
- Depends on availability and accuracy of dataset  
- No user authentication system implemented yet  
- Limited to basic route analysis (no live GPS integration)  

---

## 🚀 Future Scope

- Integration with real-time crime APIs  
- GPS-based live tracking and alerts  
- User authentication and personalization  
- Mobile application support  
- Integration with official police/government datasets  
- Advanced machine learning models for prediction  

---

## 👩‍💻 Contributors
- Liz Mary Abraham(MGP23CS090)
- Neha Ann Bijoy(MGP23CS107)
- Neha Merin Biji(MGP23CS108)
- Sara Elena Saji(MGP23CS121)
---

## 📜 License

This project is developed for academic purposes.

