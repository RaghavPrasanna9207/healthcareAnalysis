# Improved Healthcare Statistical Analysis System

**COVID-19 Survey Data Analysis with Enhanced Features**  
**Author:** [Your Name]  
**Date:** July 13, 2025  
**Version:** 2.0

---

## Overview

This project provides a robust, automated pipeline for advanced statistical analysis of COVID-19 survey data. It features object-oriented design, comprehensive data validation, error handling, modularity, and professional reporting. The system is designed for reproducibility, scalability, and actionable healthcare insights.

---

## Features

- Object-oriented architecture for maintainability
- Centralized configuration management
- Enhanced error handling and detailed logging
- Comprehensive data validation and quality checks
- Modular, reusable functions for analysis tasks
- Performance optimizations for large datasets
- Professional documentation and code comments
- Unit testing capabilities
- Export functionality for processed data and results

---

## Dataset

- **Records:** 1,000 (synthetic or real)
- **Variables:** 11
- **Period:** 2020-01-01 to 2023-12-31
- **Source:** COVID-19 Survey (Excel/Synthetic)
- **Data Types:** Numerical (e.g., temperature, IgG, IgM), Categorical (e.g., gender, age, blood group, COVID-19 status)

---

## Getting Started

### 1. Clone the Repository


### 2. Install Dependencies
pip install pandas numpy matplotlib seaborn scipy

### 3. Run the Analysis
python main.py


### 4. Review Outputs

All results, reports, and visualizations are saved in the `healthcare_analysis_results/` directory.

---

## Output Files

| File                      | Description                          |
|---------------------------|--------------------------------------|
| processed_data.csv        | Cleaned and processed dataset        |
| processed_data.pkl        | Pickled DataFrame for Python         |
| statistical_results.json  | Statistical analysis results         |
| data_quality_report.json  | Data quality summary                 |
| analysis_report.txt       | Full analysis report                 |
| *.png                     | Visualizations (demographics, stats) |

---

## Results Summary

- **Gender distribution:** Female (520), Male (480)
- **Most common age group:** 23-35
- **Average temperature:** 37.20°C
- **COVID-19 positive rate:** 25.2%
- **No significant difference** in temperature by gender or COVID-19 status by gender

For detailed findings and statistical analysis, see `analysis_report.txt` in the output directory.
