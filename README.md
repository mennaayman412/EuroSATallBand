# 🌍 Land Use & Land Cover Classification with EuroSAT

## Background  

This project was developed as part of the **Digital Egypt Pioneers Initiative (DEPI) – [مبادرة رواد مصر الرقمية](https://depi.gov.eg/content/home)** under the **AI & Data Science track**.  
It represents the culmination of a **6-month intensive training program**, combining advanced coursework with practical, real-world applications in artificial intelligence and data science.  

---
> [Bussiness Understanding](business_understanding.ipynb)

## Problem  

Land use and land cover (LULC) classification is critical for agriculture, urban planning, and environmental monitoring. Using **Sentinel-2 imagery** and the **EuroSAT dataset** (27k patches, 10 classes), we frame the task as **supervised multi-class image classification**.  

The model learns spectral + spatial patterns with CNNs (and experiments with ViTs) to automatically classify land cover types.

---

## Objectives  

- Preprocess EuroSAT (RGB & multispectral) with normalization + augmentation  
- Train CNNs with transfer learning for 3-band and 13-band inputs  
- Evaluate performance (accuracy, F1-score, confusion matrix)  
- Compare CNN vs. ViT, RGB vs. multispectral  
- Demonstrate applications in sustainability, agriculture, and urban growth monitoring  

---

## Applications  

- 🌾 Agriculture – crop health & yield  
- 🏙️ Urban growth & real estate  
- 🚉 Transportation networks  
- ⚡ Energy & resource monitoring  
- 🌳 Environmental tracking  
- 🛒 Commercial & insurance analytics  

---

## Dataset  

- **EuroSAT (Sentinel-2 imagery)** – 64×64 patches, 10 land cover classes  
- [Download from Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset?resource=download)

---

## Contributors  

- [Islam Gomaa](https://github.com/iislamgom3a)  
- Menna Allah Ayman
- Manar Saber  
- Mina Emad  
- Menna Alaa  
- Abdelrahman Afify  
