# 🍎 Automated Visual Inspection System for Fruits using ML & CV

This project presents an **Automated Visual Inspection System** designed to classify and sort fruits—**apples, bananas, and oranges**—into "Fresh" or "Rotten" categories using **computer vision** and a **robotic arm (MechARM 270 Pi)**. Developed as a practical solution to reduce human error in fruit quality control, the system leverages **image processing**, **machine learning (KNN)**, and **robotic motion** to automate the sorting process.

---

## 🛠 Requirements

> ⚠️ To successfully run and test this project, you will need **hardware equipment** alongside the software setup.

### 🔩 Hardware Components:
- **MechARM 270 Pi** with AI Kit 2023  
- **High-resolution Camera** with consistent lighting  
- **5 Sorting Bins**  
- **Suction Pump** (or adaptive gripper for better handling)  
- **Fruits** (apples, bananas, oranges)  

---

## 💻 Software Stack
- Python 3.x  
- OpenCV  
- Scikit-learn  
- pymycobot (for robotic arm communication)

---

## ⚙️ Setup Instructions

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Chaahna/Automated-Visual-Inspection-System-with-ML-and-CV
   cd Automated-Visual-Inspection-System-with-ML-and-CV
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Connect and Calibrate Hardware**
   - Mount the camera above the fruit placement area.
   - Connect the MechARM via USB or serial.
   - Ensure suction pump or gripper is responsive.

4. **Run the Program**
   Open and run the main script via Colab or locally.  
   👉 [Colab Notebook](https://colab.research.google.com/drive/1rSscGAKO2IbOKnbI6A0hnhJPWzhlltkP?usp=sharing)

---

## 🧪 Dataset

- Sourced from Kaggle: [Fruits Fresh and Rotten](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)  
- Categories: Fresh/Rotten Apples, Bananas, and Oranges  
- Images resized to 100x100 and normalized

---

## 📊 Model Overview

- **Classification Algorithm**: K-Nearest Neighbors (KNN), k=3  
- **Feature Extraction**:  
  - **Color** (HSV histograms)  
  - **Texture** (Laplacian variance)

---

## 🎥 Demo & Output

- 📹 [Watch Demo on YouTube](https://youtu.be/ubjg3KtsiEE)
- Snapshot below shows successful classification and robotic action in real time.

---

## 📄 Learn More

For detailed methodology, system architecture, results, and future enhancements, **please refer to the full project report PDF** available in this repository:  
📘 `Chaahna_Course Project Report CIS 496K.pdf`

---

## 🧠 Future Improvements
- Use adaptive grippers for better fruit handling  
- Explore deep learning models for higher classification accuracy  
- Expand dataset for more fruit categories  
