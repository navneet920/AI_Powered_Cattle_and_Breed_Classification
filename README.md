# 🐄 AI Powered Cattle and Breed Classification

An AI-powered deep learning system to detect cattle and accurately classify their breeds using **Transfer Learning (VGG16)**.  
The project achieves **~91% accuracy** by unfreezing the **last 30 layers of VGG16** and adding a custom classification head.  
This solution helps farmers, dairy companies, veterinary organizations, and AI researchers automate cattle breed identification.

---

## 🚀 Project Highlights

- ✔ Deep Learning–based breed classification  
- ✔ **Transfer Learning with VGG16 (ImageNet weights)**  
- ✔ **Unfrozen last 30 layers** for fine-tuning  
- ✔ Custom CNN classification head  
- ✔ **≈ 91% validation accuracy**  
- ✔ Interactive web app for predictions (`app.py`)  
- ✔ Full notebook for training the model  
- ✔ Clean dataset structure and preprocessing workflow  

---

## 📁 Repository Structure

├── README.md
├── LICENSE
├── app.py # Web interface for predictions
├── bovine-breed-classification-vgg16.ipynb # Model training notebook
├── models/ # Saved model weights
├── dataset/ # Train / Test / Validation dataset
└── utils/ # Helper scripts (optional)



---

## 🧠 Model Architecture  

### 1️⃣ Base Model: **VGG16**  
- Loaded with pre-trained ImageNet weights  
- Top layers removed (`include_top=False`)  
- All layers frozen initially  

### 2️⃣ Fine-Tuning  
After initial training:  
- **Last 30 layers** of VGG16 were **unfrozen**  
- Remaining layers kept frozen to avoid overfitting  
- Learning rate lowered for stability during fine-tuning  

### 3️⃣ Custom Classification Head  
Added on top of VGG16:  
- GlobalAveragePooling2D  
- Dense (512 units) + ReLU + Dropout(0.5)  
- Dense (128 units) + ReLU  
- Final Dense softmax layer for breed classification  

---

## 🎯 Performance  

- **Training Accuracy:** ~93%  
- **Validation Accuracy:** ~91%  
- **Test Accuracy:** ~90–92% (depending on dataset split)  

The model performs well in distinguishing visually similar breeds by leveraging VGG16’s deep convolutional features + fine-tuning.  
Sample confusion matrix and prediction examples can be added.

---

## 🛠️ How to Run the Project  

### Install Requirements  
```bash
pip install -r requirements.txt

python app.py

jupyter notebook bovine-breed-classification-vgg16.ipynb

The notebook covers:

- Data loading  
- Preprocessing  
- Transfer learning  
- Fine-tuning last 30 layers  
- Model saving  

---

## 🔮 Future Enhancements

- Add cattle **object detection** (YOLOv8 / Faster R-CNN)  
- Increase dataset size for rare breeds  
- Deploy complete full-stack version (React + Flask/Django)  
- Convert model to **TFLite** for mobile application  
- Add real-time webcam-based predictions  

---

## 🤝 Contributions

Contributions are welcome!  
You can help by:

- Adding more breeds  
- Improving UI  
- Optimizing training  
- Adding real-world farm photos for testing  

To contribute:  
1. Fork the repo  
2. Make changes  
3. Create a pull request  

---

## 📄 License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for details.

---

## 🙏 Acknowledgements

Special thanks to:  
- TensorFlow / Keras  
- Open datasets and cattle image repositories  
- Agricultural AI research communities  

---

If you want, I can also create:  
📌 `requirements.txt`  
📌 Project banner image  
📌 GIF showing predictions  
📌 More documentation (API, app workflow, dataset prep guide)

Just let me know!
