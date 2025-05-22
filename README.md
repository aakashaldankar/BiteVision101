# 🍔 BiteVision101: Food Image Classification Model

BiteVision101 is your **AI-powered food identifier**! Just upload a picture of any dish—be it a delicious burger, a slice of pizza, or a fresh salad—and BiteVision101 will tell you what it is!

This project uses **EfficientNetB2**, a state-of-the-art computer vision model, fine-tuned on the **Food101 dataset** which contains images of **101 different food categories**. The application is hosted with a simple and beautiful **Gradio web interface**.

HuggingFace Space Link: https://huggingface.co/spaces/aakashaldankar/BiteVision101

---

## 🚀 Demo Snapshot

Run the app and get predictions like this:
- 🍕 **Pizza**: 94%
- 🥗 **Greek Salad**: 82%
- 🍔 **Hamburger**: 88%

---

## 📁 Project Structure

```bash
.
├── examples/                   # Sample food images to test the app
│   ├── food1.jpg
│   └── food2.jpg
├── app.py                     # Gradio app entry point
├── bitevision_model.py        # EfficientNetB2 model definition
├── BiteVision101_e20.pth      # Fine-tuned model weights (42nd epoch)
├── requirements.txt           # Required Python packages
└── README.md                  # You're here!
```

---

## 📦 Installation & Running the App

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/aakashaldankar/BiteVision101.git
cd BiteVision101
```

### 2️⃣ (Optional but Recommended) Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App

```bash
python app.py
```

The app will launch in your browser with the Gradio interface.

---

## 🔍 How It Works

1. The user uploads or selects an image from the `examples/` folder.
2. The image is transformed using EfficientNetB2 preprocessing.
3. The image is classified into one of the 101 food categories.
4. The top 3 predictions and the inference time are displayed.

---

## 💡 Why This Project Matters

- 🍽️ **Real-world Applications**:
  - Restaurant apps for dish recognition.
  - Calorie-tracking and food-logging apps.
  - Food delivery platforms for enhancing search and filter options.

- 🧠 **Educational Value**:
  - Great starting point for understanding **transfer learning**, **image classification**, and **Gradio UIs**.
  - Showcase project for portfolios.

---

## 🧠 Model Details

- **Backbone**: EfficientNetB2 (pre-trained on ImageNet)
- **Fine-tuned on**: Food101 dataset
- **Parameters**: ~7.8 million
- **Accuracy**: ~85% on test set

---

## 📸 Example Foods It Can Recognize

`pizza`, `sushi`, `hamburger`, `pad thai`, `falafel`, `waffles`, `ice cream`, `steak`, `ramen`, `spaghetti bolognese`, `tacos`, and many more!

---

## 🙌 Contributions & Feedback

Got suggestions or want to improve the model? Feel free to open an issue or submit a pull request!
Or reachout me at: aakashaldankar@gmail.com


---

## 📃 License

This project is open-source and available under the GPL-2.0 License.
