# 🔍 Document Verification System

A robust Streamlit application designed to verify the authenticity of educational documents (specifically 10th Marksheets) using QR code scanning, Web Scraping, and OCR technology.

## 🚀 Live Demo
**[Click here to view the Live App](https://ari-document-verfication.streamlit.app/)**

## 🌟 Features
- **QR Code Scanning**: Automatically detects and extracts links from QR codes on marksheets.
- **Data Verification**: Scrapes official data from the QR link and compares it with OCR-extracted text from the document.
- **OCR Integration**: Uses Tesseract OCR to extract text from images and PDFs.
- **Smart Filtering**: 
    - **Supported**: 10th Marksheets (2020 onwards).
    - **Non-Support**: Older documents (Before 2020).
    - **Future Update**: Placeholders for 11th & 12th marksheets.
- **Detailed Reporting**: Provides a confidence score, verdict (Original/Suspicious/Fake), and field-by-field comparison.
- **Logging**: Maintains a CSV log of all verification attempts.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AriProject9787/doc_verify
   cd antigravity
   ```

2. **Install dependencies:**
   Ensure you have Python installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: You also need `tesseract-ocr` and `poppler-utils` installed on your system.*

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📋 Usage
1. Open the app in your browser.
2. Expand the **"Supported Doc Type"** section to select the document type and year.
3. Upload a **PDF** or **Image** (JPG/PNG) of the marksheet.
4. If the document is supported, click **"Verify Page"**.
5. View the results, including the confidence score and detailed comparison table.

## 📂 Project Structure
- `app.py`: Main Streamlit application.
- `docver.py`: Core logic for QR scanning, scraping, OCR, and verification.
- `requirements.txt`: Python dependencies.
- `packages.txt`: System dependencies (for Streamlit Cloud).

## ⚠️ Note
Currently, full verification support is enabled only for **10th Marksheets from the year 2020 and onwards**.
