import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
import docver
import tempfile
import os
import datetime
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Document Verification System",
    page_icon="🔍",
    layout="wide"
)

LOG_FILE = "verification_log.csv"

def main():
    st.title("🔍 Document Verification System")
    st.markdown("Upload a document (Image or PDF) to verify its authenticity.(only 10 marksheet support now)")

    # Sidebar for configuration or info
    with st.sidebar:
        st.header("About")
        st.info("This system scans for a QR code, scrapes the linked data, performs OCR, and verifies the document content.")
        st.markdown("---")
        st.markdown("**Supported Formats:** JPG, PNG, PDF")
        
        st.markdown("### 📥 Logs")
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "rb") as f:
                st.download_button(
                    label="Download Verification Logs (CSV)",
                    data=f,
                    file_name="verification_log.csv",
                    mime="text/csv"
                )
        else:
            st.text("No logs available yet.")

    # --- Main Content ---
    
    # 1. Document Configuration
    with st.expander("Supported Doc Type", expanded=False):
        st.markdown("""
        **Categories:**
        - ✅ **Supported**: 10th Marksheet (2020 onwards)
        - ⚠️ **Non-Support**: 10th Marksheet (Before 2020)
        - 🚧 **Future Update**: 11th & 12th Marksheets
        """)
        
        col_config1, col_config2 = st.columns(2)
        
        with col_config1:
            doc_type = st.selectbox(
                "Select Document Type",
                ["10th Marksheet", "11th Marksheet", "12th Marksheet"]
            )
            
        with col_config2:
            # Default to current year
            current_year = datetime.datetime.now().year
            selected_year = st.number_input("Select Year of Passing", min_value=2000, max_value=current_year + 1, value=current_year)

    # Determine Support Status
    is_supported = False
    status_message = ""
    status_color = "gray"
    
    if doc_type == "10th Marksheet":
        if selected_year >= 2020:
            is_supported = True
            status_message = "✅ Supported Document"
            status_color = "green"
        else:
            is_supported = False
            status_message = "⚠️ Non-Support Document (Year < 2020)"
            status_color = "orange"
    else:
        is_supported = False
        status_message = "🚧 Future Update (Coming Soon)"
        status_color = "blue"

    # Display Status
    st.markdown(f"**Status:** :{status_color}[{status_message}]")
    
    if "Future Update" in status_message:
        st.info(f"Support for {doc_type} is coming in a future update.")
        return # Stop execution for future updates

    st.divider()

    # 2. File Uploader
    st.subheader("2. Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg', 'pdf'])

    if uploaded_file is not None:
        # Determine file type and process
        file_type = uploaded_file.type
        
        images = []
        
        with st.spinner("Processing file..."):
            if "pdf" in file_type:
                # Convert PDF to images
                try:
                    images = convert_from_bytes(uploaded_file.read())
                    st.success(f"Loaded PDF with {len(images)} pages.")
                except Exception as e:
                    st.error(f"Error converting PDF: {e}")
                    return
            else:
                # Read image
                image = Image.open(uploaded_file)
                images = [image]
                st.success("Loaded Image.")

        # Process each page
        for i, img in enumerate(images):
            page_num = i + 1
            st.markdown(f"### 📄 Page {page_num}")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(img, caption=f"Page {page_num}", use_container_width=True)

            with col2:
                # Verify Button Logic
                verify_btn_disabled = False
                verify_btn_help = ""
                
                if not is_supported:
                    # User said "only verify now" for supported, but also "10 less then 2020 marksheet is non support field"
                    # and "in gui show only 10 marksheet above 2020 only verify now"
                    # This implies we should DISABLE verify for non-supported docs?
                    # "show only 10 marksheet above 2020 only verify now" -> This phrasing suggests only these should have the "Verify Now" option enabled?
                    # Or maybe it means "Only verify now" is the text? 
                    # I will assume we DISABLE verification for non-supported documents to be safe and strictly follow "only verify now" for > 2020.
                    verify_btn_disabled = True
                    verify_btn_help = "Verification is not supported for this document type/year."

                if st.button(f"Verify Page {page_num}", key=f"verify_{page_num}", disabled=verify_btn_disabled, help=verify_btn_help):
                    
                    # Container for process logs (hidden by default unless expanded, but we want to hide it completely or put in expander)
                    # User said "hide a all process in drop down"
                    
                    with st.status("Verifying...", expanded=False) as status:
                        # 1. Extract QR Link
                        status.write("Scanning for QR Code...")
                        qr_link = docver.extract_qr_link(img, page_num)
                        
                        if not qr_link:
                            status.update(label="No QR Code found.", state="error")
                            st.error("No QR Code found on this page.")
                            continue # Skip rest
                        
                        status.write(f"QR Link Found: {qr_link}")
                        
                        # 2. Scrape Web Data
                        status.write("Scraping Web Data...")
                        web_data = docver.scrape_web_data(qr_link)
                        
                        if not web_data:
                            status.update(label="Failed to scrape data.", state="error")
                            st.error("Failed to scrape data from the link.")
                            continue
                        
                        # 3. Extract OCR Text
                        status.write("Performing OCR...")
                        ocr_text = docver.extract_ocr_text(img, page_num)
                        
                        # 4. Verify
                        status.write("Comparing Data...")
                        score, verdict, details = docver.verify_document(web_data, ocr_text)
                        
                        status.update(label="Verification Complete!", state="complete")

                    # Display Result Prominently
                    st.divider()
                    st.subheader("Verification Result")
                    
                    if verdict == "ORIGINAL":
                        st.success(f"**Verdict: {verdict}**")
                    elif "SUSPICIOUS" in verdict:
                        st.warning(f"**Verdict: {verdict}**")
                    else:
                        st.error(f"**Verdict: {verdict}**")
                        
                    st.progress(score / 100, text=f"Confidence Score: {score:.2f}%")
                    
                    # Log to CSV
                    log_entry = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "file_path": uploaded_file.name, # Use filename
                        "page": page_num,
                        "qr_link": qr_link,
                        "verdict": verdict,
                        "score": score,
                        "web_data": web_data, # Pass the dict to be flattened
                        "details": str(details)
                    }
                    docver.log_to_csv(log_entry, filename=LOG_FILE)
                    st.toast("Result logged to CSV.")

                    # Details Dropdown
                    with st.expander("View Detailed Verification Data"):
                        st.markdown("#### Scraped Web Data")
                        st.json(web_data)
                        
                        with st.expander("View OCR Extracted Text"):
                            st.text(ocr_text)
                        
                        st.markdown("#### Detailed Comparison")
                        # Create a DataFrame for better display
                        table_data = []
                        for field, info in details.items():
                            status_icon = "✅" if "MATCHED" in info['OCR Status'] else "❌"
                            if "SKIPPED" in info['OCR Status']:
                                status_icon = "⚠️"
                                
                            table_data.append({
                                "Field": field,
                                "Web Value": info['Web Value'],
                                "OCR Status": f"{status_icon} {info['OCR Status']}"
                            })
                        
                        st.table(table_data)

if __name__ == "__main__":
    main()
