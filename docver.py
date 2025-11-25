#!/usr/bin/env python3
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from pdf2image import convert_from_path
import requests
from bs4 import BeautifulSoup
import pytesseract
import pandas as pd
from rapidfuzz import fuzz
import os
import datetime
import argparse
import re
from PIL import Image

# --- Configuration ---
# Path to Tesseract executable if not in PATH (Uncomment and set if needed)
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def convert_to_opencv(image):
    """Converts a PIL Image or numpy array to an OpenCV BGR image."""
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        # Assume it's already BGR if it's a numpy array from cv2.imread, 
        # but if it came from elsewhere it might need checking. 
        # For now, we assume standard pipeline usage.
        return image
    return None

def extract_qr_link(image_input, page_num=1):
    """
    Extracts the link from a QR code in a single image.
    image_input: OpenCV image (numpy array) or PIL Image
    """
    print(f"[*] Scanning for QR code on Page {page_num}...")
    
    img = convert_to_opencv(image_input)
    if img is None:
        print(f"[!] Invalid image input for Page {page_num}")
        return None

    def try_decode(image):
        decoded_objects = decode(image)
        for obj in decoded_objects:
            data = obj.data.decode('utf-8')
            if data.startswith('http'):
                return data
        return None

    # Attempt 1: Original Image
    link = try_decode(img)
    if link:
        print(f"[+] Found QR Link (Original): {link}")
        return link

    # Attempt 2: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    link = try_decode(gray)
    if link:
        print(f"[+] Found QR Link (Grayscale): {link}")
        return link

    # Attempt 3: Thresholding (Binary)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    link = try_decode(thresh)
    if link:
        print(f"[+] Found QR Link (Threshold): {link}")
        return link
        
    # Attempt 4: Otsu's Thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    link = try_decode(otsu)
    if link:
        print(f"[+] Found QR Link (Otsu): {link}")
        return link

    # Attempt 5: Resize (Upscale) - helpful for small QRs
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    link = try_decode(resized)
    if link:
        print(f"[+] Found QR Link (Upscaled): {link}")
        return link
        
    # Attempt 6: Resize + Threshold
    _, resized_thresh = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY)
    link = try_decode(resized_thresh)
    if link:
        print(f"[+] Found QR Link (Upscaled+Threshold): {link}")
        return link

    print(f"[-] No QR code link found on Page {page_num}.")
    return None

def scrape_web_data(url):
    """
    Scrapes key details (Name, Roll No, DOB, Total Marks, Father Name, School Name, Perm Reg No) from the URL.
    """
    print(f"[*] Scraping data from: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[!] Error fetching URL: {e}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    data = {}
    
    # Extract Roll No from URL if possible (Reliable fallback)
    try:
        parts = url.split('/')
        for part in parts:
            if part.isdigit() and len(part) > 5:
                data['Roll No'] = part
                break
    except:
        pass

    text = soup.get_text(separator='\n')
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Keyword mapping (Broader)
    keyword_map = {
        'Name of the Candidate': 'Name',
        'Candidate Name': 'Name',
        'Father Name': 'Father Name', 
        'Mother & Father Name': 'Father Name',
        'NAME OF THE SCHOOL': 'School Name',
        'பள்ளியின் பெயர்': 'School Name',  # Tamil for School Name
        'PERMANENT REGISTER NO': 'Perm Reg No',
        'நிரந்தர பதிவெண்': 'Perm Reg No',  # Tamil for Perm Reg No
        'Name': 'Name',
        'ROLL NO': 'Roll No',
        'Roll Number': 'Roll No',
        'தேர்வெண்': 'Roll No',  # Tamil for Roll No
        'DATE OF BIRTH': 'DOB',
        'Date of Birth': 'DOB',
        'பிறந்ததேதி': 'DOB',  # Tamil for DOB
        'TOTAL MARKS': 'Total Marks',
        'Total Marks': 'Total Marks',
        'Grand Total': 'Total Marks',
        'Result': 'Result'
    }

    for i, line in enumerate(lines):
        line_upper = line.upper()
        for key_search, key_out in keyword_map.items():
            if key_search.upper() in line_upper:
                if key_out in data:
                    continue

                # Special check: If looking for "Name", ensure we aren't matching "Father Name" or "School Name"
                if key_search == 'Name':
                    if 'FATHER' in line_upper or 'SCHOOL' in line_upper:
                        continue
                
                # Strategy 1: Value on the same line
                if ':' in line:
                    parts = line.split(':', 1)
                    if key_search.upper() in parts[0].upper():
                        if key_search == 'Name':
                            if 'FATHER' in parts[0].upper() or 'SCHOOL' in parts[0].upper():
                                continue
                        val = parts[1].strip()
                        if val:
                            # For DOB, extract just the date part
                            if key_out == 'DOB':
                                date_match = re.search(r'\d{2}[/-]\d{2}[/-]\d{4}', val)
                                if date_match:
                                    data[key_out] = date_match.group(0)
                                    continue
                            else:
                                data[key_out] = val
                                continue
                
                # Strategy 1b: Value on same line without colon (e.g., "DATE OF BIRTH 30/10/2004")
                if key_out == 'DOB' and 'DOB' not in data:
                    date_match = re.search(r'\d{2}[/-]\d{2}[/-]\d{4}', line)
                    if date_match:
                        data[key_out] = date_match.group(0)
                        continue

                # Strategy 2: Look ahead for value
                # Increased lookahead for fields that appear further from their labels
                if key_out == 'Name':
                    lookahead_range = 10
                elif key_out in ['DOB', 'School Name']:
                    lookahead_range = 12  # These can be far from their labels in 12th marksheets
                else:
                    lookahead_range = 6
                
                for offset in range(1, lookahead_range):
                    if i + offset < len(lines):
                        val = lines[i+offset].strip()
                        if not val: continue
                        
                        # Stop if we hit another keyword (conservative)
                        # Exception: "Father Name" might be close to "Name"
                        if any(k.upper() == val.upper() for k in keyword_map.keys()): break
                        
                        if key_out == 'Name':
                            if re.search(r'[a-zA-Z]', val) and len(val) > 2:
                                if 'SESSION' in val.upper() or 'FATHER' in val.upper() or 'MOTHER' in val.upper() or 'SCHOOL' in val.upper():
                                    continue
                                data[key_out] = val
                                break
                        elif key_out == 'Father Name':
                             if re.search(r'[a-zA-Z]', val) and len(val) > 2:
                                data[key_out] = val
                                break
                        elif key_out == 'School Name':
                            # Skip Tamil text and label lines (containing '/')
                            # Look for lines with predominantly English alphabetic characters
                            if '/' in val or len(val) < 5:
                                continue
                            # Count English letters
                            english_letters = sum(1 for c in val if c.isascii() and c.isalpha())
                            if english_letters > 10 and re.search(r'[A-Z]', val):
                                data[key_out] = val
                                break
                        elif key_out == 'Roll No':
                            # 12th: Roll No might be in format "5094556 / MAY / 2022"
                            if '/' in val:
                                parts = val.split('/')
                                for p in parts:
                                    p = p.strip()
                                    if p.isdigit() and len(p) > 5:
                                        data[key_out] = p
                                        break
                                if key_out in data: break
                            
                            if re.search(r'\d', val):
                                data[key_out] = val
                                break
                        elif key_out == 'DOB':
                            # Look for date patterns: DD/MM/YYYY or DD-MM-YYYY
                            date_match = re.search(r'\d{2}[/-]\d{2}[/-]\d{4}', val)
                            if date_match:
                                data[key_out] = date_match.group(0)
                                break
                        elif key_out == 'Total Marks':
                            # 12th marksheet has 3-4 digit totals (e.g., 0375)
                            # Strategy: For grand total, look for pattern "0375 ZERO THREE SEVEN FIVE"
                            # This distinguishes it from individual subject marks
                            # First check if this line has spelled-out numbers
                            number_words = ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE']
                            if any(word in val.upper() for word in number_words):
                                # This line contains the spelled out number, extract the digits
                                match = re.search(r'\b0*(\d{3,4})\b', val)
                                if match:
                                    # Extract the number and remove leading zeros
                                    total = match.group(1).lstrip('0') or '0'
                                    data[key_out] = total
                                    break  # Found grand total with spelled words, stop looking
                            # Only accept plain numbers if we haven't found a spelled-out version yet
                            # and we're not already processing a higher value
                            if key_out not in data or 'ZERO' not in str(data.get(key_out, '')):
                                match = re.search(r'\b\d{3,4}\b', val)
                                if match:
                                    candidate = match.group(0).lstrip('0') or '0'
                                    # Keep the larger value (grand total is bigger than subject marks)
                                    if key_out in data:
                                        try:
                                            if int(candidate) > int(data[key_out]):
                                                data[key_out] = candidate
                                        except ValueError:
                                            pass
                                    else:
                                        # Only set if it's a reasonable total (> 100 for 12th)
                                        if int(candidate) > 100:
                                            data[key_out] = candidate
            
            # Special handling for fields that appear BEFORE their labels
            
            # Perm Reg No lookbehind
            if key_out == 'Perm Reg No' and key_out not in data:
                 # Look backwards 5 lines
                 for offset in range(1, 6):
                    if i - offset >= 0:
                        val = lines[i-offset].strip()
                        if not val: continue
                        # Pattern: alphanumeric, length > 8, MUST contain digits (10th and 12th)
                        if re.search(r'^[A-Z0-9]{8,}$', val) and re.search(r'\d', val) and not re.search(r'^\d{2}/\d{2}/\d{4}', val):
                             data[key_out] = val
                             break
            
            # DOB lookbehind (for 12th marksheets where DOB appears before its label)
            if key_out == 'DOB' and key_out not in data:
                 # Look backwards 10 lines
                 for offset in range(1, 11):
                    if i - offset >= 0:
                        val = lines[i-offset].strip()
                        if not val: continue
                        # Pattern: DD/MM/YYYY or DD-MM-YYYY
                        date_match = re.search(r'^\d{2}[/-]\d{2}[/-]\d{4}$', val)
                        if date_match:
                             data[key_out] = date_match.group(0)
                             break
    
    # Try to extract Year/Session if not found
    if 'Year' not in data:
        year_keywords = ['Year of Passing', 'Session', 'Examination Year', 'Year']
        for line in lines:
            for yk in year_keywords:
                if yk.upper() in line.upper():
                    # Look for 4 digit year
                    years = re.findall(r'\b20\d{2}\b', line)
                    if years:
                        data['Year'] = years[0]
                        break
            if 'Year' in data: break
    
    # Fallback for Total Marks
    if 'Total Marks' not in data:
        numbers = []
        for line in lines:
            nums = re.findall(r'\b\d{3,4}\b', line)
            for n in nums:
                if 100 < int(n) <= 600:  # 12th is out of 600 usually
                    numbers.append(int(n))
        if numbers:
            data['Total Marks'] = str(max(numbers))

    print(f"[+] Scraped Data: {data}")
    return data

def extract_ocr_text(image_input, page_num=1):
    """
    Extracts raw text from a single image using OCR.
    """
    print(f"[*] Performing OCR on Page {page_num}...")
    
    img = convert_to_opencv(image_input)
    if img is None: return ""

    full_text = ""
    
    # Define preprocessing pipelines
    pipelines = []
    
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pipelines.append(("Grayscale", gray))
    
    # 2. Otsu Thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    pipelines.append(("Otsu", otsu))
    
    # 3. Adaptive Thresholding
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    pipelines.append(("Adaptive", adaptive))
    
    # 4. Upscaling
    upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    pipelines.append(("Upscaled", upscaled))

    # Run OCR on all pipelines
    print(f"    Running multi-stage OCR...")
    
    for name, processed_img in pipelines:
        text = pytesseract.image_to_string(processed_img)
        full_text += text + "\n"
        
        if len(text) < 100:
            text_psm6 = pytesseract.image_to_string(processed_img, config='--psm 6')
            full_text += text_psm6 + "\n"

    # Normalize text
    full_text = full_text.replace('\n', ' ').replace('  ', ' ')
    
    print(f"[+] OCR Text Length: {len(full_text)} chars")
    return full_text

def verify_document(web_data, ocr_text):
    """
    Verifies the document by checking if Web Data exists in OCR Text.
    """
    print("[*] Verifying Document...")
    
    score = 0
    max_score = 0
    details = {}
    
    # Initialize details with N/A
    fields = ['Roll No', 'Name', 'DOB', 'Total Marks', 'Father Name', 'School Name', 'Perm Reg No']
    for f in fields:
        details[f] = {
            "Web Value": "N/A",
            "OCR Status": "N/A"
        }

    if not web_data:
        return 0, "FAKE (No Web Data)", details
    
    def is_garbage_text(text):
        """
        Checks if the text is likely garbage/encoding error.
        Heuristic: High ratio of non-ASCII or specific garbage characters.
        """
        if not text: return False
        # Common garbage chars seen: #, ‹, Â, ¥, °, £, etc.
        garbage_chars = '#‹Â¥°£'
        non_ascii_count = sum(1 for c in text if ord(c) > 127 or c in garbage_chars)
        ratio = non_ascii_count / len(text)
        return ratio > 0.3

    # Helper to update details
    def check_field(field_name, weight, fuzzy=False, strategy="default"):
        nonlocal score, max_score
        val = web_data.get(field_name)
        
        if val:
            # Special check for School Name encoding issues
            if field_name == 'School Name' and is_garbage_text(val):
                details[field_name] = {
                    "Web Value": val,
                    "OCR Status": "SKIPPED (Encoding Issue)"
                }
                # Do not add to max_score, effectively neutralizing this field
                return

            max_score += weight
            status = "NOT FOUND"
            match_score = 0
            
            if strategy == "best_word_match":
                # Split OCR text into words and find best match
                words = ocr_text.split()
                best_ratio = 0
                for word in words:
                    # Clean word slightly? Maybe not needed if fuzz handles it
                    r = fuzz.ratio(val, word)
                    if r > best_ratio:
                        best_ratio = r
                
                match_score = best_ratio
                if match_score > 85: # High threshold for ID
                    score += weight
                    status = f"MATCHED ({match_score:.2f}%)"
                else:
                    status = f"MISMATCH ({match_score:.2f}%)"

            elif fuzzy:
                match_score = fuzz.token_set_ratio(val, ocr_text)
                if match_score > 70:
                    score += weight
                    status = f"MATCHED ({match_score:.2f}%)"
                else:
                    status = f"MISMATCH ({match_score:.2f}%)"
            else:
                # Exact(ish) match
                clean_ocr = ocr_text.replace(' ', '')
                if val in ocr_text or val in clean_ocr:
                    score += weight
                    status = "MATCHED"
                else:
                    status = "NOT FOUND"
            
            details[field_name] = {
                "Web Value": val,
                "OCR Status": status
            }
        # Else keep the N/A default

    # 1. Verify Roll No (Critical)
    check_field('Roll No', 30, fuzzy=False)

    # 2. Verify Name (Secondary)
    check_field('Name', 20, fuzzy=True)

    # 3. Verify DOB
    check_field('DOB', 10, fuzzy=False)

    # 4. Verify Total Marks
    check_field('Total Marks', 10, fuzzy=False)
    
    # 5. Verify Father Name
    check_field('Father Name', 10, fuzzy=True)
    
    # 6. Verify School Name
    check_field('School Name', 10, fuzzy=True)
    
    # 7. Verify Perm Reg No
    check_field('Perm Reg No', 10, strategy="best_word_match")

    # Calculate percentage
    final_score = (score / max_score) * 100 if max_score > 0 else 0
    
    # Verdict
    verdict = "FAKE"
    if final_score >= 90:
        verdict = "ORIGINAL"
    elif final_score >= 60:
        verdict = "SUSPICIOUS (Partial Match)"

    # CRITICAL CHECK: Total Marks
    # If Total Marks was present in Web Data but NOT matched in OCR, downgrade verdict
    tm_detail = details.get('Total Marks', {})
    if tm_detail.get('Web Value') != "N/A":
        tm_status = tm_detail.get('OCR Status', 'N/A')
        if "MATCHED" not in tm_status:
             if verdict == "ORIGINAL":
                 verdict = "SUSPICIOUS (Total Marks Mismatch)"
             # If it was already SUSPICIOUS or FAKE, we leave it, or we could append the reason
             elif "SUSPICIOUS" in verdict and "Total Marks Mismatch" not in verdict:
                 verdict += " + Total Marks Mismatch"
    
    return final_score, verdict, details

def log_to_csv(result_data, filename="verification_log.csv"):
    file_exists = os.path.isfile(filename)
    
    # Flatten web_data if present in result_data
    # We expect result_data to potentially have a 'web_data' key which is a dict
    # We want to promote its keys to top-level keys with a prefix
    
    flat_data = result_data.copy()
    if 'web_data' in flat_data:
        web_data = flat_data.pop('web_data')
        if web_data:
            for k, v in web_data.items():
                flat_data[f"Web_{k}"] = v
    
    df = pd.DataFrame([flat_data])
    
    # If file exists, we need to check if columns match. 
    # If new columns are introduced, we might need to handle it. 
    # For simplicity, we'll append. Pandas handles missing columns by adding NaNs if we read and write, 
    # but mode='a' with header=False assumes same columns.
    # To be safe, if file exists, we read it, concat, and write back.
    
    if not file_exists:
        df.to_csv(filename, index=False)
    else:
        # Read existing to get columns
        try:
            existing_df = pd.read_csv(filename)
            # Concat
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(filename, index=False)
        except Exception as e:
            print(f"[!] Error updating CSV: {e}")
            # Fallback to append if read fails (might be empty or corrupt)
            df.to_csv(filename, mode='a', header=False, index=False)

    print(f"[+] Logged result to {filename}")

def main():
    parser = argparse.ArgumentParser(description="10th Marksheet Verification System")
    parser.add_argument("--input", required=True, help="Path to the image or PDF file")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"[!] File not found: {input_path}")
        return

    images = []
    if input_path.lower().endswith('.pdf'):
        print(f"[*] Converting PDF to images: {input_path}")
        try:
            images = convert_from_path(input_path)
        except Exception as e:
            print(f"[!] Error converting PDF: {e}")
            return
    else:
        img = cv2.imread(input_path)
        if img is None:
            print(f"[!] Could not read image: {input_path}")
            return
        images = [img]

    print(f"[*] Total Pages to Process: {len(images)}")

    for i, img in enumerate(images):
        page_num = i + 1
        print(f"\n--- Processing Page {page_num} ---")
        
        # 1. Extract QR Link
        qr_link = extract_qr_link(img, page_num)
        if not qr_link:
            print(f"[-] Skipping Page {page_num}: No QR Code found.")
            continue

        # 2. Scrape Web Data
        web_data = scrape_web_data(qr_link)
        
        # Display Extracted Data
        print("\n" + "="*30)
        print(f"EXTRACTED WEB DATA (Page {page_num}):")
        if web_data:
            for k, v in web_data.items():
                print(f"  {k}: {v}")
        else:
            print("  No data extracted.")
        print("="*30 + "\n")
        
        # 3. Extract OCR Text
        ocr_text = extract_ocr_text(img, page_num)
        
        # 4. Verify
        score, verdict, details = verify_document(web_data, ocr_text)
        
        print("\n" + "="*60)
        print(f"VERIFICATION RESULT (Page {page_num}): {verdict}")
        print(f"Confidence Score: {score:.2f}%")
        print("="*60)
        print(f"{'Field':<15} | {'Web Data':<30} | {'OCR Comparison':<20}")
        print("-" * 70)
        for field, info in details.items():
            print(f"{field:<15} | {str(info['Web Value']):<30} | {info['OCR Status']:<20}")
        print("="*60)
        
        # Display OCR Text
        print("\n" + "="*30)
        print(f"OCR EXTRACTED TEXT (Page {page_num}):")
        print("-" * 30)
        print(ocr_text.strip())
        print("="*30 + "\n")

        # 5. Log
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "file_path": input_path,
            "page": page_num,
            "qr_link": qr_link,
            "verdict": verdict,
            "score": score,
            "details": str(details)
        }
        log_to_csv(log_entry)

if __name__ == "__main__":
    main()
