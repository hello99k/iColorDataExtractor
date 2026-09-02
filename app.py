import streamlit as st
import cv2
import pytesseract
import pandas as pd
import numpy as np
import re
import io

# IMPORTANT: Point this to your Tesseract installation path if running locally on Windows.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_data_from_image(image_bytes):
    """Decodes uploaded image bytes and extracts the WL and %R/%T table."""
    # Convert Streamlit uploaded bytes to OpenCV image
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return pd.DataFrame()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Crop to the left panel to isolate the table
    height, width = gray.shape
    left_panel = gray[:, :int(width * 0.12)] 

    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(left_panel, config=custom_config)
    
    data = []
    for line in text.split('\n'):
        match = re.search(r'^(\d{3})\s+([\d\.]+)', line.strip())
        if match:
            wl = int(match.group(1))
            val = float(match.group(2))
            data.append({'WL (nm)': wl, 'Value': val})
            
    return pd.DataFrame(data).drop_duplicates(subset=['WL (nm)'])

# --- Streamlit UI ---

st.set_page_config(page_title="Spectra Data Extractor", layout="centered")
st.title("Spectra Screenshot Extractor")
st.write("Upload standardized screenshots (e.g., `Black Band.png`, `Black XE.png`). The app will extract the Wavelength tables via OCR and combine them into a single Excel file.")

uploaded_files = st.file_uploader("Upload Screenshots", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Process Images"):
        with st.spinner("Extracting data..."):
            color_groups = {}
            
            for uploaded_file in uploaded_files:
                # Parse filename (e.g., "Black Band.png" -> "Black", "Band")
                filename = uploaded_file.name
                name_without_ext = filename.rsplit('.', 1)[0]
                parts = name_without_ext.split(' ', 1)
                
                if len(parts) < 2:
                    st.warning(f"Skipping `{filename}`: Name must be formatted as 'COLOR MATERIAL'.")
                    continue
                    
                color = parts[0]
                material = parts[1].strip()
                col_name = 'Band' if material.lower() == 'band' else f'Housing ({material})'
                
                # Extract data
                df = extract_data_from_image(uploaded_file)
                
                if df.empty:
                    st.error(f"No table data found in `{filename}`. Check image quality.")
                    continue
                    
                df = df.rename(columns={'Value': col_name})
                
                if color not in color_groups:
                    color_groups[color] = []
                color_groups[color].append(df)
            
            if not color_groups:
                st.error("No valid data could be extracted.")
            else:
                # Generate Excel file in memory
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    for color, dfs in color_groups.items():
                        merged_df = dfs[0]
                        for df in dfs[1:]:
                            merged_df = pd.merge(merged_df, df, on='WL (nm)', how='outer')
                            
                        merged_df = merged_df.sort_values('WL (nm)').reset_index(drop=True)
                        # Put each color in its own sheet
                        merged_df.to_excel(writer, sheet_name=color, index=False)
                
                excel_data = output.getvalue()
                
                st.success("Extraction complete!")
                st.download_button(
                    label="Download Compiled Excel File",
                    data=excel_data,
                    file_name="Compiled_Spectra_Data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
