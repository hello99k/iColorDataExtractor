import streamlit as st
import cv2
import pytesseract
import pandas as pd
import numpy as np
import re
import io

def extract_data_from_image(image_bytes, start_wl, end_wl, interval):
    """Decodes uploaded image bytes, pre-processes for OCR, and extracts the WL and %R/%T table."""
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return pd.DataFrame()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Tighter crop based on the iColor layout (left ~10%)
    height, width = gray.shape
    left_panel = gray[:, :int(width * 0.10)] 
    
    # 2. Upscale 3x and apply binarization (Otsu's Threshold) to make decimal points crystal clear
    left_panel = cv2.resize(left_panel, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, left_panel = cv2.threshold(left_panel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 3. Run OCR with Page Segmentation Mode 6
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(left_panel, config=custom_config)
    
    # 4. Generate the expected wavelengths dynamically based on user input
    # We add the interval to the end_wl to ensure the `range` includes the final number
    expected_wls = list(range(start_wl, end_wl + 1, interval))
    data_dict = {wl: None for wl in expected_wls}
    
    # 5. Extract values using a generalized Regular Expression
    for line in text.split('\n'):
        # Matches any 3 or 4 digit number (Wavelength) followed by a decimal or whole number
        match = re.search(r'(\d{3,4})\s*[^\d]*(\d+[\.,]\d+|\d+)', line.strip())
        if match:
            wl = int(match.group(1))
            # Handle instances where OCR reads a comma instead of a decimal point
            raw_val = match.group(2).replace(',', '.')
            val = float(raw_val)
            
            # Failsafe: If OCR still misses the decimal (e.g., reads "344" instead of "3.44")
            if val > 100:
                val = val / 100
                
            if wl in data_dict:
                data_dict[wl] = val
                
    # 6. Build the final DataFrame, keeping only the rows where data was successfully found
    df = pd.DataFrame([{'WL (nm)': k, 'Value': v} for k, v in data_dict.items() if v is not None])
    
    return df

# --- Streamlit UI ---

st.set_page_config(page_title="Spectra Data Extractor", layout="centered")
st.title("Spectra Screenshot Extractor")
st.write("Upload standardized screenshots (e.g., `Black Band.png`, `Black XE.png`). The app will extract the Wavelength tables via OCR and combine them into a single Excel file.")

uploaded_files = st.file_uploader("Upload Screenshots", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Add an expandable options tab for WL parameters
with st.expander("⚙️ Wavelength Options (Advanced)"):
    st.write("Adjust the expected wavelength range and interval if it differs from the iColor defaults.")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_wl = st.number_input("Start WL (nm)", value=360, step=10)
    with col2:
        end_wl = st.number_input("End WL (nm)", value=750, step=10)
    with col3:
        interval_wl = st.number_input("Interval (nm)", value=10, step=5)

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
                
                # Extract data using the custom WL parameters
                df = extract_data_from_image(uploaded_file, start_wl, end_wl, interval_wl)
                
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
