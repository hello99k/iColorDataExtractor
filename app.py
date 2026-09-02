import streamlit as st
import cv2
import pytesseract
import pandas as pd
import numpy as np
import re
import io

def extract_data_from_image(image_bytes, start_wl, end_wl, interval):
    """Decodes image, validates it is a spectra screenshot, and extracts the table."""
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Return empty DataFrame and False (is_valid = False) if image is unreadable
    if img is None:
        return pd.DataFrame(), False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Tighter crop based on the iColor layout (left ~10%)
    height, width = gray.shape
    left_panel = gray[:, :int(width * 0.10)] 
    
    # 2. Upscale 3x and apply binarization (Otsu's Threshold)
    left_panel = cv2.resize(left_panel, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, left_panel = cv2.threshold(left_panel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 3. Run OCR with Page Segmentation Mode 6
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(left_panel, config=custom_config)
    
    # 4. Validation Check: Look for signature headers
    # If the OCR didn't find "WL" and a "%" sign, it's likely not the right table
    if 'WL' not in text.upper() or '%' not in text:
        return pd.DataFrame(), False
    
    # 5. Generate the expected wavelengths dynamically based on user input
    expected_wls = list(range(start_wl, end_wl + 1, interval))
    data_dict = {wl: None for wl in expected_wls}
    
    # 6. Extract values using a generalized Regular Expression
    for line in text.split('\n'):
        # Matches any 3 or 4 digit number (Wavelength) followed by a decimal or whole number
        match = re.search(r'(\d{3,4})\s*[^\d]*(\d+[\.,]\d+|\d+)', line.strip())
        if match:
            wl = int(match.group(1))
            raw_val = match.group(2).replace(',', '.')
            val = float(raw_val)
            
            # Failsafe: If OCR still misses the decimal (e.g., reads "344" instead of "3.44")
            if val > 100:
                val = val / 100
                
            if wl in data_dict:
                data_dict[wl] = val
                
    # 7. Build the final DataFrame
    df = pd.DataFrame([{'WL (nm)': k, 'Value': v} for k, v in data_dict.items() if v is not None])
    
    return df, True

# --- Streamlit UI ---

st.set_page_config(page_title="Spectra Batch Extractor", layout="wide")
st.title("Spectra Batch Extractor")
st.write("""
**Instructions:**
1. Drag and drop a **folder** containing your screenshots directly into the uploader below (or select multiple files).
2. The app will automatically ignore non-image files, images without "WL" and "%R/%T" headers, and images not matching the `Color Material` naming format.
3. Every distinct color will be placed on its own sheet in the final Excel file.
""")

# accept_multiple_files=True allows dragging a whole folder; Streamlit will filter to the allowed types automatically
uploaded_files = st.file_uploader("Upload Screenshots (or drag a folder here)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

with st.expander("⚙️ Wavelength Options (Advanced)"):
    st.write("Adjust the expected wavelength range and interval if it differs from the defaults.")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_wl = st.number_input("Start WL (nm)", value=360, step=10)
    with col2:
        end_wl = st.number_input("End WL (nm)", value=750, step=10)
    with col3:
        interval_wl = st.number_input("Interval (nm)", value=10, step=5)

if uploaded_files:
    if st.button("Process Batch"):
        with st.spinner("Processing batch and extracting data..."):
            color_groups = {}
            processed_count = 0
            skipped_count = 0
            
            # Use a progress bar for large batches
            progress_bar = st.progress(0)
            
            for index, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                name_without_ext = filename.rsplit('.', 1)[0]
                parts = name_without_ext.split(' ', 1)
                
                # Check 1: Naming Convention
                if len(parts) < 2:
                    skipped_count += 1
                    continue
                    
                color = parts[0].strip()
                material = parts[1].strip()
                col_name = 'Band' if material.lower() == 'band' else f'Housing ({material})'
                
                # Extract data and Validate
                df, is_valid = extract_data_from_image(uploaded_file, start_wl, end_wl, interval_wl)
                
                # Check 2: Image Validation (Does it have the headers?)
                if not is_valid:
                    skipped_count += 1
                    continue
                    
                # Check 3: Did we actually extract data?
                if df.empty:
                    st.warning(f"Valid headers found in `{filename}`, but no table data could be extracted.")
                    skipped_count += 1
                    continue
                    
                # Passed all checks, format and store the dataframe
                df = df.rename(columns={'Value': col_name})
                
                if color not in color_groups:
                    color_groups[color] = []
                color_groups[color].append(df)
                
                processed_count += 1
                progress_bar.progress((index + 1) / len(uploaded_files))
            
            progress_bar.empty()
            
            # Display summary
            st.write(f"**Summary:** Processed {processed_count} files | Ignored {skipped_count} files.")
            
            if not color_groups:
                st.error("No valid spectra data could be extracted from the uploaded batch.")
            else:
                # Generate Excel file in memory
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    for color, dfs in color_groups.items():
                        # Merge all dataframes for this color on 'WL (nm)'
                        merged_df = dfs[0]
                        for df in dfs[1:]:
                            merged_df = pd.merge(merged_df, df, on='WL (nm)', how='outer')
                            
                        # Sort by Wavelength
                        merged_df = merged_df.sort_values('WL (nm)').reset_index(drop=True)
                        
                        # Create a new sheet for this color
                        merged_df.to_excel(writer, sheet_name=color, index=False)
                
                excel_data = output.getvalue()
                
                st.success("Batch extraction complete!")
                st.download_button(
                    label="Download Compiled Excel File",
                    data=excel_data,
                    file_name="Compiled_Spectra_Batch.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
