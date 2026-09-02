import streamlit as st
import cv2
import pytesseract
import pandas as pd
import numpy as np
import re
import io
import zipfile
import os

# --- Session State Management ---
# We use this to track files in the background so we can clear/undo specific batches
if "batches" not in st.session_state:
    st.session_state.batches = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

def handle_upload():
    """Callback to process files immediately when uploaded and queue them."""
    upload_key = f"uploader_{st.session_state.uploader_key}"
    uploaded_files = st.session_state.get(upload_key)
    
    if not uploaded_files:
        return
        
    current_batch = []
    for f in uploaded_files:
        # Handle ZIP files
        if f.name.lower().endswith('.zip'):
            try:
                with zipfile.ZipFile(f, 'r') as zip_ref:
                    for zip_info in zip_ref.infolist():
                        # Ignore directories and hidden macOS files
                        if zip_info.is_dir() or zip_info.filename.startswith('__MACOSX') or os.path.basename(zip_info.filename).startswith('.'):
                            continue
                        # Extract images from inside the zip
                        if zip_info.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            current_batch.append({
                                'name': os.path.basename(zip_info.filename),
                                'bytes': zip_ref.read(zip_info.filename)
                            })
            except Exception as e:
                st.error(f"Failed to read ZIP file {f.name}: {e}")
        # Handle standard image files
        else:
            current_batch.append({
                'name': f.name,
                'bytes': f.getvalue()
            })
    
    # Save this upload instance as a distinct series
    if current_batch:
        st.session_state.batches.append(current_batch)
        
    # Increment the key to visually reset the uploader box for the next drag-and-drop
    st.session_state.uploader_key += 1

def undo_last_upload():
    if st.session_state.batches:
        st.session_state.batches.pop()

def clear_all_uploads():
    st.session_state.batches.clear()

def extract_data_from_image(image_bytes, start_wl, end_wl, interval):
    """Decodes image, validates it is a spectra screenshot, and extracts the table."""
    # Modified to accept raw bytes from our session state list
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return pd.DataFrame(), False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    height, width = gray.shape
    left_panel = gray[:, :int(width * 0.10)] 
    
    left_panel = cv2.resize(left_panel, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, left_panel = cv2.threshold(left_panel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(left_panel, config=custom_config)
    
    if 'WL' not in text.upper() or '%' not in text:
        return pd.DataFrame(), False
    
    expected_wls = list(range(start_wl, end_wl + 1, interval))
    data_dict = {wl: None for wl in expected_wls}
    
    for line in text.split('\n'):
        match = re.search(r'(\d{3,4})\s*[^\d]*(\d+[\.,]\d+|\d+)', line.strip())
        if match:
            wl = int(match.group(1))
            raw_val = match.group(2).replace(',', '.')
            val = float(raw_val)
            
            if val > 100:
                val = val / 100
                
            if wl in data_dict:
                data_dict[wl] = val
                
    df = pd.DataFrame([{'WL (nm)': k, 'Value': v} for k, v in data_dict.items() if v is not None])
    return df, True


# --- Streamlit UI ---

st.set_page_config(page_title="Spectra Batch Extractor", layout="wide")
st.title("Spectra Batch Extractor")
st.write("""
**Instructions:**
1. Drag and drop **folders, images, or .zip files** into the uploader below.
2. The app will capture the files and immediately clear the box so you can upload more.
3. Use the Undo/Clear buttons if you accidentally upload the wrong folder.
""")

# File Uploader triggers the background saving process automatically
st.file_uploader(
    "Upload Screenshots (Images, Folders, or ZIP files)", 
    type=["png", "jpg", "jpeg", "zip"], 
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
    on_change=handle_upload
)

# Only show the queue management UI if there are files in the background queue
if st.session_state.batches:
    total_files = sum(len(batch) for batch in st.session_state.batches)
    total_batches = len(st.session_state.batches)
    
    st.info(f"📁 **Current Queue:** {total_files} files queued across {total_batches} upload event(s).")
    
    col1, col2, col3, col4 = st.columns([1, 1, 2, 2])
    with col1:
        st.button("↩️ Undo Last Upload", on_click=undo_last_upload, use_container_width=True)
    with col2:
        st.button("🗑️ Clear All", on_click=clear_all_uploads, use_container_width=True)


with st.expander("⚙️ Wavelength Options (Advanced)"):
    st.write("Adjust the expected wavelength range and interval if it differs from the defaults.")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_wl = st.number_input("Start WL (nm)", value=360, step=10)
    with col2:
        end_wl = st.number_input("End WL (nm)", value=750, step=10)
    with col3:
        interval_wl = st.number_input("Interval (nm)", value=10, step=5)

# Process Button
if st.session_state.batches:
    st.write("---")
    if st.button("🚀 Process Batch Queue"):
        with st.spinner("Processing queue and extracting data..."):
            color_groups = {}
            processed_count = 0
            skipped_count = 0
            
            # Flatten the batches into a single list of files for processing
            all_files = [file for batch in st.session_state.batches for file in batch]
            progress_bar = st.progress(0)
            
            for index, file_data in enumerate(all_files):
                filename = file_data['name']
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
                df, is_valid = extract_data_from_image(file_data['bytes'], start_wl, end_wl, interval_wl)
                
                # Check 2 & 3: Validation and Data check
                if not is_valid or df.empty:
                    skipped_count += 1
                    continue
                    
                # Store the dataframe
                df = df.rename(columns={'Value': col_name})
                
                if color not in color_groups:
                    color_groups[color] = []
                color_groups[color].append(df)
                
                processed_count += 1
                progress_bar.progress((index + 1) / len(all_files))
            
            progress_bar.empty()
            st.write(f"**Summary:** Successfully extracted data from {processed_count} files. Ignored {skipped_count} files.")
            
            if not color_groups:
                st.error("No valid spectra data could be extracted from the queue.")
            else:
                # Generate Excel file in memory
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    for color, dfs in color_groups.items():
                        merged_df = dfs[0]
                        for df in dfs[1:]:
                            merged_df = pd.merge(merged_df, df, on='WL (nm)', how='outer')
                            
                        merged_df = merged_df.sort_values('WL (nm)').reset_index(drop=True)
                        merged_df.to_excel(writer, sheet_name=color, index=False)
                
                excel_data = output.getvalue()
                
                st.success("Batch extraction complete!")
                st.download_button(
                    label="📥 Download Compiled Excel File",
                    data=excel_data,
                    file_name="Compiled_Spectra_Batch.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
