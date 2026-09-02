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
if "batches" not in st.session_state:
    st.session_state.batches = []  # Will store lists of files tagged with a batch_id
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "batch_counter" not in st.session_state:
    st.session_state.batch_counter = 1

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
                        if zip_info.is_dir() or zip_info.filename.startswith('__MACOSX') or os.path.basename(zip_info.filename).startswith('.'):
                            continue
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
    
    # Save this upload instance as a distinct series with a batch ID
    if current_batch:
        st.session_state.batches.append({
            'batch_id': st.session_state.batch_counter,
            'files': current_batch
        })
        st.session_state.batch_counter += 1
        
    st.session_state.uploader_key += 1

def undo_last_upload():
    if st.session_state.batches:
        st.session_state.batches.pop()
        # Reset counter if empty
        if not st.session_state.batches:
            st.session_state.batch_counter = 1

def clear_all_uploads():
    st.session_state.batches.clear()
    st.session_state.batch_counter = 1

def extract_data_from_image(image_bytes, start_wl, end_wl, interval):
    """Decodes image, validates it is a spectra screenshot, and extracts the table."""
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

# CSS to enable horizontal scrolling for Streamlit columns
st.markdown("""
    <style>
    [data-testid="stHorizontalBlock"] {
        overflow-x: auto;
        flex-wrap: nowrap;
        padding-bottom: 10px;
    }
    [data-testid="column"] {
        min-width: 140px !important;
        background-color: rgba(128, 128, 128, 0.05);
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Spectra Batch Extractor")
st.write("""
**Instructions:**
1. Drag and drop **folders, images, or .zip files** into the uploader below.
2. The app will capture the files and immediately clear the box so you can upload more.
""")

st.file_uploader(
    "Upload Screenshots (Images, Folders, or ZIP files)", 
    type=["png", "jpg", "jpeg", "zip"], 
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
    on_change=handle_upload
)

# --- Visual Queue & Queue Management ---
if st.session_state.batches:
    st.write("---")
    
    # 1. Parse current queue to group by Color
    ui_groups = {}
    total_files = 0
    for batch in st.session_state.batches:
        b_id = batch['batch_id']
        # List of all raw file names in this specific instance
        all_files_in_instance = [f['name'] for f in batch['files']]
        total_files += len(batch['files'])
        
        for f in batch['files']:
            name = f['name']
            name_without_ext = name.rsplit('.', 1)[0]
            parts = name_without_ext.split(' ', 1)
            
            # If it matches our naming convention, categorize it for the UI
            if len(parts) >= 2:
                color = parts[0].strip()
                if color not in ui_groups:
                    ui_groups[color] = {'relevant': set(), 'instances': {}}
                
                ui_groups[color]['relevant'].add(name)
                
                # Keep track of which batch instance(s) this color appeared in
                if b_id not in ui_groups[color]['instances']:
                    ui_groups[color]['instances'][b_id] = all_files_in_instance

    # 2. Display Horizontal Scrolling List of Colors
    st.subheader("🎨 Queued Colors Overview")
    if ui_groups:
        cols = st.columns(len(ui_groups))
        for idx, (color, data) in enumerate(ui_groups.items()):
            with cols[idx]:
                st.markdown(f"**{color}**")
                
                # Popover button (tally of relevant files)
                with st.popover(f"🖼️ {len(data['relevant'])} files"):
                    st.write(f"**Relevant '{color}' Files:**")
                    for rf in sorted(data['relevant']):
                        st.write(f"- `{rf}`")
                    
                    st.divider()
                    
                    # Small text for upload instances
                    st.caption("📦 **Full Instance Upload History:**")
                    for b_id, all_files in data['instances'].items():
                        st.caption(f"**Instance {b_id}:** {', '.join(all_files)}")
    else:
        st.info("No files matching the 'Color Material' format found in queue yet.")

    # 3. Queue Management Buttons
    st.write("")
    st.info(f"📁 **Total Queue Size:** {total_files} files queued across {len(st.session_state.batches)} upload event(s).")
    col1, col2, col3, col4 = st.columns([1, 1, 2, 2])
    with col1:
        st.button("↩️ Undo Last Upload", on_click=undo_last_upload, use_container_width=True)
    with col2:
        st.button("🗑️ Clear All", on_click=clear_all_uploads, use_container_width=True)


with st.expander("⚙️ Wavelength Options (Advanced)"):
    st.write("Adjust the expected wavelength range and interval if it differs from the defaults.")
    w_col1, w_col2, w_col3 = st.columns(3)
    with w_col1:
        start_wl = st.number_input("Start WL (nm)", value=360, step=10)
    with w_col2:
        end_wl = st.number_input("End WL (nm)", value=750, step=10)
    with w_col3:
        interval_wl = st.number_input("Interval (nm)", value=10, step=5)

# --- Processing Execution ---
if st.session_state.batches:
    st.write("---")
    if st.button("🚀 Process Batch Queue"):
        with st.spinner("Processing queue and extracting data..."):
            color_groups = {}
            processed_count = 0
            skipped_count = 0
            
            # Flatten batches into a single list of files
            all_files = [file for batch in st.session_state.batches for file in batch['files']]
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
