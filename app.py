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
    st.session_state.batches = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "batch_counter" not in st.session_state:
    st.session_state.batch_counter = 1
if "is_processed" not in st.session_state:
    st.session_state.is_processed = False
if "excel_data" not in st.session_state:
    st.session_state.excel_data = None

def reset_processing_state():
    """Clears the generated Excel file if the queue is modified."""
    st.session_state.is_processed = False
    st.session_state.excel_data = None

def handle_upload():
    """Callback to process files immediately when uploaded and queue them."""
    upload_key = f"uploader_{st.session_state.uploader_key}"
    uploaded_files = st.session_state.get(upload_key)
    
    if not uploaded_files:
        return
        
    current_batch = []
    for f in uploaded_files:
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
        else:
            current_batch.append({
                'name': f.name,
                'bytes': f.getvalue()
            })
    
    if current_batch:
        st.session_state.batches.append({
            'batch_id': st.session_state.batch_counter,
            'files': current_batch
        })
        st.session_state.batch_counter += 1
        reset_processing_state()
        
    st.session_state.uploader_key += 1

def undo_last_upload():
    if st.session_state.batches:
        st.session_state.batches.pop()
        if not st.session_state.batches:
            st.session_state.batch_counter = 1
        reset_processing_state()

def clear_all_uploads():
    st.session_state.batches.clear()
    st.session_state.batch_counter = 1
    reset_processing_state()

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

st.markdown("""
    <style>
    /* --- 1. Scoped Card Styling for Queued Colors --- */
    .color-cards-row + div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        background-color: rgba(128, 128, 128, 0.08) !important;
        padding: 12px !important;
        border-radius: 8px !important;
        text-align: center !important;
        min-width: 150px !important;
    }

    .color-cards-row + div[data-testid="stHorizontalBlock"] [data-testid="stPopover"],
    .color-cards-row + div[data-testid="stHorizontalBlock"] [data-testid="stPopover"] button {
        width: 100% !important;
    }

    /* --- 2. Uploader Centering with Emoji --- */
    [data-testid="stFileUploader"] label {
        display: none !important;
    }
    
    [data-testid="stFileUploaderDropzone"] {
        height: 216px !important; 
        min-height: 216px !important; 
        padding: 0 !important;
    }
    
    [data-testid="stFileUploaderDropzone"] > div {
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] svg,
    [data-testid="stFileUploaderDropzone"] > div > span {
        display: none !important;
    }

    [data-testid="stFileUploaderDropzone"] > div::before {
        content: "📤 Upload";
        font-weight: 600;
        font-size: 1.25rem;
        margin-bottom: 8px;
    }

    [data-testid="stFileUploaderDropzone"] small {
        margin: 0 !important;
        text-align: center !important;
    }

    /* --- 3. Undo/Clear Buttons Alignment --- */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div[data-testid="stButton"] button {
        height: 100px !important; 
        min-height: 100px !important; 
        justify-content: flex-start !important; 
        padding-left: 24px !important; 
        width: 100% !important;
        margin: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Spectra Batch Extractor")
st.write("""
**Instructions:** Drag and drop **folders, images, or .zip files** into the uploader below. The box will immediately capture the files and clear itself so you can upload more.
""")

with st.expander("⚙️ Optical Text Search Parameters (Advanced)"):
    st.write("Adjust the expected wavelength range and interval if it differs from the defaults.")
    w_col1, w_col2, w_col3 = st.columns(3)
    with w_col1:
        start_wl = st.number_input("Start WL (nm)", value=360, step=10)
    with w_col2:
        end_wl = st.number_input("End WL (nm)", value=750, step=10)
    with w_col3:
        interval_wl = st.number_input("Interval (nm)", value=10, step=5)

# --- Upload & Queue Management Row ---
col_upload, col_buttons = st.columns([3, 1])

with col_upload:
    st.file_uploader(
        "Upload Screenshots (Images, Folders, or ZIP files)", 
        type=["png", "jpg", "jpeg", "zip"], 
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        on_change=handle_upload,
        label_visibility="collapsed"
    )

with col_buttons:
    st.button("↩️ Undo Last Upload", on_click=undo_last_upload, use_container_width=True)
    st.button("🗑️ Clear All", on_click=clear_all_uploads, use_container_width=True)


# --- Permanent Queued Colors Section ---
st.write("---")
st.subheader("🎨 Queued Colors")

ui_groups = {}
total_files = 0

if st.session_state.batches:
    for batch in st.session_state.batches:
        b_id = batch['batch_id']
        all_files_in_instance = [f['name'] for f in batch['files']]
        total_files += len(batch['files'])
        
        for f in batch['files']:
            name = f['name']
            name_without_ext = name.rsplit('.', 1)[0]
            parts = name_without_ext.split(' ', 1)
            
            if len(parts) >= 2:
                color = parts[0].strip()
                if color not in ui_groups:
                    ui_groups[color] = {'relevant': set(), 'instances': {}}
                
                ui_groups[color]['relevant'].add(name)
                
                if b_id not in ui_groups[color]['instances']:
                    ui_groups[color]['instances'][b_id] = all_files_in_instance

    st.markdown('<div class="color-cards-row"></div>', unsafe_allow_html=True)
    
    if ui_groups:
        cols = st.columns(len(ui_groups), wrap=False)
        for idx, (color, data) in enumerate(ui_groups.items()):
            with cols[idx]:
                st.markdown(f"**{color}**")
                
                # Capitalized 'Materials' for the button label
                with st.popover(f"{len(data['relevant'])} Materials"):
                    st.write(f"**Relevant '{color}' Files:**")
                    for rf in sorted(data['relevant']):
                        st.write(f"- `{rf}`")
                    
                    st.divider()
                    st.caption("📦 **Full Instance Upload History:**")
                    
                    # Converted instance history into clickable drop-downs (expanders)
                    for b_id, all_files in data['instances'].items():
                        with st.expander(f"Instance {b_id} ({len(all_files)} files)"):
                            for inst_file in all_files:
                                st.caption(f"- `{inst_file}`")
    else:
        st.info("Files uploaded, but none match the required 'Color Material' naming format.")
else:
    st.info("The queue is currently empty. Upload files to begin.")


# --- Processing Execution & Download ---
if st.session_state.batches:
    st.write("---")
    st.write(f"📁 **Total Queue Size:** {total_files} files queued across {len(st.session_state.batches)} upload event(s).")
    
    if st.button("🚀 Process Batch Queue", use_container_width=True):
        with st.spinner("Processing queue and extracting data..."):
            color_groups = {}
            processed_count = 0
            skipped_count = 0
            
            all_files = [file for batch in st.session_state.batches for file in batch['files']]
            progress_bar = st.progress(0)
            
            for index, file_data in enumerate(all_files):
                filename = file_data['name']
                name_without_ext = filename.rsplit('.', 1)[0]
                parts = name_without_ext.split(' ', 1)
                
                if len(parts) < 2:
                    skipped_count += 1
                    continue
                    
                color = parts[0].strip()
                material = parts[1].strip()
                col_name = 'Band' if material.lower() == 'band' else f'Housing ({material})'
                
                df, is_valid = extract_data_from_image(file_data['bytes'], start_wl, end_wl, interval_wl)
                
                if not is_valid or df.empty:
                    skipped_count += 1
                    continue
                    
                df = df.rename(columns={'Value': col_name})
                
                if color not in color_groups:
                    color_groups[color] = []
                color_groups[color].append(df)
                
                processed_count += 1
                progress_bar.progress((index + 1) / len(all_files))
            
            progress_bar.empty()
            
            if not color_groups:
                st.error("No valid spectra data could be extracted from the queue.")
            else:
                st.success(f"Successfully extracted data from {processed_count} files. Ignored {skipped_count} irrelevant files.")
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    for color, dfs in color_groups.items():
                        merged_df = dfs[0]
                        for df in dfs[1:]:
                            merged_df = pd.merge(merged_df, df, on='WL (nm)', how='outer')
                            
                        merged_df = merged_df.sort_values('WL (nm)').reset_index(drop=True)
                        merged_df.to_excel(writer, sheet_name=color, index=False)
                
                st.session_state.excel_data = output.getvalue()
                st.session_state.is_processed = True

    # Reveal filename input and download button AFTER processing is complete
    if st.session_state.is_processed and st.session_state.excel_data:
        st.write("---")
        default_filename = ", ".join(ui_groups.keys())
        
        user_filename = st.text_input("📝 Enter a name for your compiled Excel file:", value=default_filename)
        
        final_filename = user_filename.strip()
        if not final_filename.endswith(".xlsx"):
            final_filename += ".xlsx"
            
        st.download_button(
            label="📥 Download Compiled Excel File",
            data=st.session_state.excel_data,
            file_name=final_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
