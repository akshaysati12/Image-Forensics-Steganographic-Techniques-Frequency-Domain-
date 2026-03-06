import streamlit as st
import os
import numpy as np
from PIL import Image, ImageChops
import cv2
import tempfile
import math
import matplotlib.pyplot as plt
import io

# --- Helper Functions ---

def compute_ela(img_pil, quality):
    # Save to temp
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img_pil.convert("RGB").save(tmp.name, "JPEG", quality=quality)
        tmp_name = tmp.name
    
    # Reload
    reloaded = Image.open(tmp_name).convert("RGB")
    orig_rgb = img_pil.convert("RGB")
    
    diff = ImageChops.difference(orig_rgb, reloaded)
    diff_np = np.array(diff)
    
    # Clean up
    os.remove(tmp_name)
    
    # Enhance
    max_diff = np.max(diff_np)
    scale = (255.0 / max_diff) if max_diff > 0 else 1.0
    diff_scaled = (diff_np.astype(float) * scale).clip(0, 255).astype(np.uint8)
    
    ela_img = Image.fromarray(diff_scaled)
    return ela_img, diff_np

def compute_mse_psnr(img1_gray, img2_gray):
    img2_gray = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))
    mse = np.mean((img1_gray.astype(float) - img2_gray.astype(float)) ** 2)
    if mse == 0:
        return 0, float('inf')
    psnr = 10 * math.log10((255**2) / mse)
    return mse, psnr

# --- DCT Steganography Functions (Frequency Domain) ---

# Standard JPEG Quantization Table
Q_TABLE = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])

def embed_dct_steganography(img_gray, secret_message):
    """Embeds text into the frequency domain (DCT) of an image."""
    # Append marker to know when to stop reading
    secret_message += "#####"
    
    # Convert message to bits
    bits = ''.join([format(ord(c), '08b') for c in secret_message])
    bit_id = 0
    bit_len = len(bits)
    
    h, w = img_gray.shape
    stego_img = img_gray.copy()
    
    # Process 8x8 blocks
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            if bit_id >= bit_len:
                break
            
            # Get 8x8 block
            block = stego_img[y:y+8, x:x+8]
            
            # Skip if block is not 8x8 (edges)
            if block.shape[0] != 8 or block.shape[1] != 8:
                continue
                
            # 1. Apply DCT (Frequency Domain)
            block_f = np.float32(block)
            dct_block = cv2.dct(block_f)
            
            # 2. Quantize
            q_block = np.round(dct_block / Q_TABLE)
            
            # 3. Embed bit in (4,3) coefficient (Mid-frequency)
            cy, cx = 4, 3
            coeff = int(q_block[cy, cx])
            bit = int(bits[bit_id])
            
            # Replace LSB of the coefficient
            q_block[cy, cx] = (coeff & ~1) | bit
            bit_id += 1
            
            # 4. Dequantize and Inverse DCT
            deq_block = q_block * Q_TABLE
            idct_block = cv2.idct(deq_block)
            
            # 5. Update image block
            stego_img[y:y+8, x:x+8] = np.uint8(np.clip(idct_block, 0, 255))
            
    return stego_img

def extract_dct_steganography(stego_img):
    """Extracts text from the frequency domain (DCT) of an image."""
    h, w = stego_img.shape
    binary = ""
    message = ""
    
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            block = stego_img[y:y+8, x:x+8]
            if block.shape[0] != 8 or block.shape[1] != 8:
                continue
            
            # DCT & Quantize
            dct_block = cv2.dct(np.float32(block))
            q_block = np.round(dct_block / Q_TABLE)
            
            # Read LSB from (4,3)
            bit = int(q_block[4, 3]) & 1
            binary += str(bit)
            
            # Convert every 8 bits to char
            if len(binary) % 8 == 0:
                byte = binary[-8:]
                try:
                    char = chr(int(byte, 2))
                    message += char
                    if message.endswith("#####"):
                        return message[:-5]
                except ValueError:
                    pass
                    
    return message

# --- Streamlit UI ---

st.set_page_config(page_title="Image Forensics Tool", layout="wide")

st.title("🕵️ Image Forensics & Steganography Detection")
st.markdown("""
This tool helps detect **Forgery** (via ELA) and **Steganography** (via Frequency Analysis).
""")

# Sidebar
st.sidebar.header("Settings")
mode = st.sidebar.radio("Select Module", [
    "1. Hide Data (DCT Steganography)",
    "2. Analyze (Forensics)",
    "3. Extract Data (Recovery)",
    "4. ELA (Forgery Detection)"
])

# 1. HIDE DATA MODE
if mode == "1. Hide Data (DCT Steganography)":
    st.header("1. Hide Data using Frequency Domain (DCT)")
    st.info("This module hides text inside the frequency coefficients of an image.")
    
    cover_file = st.file_uploader("Upload Cover Image", type=["jpg", "png", "bmp"])
    secret_text = st.text_area("Enter Secret Message", "This is a secret message.")
    
    if cover_file and secret_text:
        # Load and convert to grayscale for DCT demo
        file_bytes = np.asarray(bytearray(cover_file.read()), dtype=np.uint8)
        img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        st.image(img_gray, caption="Original Cover Image (Grayscale)", width=400)
        
        if st.button("Hide Data"):
            with st.spinner("Embedding data into DCT coefficients..."):
                stego_img = embed_dct_steganography(img_gray, secret_text)
                
                st.success("Data Hidden Successfully!")
                st.image(stego_img, caption="Stego Image (Contains Hidden Data)", width=400)
                
                # Convert to PNG for download
                is_success, buffer = cv2.imencode(".png", stego_img)
                if is_success:
                    st.download_button(
                        label="Download Stego Image",
                        data=io.BytesIO(buffer),
                        file_name="stego_image.png",
                        mime="image/png"
                    )

# 3. EXTRACT DATA MODE
elif mode == "3. Extract Data (Recovery)":
    st.header("3. Extract Data from Stego Image")
    st.info("Recover the hidden message from the frequency domain.")
    
    stego_file = st.file_uploader("Upload Stego Image", type=["png", "bmp", "jpg"])
    
    if stego_file:
        file_bytes = np.asarray(bytearray(stego_file.read()), dtype=np.uint8)
        stego_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if st.button("Extract Message"):
            recovered_msg = extract_dct_steganography(stego_img)
            if recovered_msg:
                st.success(f"Recovered Message: {recovered_msg}")
            else:
                st.warning("No valid message found or marker missing.")

# 4. ELA MODE
elif mode == "4. ELA (Forgery Detection)":
    st.header("Error Level Analysis (ELA)")
    st.info("Upload a SUSPECT image to check for compression anomalies (splicing).")
    
    uploaded_file = st.file_uploader("Upload Suspect Image", type=["jpg", "png", "jpeg", "bmp"])
    quality = st.sidebar.slider("ELA JPEG Quality", 10, 100, 90)
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Suspect Image", use_container_width=True)
        
        if st.button("Run ELA Analysis"):
            with st.spinner("Processing..."):
                ela_result, diff_np = compute_ela(img, quality)
                
                # ELA Result
                with col2:
                    st.image(ela_result, caption=f"ELA Result (Quality {quality})", use_container_width=True)
                
                # Histogram
                diff_gray = np.max(diff_np, axis=2)
                fig, ax = plt.subplots()
                ax.hist(diff_gray.flatten(), bins=50, color='purple')
                ax.set_title("ELA Difference Histogram")
                st.pyplot(fig)
                
                st.success("ELA Analysis Complete. Bright regions indicate higher compression error (potential manipulation).")

# 2. ANALYZE MODE
elif mode == "2. Analyze (Forensics)":
    st.header("Frequency Domain Analysis (Steganography Detection)")
    st.info("Compare an ORIGINAL image with a SUSPECT image to find hidden data (Steganography).")
    
    col1, col2 = st.columns(2)
    with col1:
        orig_file = st.file_uploader("1. Upload ORIGINAL Image", type=["jpg", "png", "jpeg", "bmp"])
    with col2:
        sus_file = st.file_uploader("2. Upload SUSPECT Image", type=["jpg", "png", "jpeg", "bmp"])
        
    if orig_file and sus_file:
        # Convert to OpenCV format
        file_bytes_o = np.asarray(bytearray(orig_file.read()), dtype=np.uint8)
        img_o = cv2.imdecode(file_bytes_o, cv2.IMREAD_GRAYSCALE)
        
        file_bytes_s = np.asarray(bytearray(sus_file.read()), dtype=np.uint8)
        img_s = cv2.imdecode(file_bytes_s, cv2.IMREAD_GRAYSCALE)
        
        if st.button("Run Frequency Analysis"):
            with st.spinner("Analyzing Frequency Domains (DCT, FFT)..."):
                
                # --- 1. SPATIAL ANALYSIS ---
                st.subheader("1. Spatial Domain Analysis")
                
                mse, psnr = compute_mse_psnr(img_o, img_s)
                
                # Metrics Layout
                m1, m2, m3 = st.columns(3)
                m1.metric("MSE (Error)", f"{mse:.5f}", help="Mean Squared Error. 0 is perfect match.")
                m2.metric("PSNR (Quality)", f"{psnr:.2f} dB", help="Peak Signal-to-Noise Ratio. >40dB is high quality.")
                
                # Verdict Logic
                if mse == 0:
                    m3.success("✅ Identical")
                elif psnr > 45:
                    m3.error("🚨 Likely Steganography")
                    st.warning("⚠️ **Suspicious Result**: Images look identical (High PSNR) but contain mathematical differences. This is a strong indicator of hidden data.")
                else:
                    m3.info("ℹ️ Visible Differences")

                # Visual Difference Map
                st.write("#### Pixel Difference Map")
                diff_spatial = cv2.absdiff(img_o, img_s)
                
                # Contrast stretching to make invisible noise visible
                if np.max(diff_spatial) > 0:
                    diff_enhanced = cv2.normalize(diff_spatial, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                else:
                    diff_enhanced = diff_spatial

                c_diff1, c_diff2 = st.columns([1, 1])
                with c_diff1:
                    st.image(diff_enhanced, caption="Enhanced Difference (Noise Map)", use_container_width=True)
                with c_diff2:
                    st.markdown("""
                    **Interpretation:**
                    *   **Black**: No change.
                    *   **Snow/Static**: Indicates LSB Steganography or noise addition.
                    *   **Shapes/Lines**: Indicates image editing (cropping, painting).
                    """)

                # --- 1.5 LOCATION VISUALIZATION ---
                st.write("#### 🔍 Hidden Data Location Analysis")
                st.info("Visualizing where the image data has been altered (likely where the message is hidden).")
                
                # Create a heatmap overlay
                _, mask = cv2.threshold(diff_spatial, 0, 255, cv2.THRESH_BINARY)
                
                # Calculate percentage
                modified_pixels = np.count_nonzero(mask)
                total_pixels = mask.size
                modification_rate = (modified_pixels / total_pixels) * 100
                
                c_loc1, c_loc2 = st.columns([1, 1])
                with c_loc1:
                    # Create red overlay for modified regions
                    suspect_rgb = cv2.cvtColor(img_s, cv2.COLOR_GRAY2RGB)
                    # Alpha blend red on modified pixels
                    overlay = suspect_rgb.copy()
                    # Pixels where mask > 0 get blended with Red (255, 0, 0)
                    overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
                    st.image(overlay, caption="Stego Location Overlay (Red = Modified)", use_container_width=True)
                
                with c_loc2:
                    st.metric("Modified Area", f"{modification_rate:.4f}%")
                    st.markdown("**Insight:** The red highlighted regions indicate exactly which 8x8 blocks or pixels contain the hidden data.")
                    
                    with st.expander("🕵️ Attempt to Decode Message"):
                        if st.button("Decode Now"):
                            msg = extract_dct_steganography(img_s)
                            if msg:
                                st.success(f"Decoded: **{msg}**")
                            else:
                                st.warning("Could not decode a valid message (or no marker found).")

                # --- 2. FREQUENCY ANALYSIS ---
                st.subheader("2. Frequency Domain (DCT)")
                
                img_s_resized = cv2.resize(img_s, (img_o.shape[1], img_o.shape[0]))
                dct_o = cv2.dct(img_o.astype(np.float32))
                dct_s = cv2.dct(img_s_resized.astype(np.float32))
                dct_diff = np.abs(dct_o - dct_s)
                
                # Heatmap (Log Scale for better visibility)
                st.write("**DCT Coefficient Difference Heatmap**")
                fig_hm, ax_hm = plt.subplots()
                # Log scale helps see small steganographic changes
                dct_diff_log = np.log1p(dct_diff)
                im = ax_hm.imshow(dct_diff_log, cmap='inferno', interpolation='nearest')
                plt.colorbar(im, ax=ax_hm, label="Log Difference")
                ax_hm.set_title("Frequency Differences (Log Scale)")
                ax_hm.axis('off')
                st.pyplot(fig_hm)
                
                # Histogram Comparison
                st.write("**DCT Histogram Comparison**")
                st.caption("Steganography often distorts the natural distribution of DCT coefficients.")
                
                fig_hist, ax_hist = plt.subplots()
                data_o = np.log1p(np.abs(dct_o.flatten()))
                data_s = np.log1p(np.abs(dct_s.flatten()))
                
                ax_hist.hist(data_o, bins=100, alpha=0.5, label='Original', color='blue', log=True)
                ax_hist.hist(data_s, bins=100, alpha=0.5, label='Suspect', color='red', log=True)
                ax_hist.legend()
                ax_hist.set_xlabel("Log Magnitude")
                ax_hist.set_ylabel("Count (Log Scale)")
                ax_hist.grid(True, which="both", ls="-", alpha=0.2)
                st.pyplot(fig_hist)
                
                st.success("Frequency Analysis Complete.")