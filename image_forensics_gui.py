
print("--- Starting Image Forensics Application ---")

from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageChops, ImageEnhance, ImageDraw
import os
import tempfile
import numpy as np
import datetime
import math
import cv2
import pywt

from forensics_graphs import save_ela_graphs, save_freq_graphs, save_dct_histogram_comparison

# Globals
img_path = None               # suspect image path
img = None                    # suspect PIL image
start_dir = os.path.expanduser("~")
original_compare_path = None  # original/reference image path for frequency analysis


# ===========================
#  Folder chooser
# ===========================

def choose_start_folder():
    """Let user change the starting folder for the file dialog."""
    global start_dir
    d = filedialog.askdirectory(title="Choose start directory for dialogs")
    if d:
        start_dir = d
        start_folder_label.config(text="Start folder: " + start_dir)


# ===========================
#  Upload SUSPECT image
# ===========================

def upload_image():
    global img_path, img
    default_dir = start_dir if os.path.isdir(start_dir) else os.path.expanduser("~")

    img_path = filedialog.askopenfilename(
        title="Select SUSPECT image",
        initialdir=default_dir,
        filetypes=[
            ("Image files",
             ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff",
              "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIFF"))
        ]
    )
    if not img_path:
        return

    try:
        img = Image.open(img_path)
    except Exception as e:
        messagebox.showerror("Error", f"Unable to open image:\n{e}")
        img_path = None
        img = None
        return

    preview = img.copy()
    preview.thumbnail((360, 360))
    tk_img = ImageTk.PhotoImage(preview)
    panel.config(image=tk_img, text="")
    panel.image = tk_img

    result_label.config(text="Suspect image loaded: " + os.path.basename(img_path), fg="black")
    ela_panel.config(image="", text="ELA Result")
    ela_panel.image = None
    overlay_panel.config(image="", text="Overlay Result")
    overlay_panel.image = None


# ===========================
#  Upload ORIGINAL image
# ===========================

def upload_original_image():
    """Upload original/reference image for frequency-domain comparison."""
    global original_compare_path
    default_dir = start_dir if os.path.isdir(start_dir) else os.path.expanduser("~")

    path = filedialog.askopenfilename(
        title="Select ORIGINAL image (for comparison)",
        initialdir=default_dir,
        filetypes=[
            ("Image files",
             ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff",
              "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIFF"))
        ]
    )
    if not path:
        return

    original_compare_path = path
    original_label.config(text="Original for Compare: " + os.path.basename(original_compare_path))


# ===========================
#  ELA report text builder
# ===========================

def make_report_lines(basename, ela_quality, width, height, max_diff, mean_diff, std_diff,
                      threshold_val, flagged_pixels, total_pixels, boxes, percent_flagged):
    """Return formatted list of lines to write to ELA report file."""
    lines = []
    lines.append("-----------------------------------------")
    lines.append("IMAGE FORGERY DETECTION SYSTEM REPORT (ELA)")
    lines.append("-----------------------------------------")
    lines.append(f"File Name: {os.path.basename(basename + os.path.splitext(img_path)[1])}")
    lines.append(f"Full path: {img_path}")
    lines.append(f"Result ELA image: {os.path.basename(basename + f'_ela_q{ela_quality}_result.jpg')}")
    lines.append(f"Overlay image: {os.path.basename(basename + '_overlay_boxes.jpg')}")
    lines.append(f"Date: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    lines.append(f"ELA JPEG quality used: {ela_quality}")
    lines.append(f"Image size (W x H): {width} x {height} pixels")
    lines.append("")
    lines.append("Processing steps:")
    lines.append("1. Original image loaded.")
    lines.append("2. Image re-saved as JPEG at the chosen quality for ELA comparison.")
    lines.append("3. Pixel-wise difference computed between original (converted to RGB) and re-saved JPEG.")
    lines.append("4. Difference image scaled, contrast-enhanced and thresholded to find suspicious regions.")
    lines.append("")
    lines.append("Pixel Statistics:")
    lines.append(f"- Max Difference (0-255): {int(max_diff)}")
    lines.append(f"- Mean Difference: {mean_diff:.3f}")
    lines.append(f"- Std Deviation: {std_diff:.3f}")
    lines.append(f"- Threshold method: mean + k*std   (k = {threshold_k_var.get()})")
    lines.append(f"- Threshold value used: {threshold_val:.3f}")
    lines.append(f"- Flagged pixels: {flagged_pixels} / {total_pixels}  ({percent_flagged:.4f}%)")
    lines.append("")
    lines.append(f"Detected {len(boxes)} suspicious region(s):")
    for idx, b in enumerate(boxes, start=1):
        minx, miny, maxx, maxy = b["bbox"]
        w_box = maxx - minx + 1
        h_box = maxy - miny + 1
        area = b["area_pixels"]
        lines.append(f" Region {idx}: bbox=(x={minx}, y={miny}, w={w_box}, h={h_box}), area_pixels={area}")
        lines.append(f"   - region percent of image: {100.0 * area / total_pixels:.6f}%")
    lines.append("")
    # Human-friendly conclusion using percent_flagged
    if percent_flagged < 0.05:
        lines.append("Conclusion: No significant tampering found. The image appears to be ORIGINAL or shows only very small differences/noise.")
        conclusion_text = "ORIGINAL IMAGE"
    elif percent_flagged < 1.0:
        lines.append("Conclusion: Small areas flagged. Could be minor edits, recompression artifacts, or noise. Manual review recommended.")
        conclusion_text = "POSSIBLE MINOR EDITS"
    else:
        lines.append("Conclusion: Significant flagged area (>1%). Likely tampering/splicing detected. Review overlay images for suspected regions.")
        conclusion_text = "LIKELY FORGED"
    lines.append("")
    lines.append("Interpretation Guide (Hidden Edits & Watermarks):")
    lines.append("- ELA detects when different parts of an image have different compression levels.")
    lines.append("- Normal/Original: Dark, uniform noise.")
    lines.append("- Modified/Watermarked: Bright, glowing patches (high error level).")
    lines.append("- If you see a shape 'behind' the image glowing, that area was likely added later.")
    lines.append("-----------------------------------------")
    lines.append(f"Result Status: {conclusion_text}")
    lines.append("-----------------------------------------")
    return lines


# ===========================
#  ELA Forgery Detection
# ===========================

def detect_forgery_and_report():
    global img_path, img
    if not img_path or img is None:
        result_label.config(text="Please upload a SUSPECT image first!", fg="red")
        return

    print("--- Starting ELA Detection ---")
    try:
        # Temporary JPEG path
        fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)

        # Convert to RGB (handle alpha)
        base_rgba = img.convert("RGBA")
        if base_rgba.mode == "RGBA":
            background = Image.new("RGBA", base_rgba.size, (255, 255, 255, 255))
            background.paste(base_rgba, mask=base_rgba.split()[3])
            rgb_img = background.convert("RGB")
        else:
            rgb_img = base_rgba.convert("RGB")

        # Save JPEG re-encoded for ELA
        ela_quality = ela_quality_var.get()
        if not (10 <= ela_quality <= 100):
            ela_quality = 90

        rgb_img.save(temp_path, 'JPEG', quality=ela_quality)

        # Reload and compute difference
        reloaded = Image.open(temp_path).convert("RGB")
        diff_img = ImageChops.difference(rgb_img, reloaded)

        diff_np = np.array(diff_img).astype(np.uint8)   # (H,W,3)
        diff_gray = diff_np.max(axis=2)                 # 0..255

        max_diff = int(diff_gray.max())
        mean_diff = float(diff_gray.mean())
        std_diff = float(diff_gray.std())

        # Scale to full range
        scale = (255.0 / max_diff) if max_diff != 0 else 1.0
        diff_scaled = np.clip((diff_gray.astype(np.float32) * scale), 0, 255).astype(np.uint8)
      
        # Enhance for visualization
        ela_image = Image.fromarray(diff_scaled).convert("RGB")
        ela_image = ImageEnhance.Contrast(ela_image).enhance(1.6)
        ela_image = ImageEnhance.Brightness(ela_image).enhance(1.05)

        base_name, ext = os.path.splitext(img_path)
        ela_out_path = base_name + f"_ela_q{ela_quality}_result.jpg"
        ela_image.save(ela_out_path)
        print(f"ELA image saved: {ela_out_path}")

        # Thresholding
        k = threshold_k_var.get()
        threshold_val = mean_diff + k * std_diff
        threshold_val = min(max(threshold_val, 5), 200)
        mask = (diff_gray >= threshold_val).astype(np.uint8) * 255

        total_pixels = mask.size
        flagged_pixels = int((mask > 0).sum())
        percent_flagged = (flagged_pixels / total_pixels) * 100

        # --- NEW: Save ELA graphs (histogram + pie chart) ---
        try:
            ela_hist_path, ela_pie_path = save_ela_graphs(diff_gray, mask, base_name)
        except Exception as ge:
            print("Warning: could not save ELA graphs:", ge)
            ela_hist_path, ela_pie_path = None, None

        # Connected components (simple flood-fill)
        height, width = mask.shape
        visited = np.zeros_like(mask, dtype=np.uint8)
        boxes = []
        for y in range(height):
            for x in range(width):
                if mask[y, x] and not visited[y, x]:
                    stack = [(y, x)]
                    visited[y, x] = 1
                    minx, miny = x, y
                    maxx, maxy = x, y
                    area_count = 0
                    while stack:
                        cy, cx = stack.pop()
                        area_count += 1
                        if cx < minx: minx = cx
                        if cx > maxx: maxx = cx
                        if cy < miny: miny = cy
                        if cy > maxy: maxy = cy
                        for ny in range(max(0, cy - 1), min(height, cy + 2)):
                            for nx in range(max(0, cx - 1), min(width, cx + 2)):
                                if mask[ny, nx] and not visited[ny, nx]:
                                    visited[ny, nx] = 1
                                    stack.append((ny, nx))
                    boxes.append({
                        "bbox": (minx, miny, maxx, maxy),
                        "area_pixels": area_count
                    })

        # Overlay image
        orig_np = np.array(rgb_img).astype(np.uint8)
        overlay_np = orig_np.copy()
        overlay_layer = np.zeros((height, width, 3), dtype=np.uint8)
        overlay_layer[:, :, 0] = mask  # red channel
        alpha = 0.45
        blended = np.clip((overlay_np * (1 - alpha) + overlay_layer * alpha), 0, 255).astype(np.uint8)
        overlay_img = Image.fromarray(blended)

        # Draw bounding boxes (yellow)
        draw = ImageDraw.Draw(overlay_img)
        for b in boxes:
            minx, miny, maxx, maxy = b["bbox"]
            pad = max(2, int(0.01 * max(width, height)))
            draw.rectangle([minx - pad, miny - pad, maxx + pad, maxy + pad],
                           outline=(255, 255, 0), width=2)

        overlay_with_boxes_path = base_name + "_overlay_boxes.jpg"
        overlay_img.save(overlay_with_boxes_path)

        # ELA text report
        report_lines = make_report_lines(
            base_name, ela_quality, width, height,
            max_diff, mean_diff, std_diff,
            threshold_val, flagged_pixels, total_pixels, boxes, percent_flagged
        )
        report_path = base_name + "_ela_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        # Append info about graphs
        if ela_hist_path or ela_pie_path:
            with open(report_path, "a", encoding="utf-8") as f:
                f.write("\n\nGraphical Outputs:\n")
                if ela_hist_path:
                    f.write(f"- ELA Histogram: {os.path.basename(ela_hist_path)}\n")
                if ela_pie_path:
                    f.write(f"- ELA Suspicious vs Normal Pixels Pie Chart: {os.path.basename(ela_pie_path)}\n")

        # Show previews
        ela_preview = Image.open(ela_out_path)
        ela_preview.thumbnail((360, 360))
        tk_ela = ImageTk.PhotoImage(ela_preview)
        ela_panel.config(image=tk_ela, text="")
        ela_panel.image = tk_ela

        overlay_preview = Image.open(overlay_with_boxes_path)
        overlay_preview.thumbnail((360, 360))
        tk_overlay = ImageTk.PhotoImage(overlay_preview)
        overlay_panel.config(image=tk_overlay, text="")
        overlay_panel.image = tk_overlay

        # Final conclusion (same logic as report)
        if percent_flagged < 0.05:
            conclusion_text = "ORIGINAL IMAGE"
            human_readable = "No significant tampering found. The image appears to be ORIGINAL or contains only very small noise."
        elif percent_flagged < 1.0:
            conclusion_text = "POSSIBLE MINOR EDITS"
            human_readable = "Small areas flagged. Could be minor edits or compression artifacts. Manual review recommended."
        else:
            conclusion_text = "LIKELY FORGED"
            human_readable = "Significant flagged area detected. The image is likely forged — review overlay images."

        result_label.config(
            text=f"ELA done | ELA: {os.path.basename(ela_out_path)} | Report: {os.path.basename(report_path)} | Status: {conclusion_text}",
            fg="green"
        )
        messagebox.showinfo(
            "ELA Report Summary",
            f"ELA completed.\n\nELA image: {os.path.basename(ela_out_path)}\n"
            f"Overlay: {os.path.basename(overlay_with_boxes_path)}\n"
            f"Report: {os.path.basename(report_path)}\n\n"
            f"Status: {conclusion_text}\n\n{human_readable}"
        )

        print("ELA Analysis Completed Successfully.")

    except Exception as e:
        print(f"ERROR during ELA: {e}")
        messagebox.showerror("Detection Error", f"Error during ELA/report:\n{e}")
        result_label.config(text="Error during ELA/report. See message.", fg="red")
    finally:
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


# ===========================
#  Frequency-domain Forensics
# ===========================

def compute_mse_psnr(img1, img2):
    """Compute MSE and PSNR between two grayscale images."""
    img1 = cv2_resize_to_match(img1, img2)
    a = img1.astype(np.float32)
    b = img2.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * math.log10((255 ** 2) / mse)
    return mse, psnr


def cv2_resize_to_match(a, b):
    """Resize a to the same size as b using nearest neighbor."""
    return cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_NEAREST)


def frequency_analysis_compare():
    """
    Perform frequency-domain forensics between ORIGINAL and SUSPECT image:
    - MSE / PSNR
    - DCT differences
    - DWT (Haar) differences
    - FFT magnitude differences
    """
    global original_compare_path, img_path

    if not original_compare_path:
        messagebox.showwarning("Missing Original", "Please upload the ORIGINAL image for comparison.")
        return
    if not img_path:
        messagebox.showwarning("Missing Suspect", "Please upload the SUSPECT image first.")
        return

    print("--- Starting Frequency Analysis ---")
    try:
        orig = cv2.imread(original_compare_path, cv2.IMREAD_GRAYSCALE)
        sus = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if orig is None:
            messagebox.showerror("Error", "Could not open original image for frequency analysis.")
            return
        if sus is None:
            messagebox.showerror("Error", "Could not open suspect image for frequency analysis.")
            return

        sus = cv2.resize(sus, (orig.shape[1], orig.shape[0]))

        # 1) Spatial metrics
        mse, psnr = compute_mse_psnr(orig, sus)

        # 2) DCT analysis
        orig_f = orig.astype(np.float32)
        sus_f = sus.astype(np.float32)
        dct_orig = cv2.dct(orig_f)
        dct_sus = cv2.dct(sus_f)
        dct_diff = np.abs(dct_orig - dct_sus)
        mean_dct_diff = float(np.mean(dct_diff))
        max_dct_diff = float(np.max(dct_diff))

        h, w = dct_orig.shape
        h_mid_start, h_mid_end = h // 4, (3 * h) // 4
        w_mid_start, w_mid_end = w // 4, (3 * w) // 4
        mid_orig = dct_orig[h_mid_start:h_mid_end, w_mid_start:w_mid_end]
        mid_sus = dct_sus[h_mid_start:h_mid_end, w_mid_start:w_mid_end]
        mid_diff = np.abs(mid_orig - mid_sus)
        mean_mid_diff = float(np.mean(mid_diff))

        # 3) DWT analysis
        coeffs_o = pywt.dwt2(orig_f, 'haar')
        LL_o, (LH_o, HL_o, HH_o) = coeffs_o
        coeffs_s = pywt.dwt2(sus_f, 'haar')
        LL_s, (LH_s, HL_s, HH_s) = coeffs_s

        ll_diff = np.abs(LL_o - LL_s)
        lh_diff = np.abs(LH_o - LH_s)
        hl_diff = np.abs(HL_o - HL_s)
        hh_diff = np.abs(HH_o - HH_s)

        mean_ll_diff = float(np.mean(ll_diff))
        mean_lh_diff = float(np.mean(lh_diff))
        mean_hl_diff = float(np.mean(hl_diff))
        mean_hh_diff = float(np.mean(hh_diff))

        # 4) FFT analysis
        F_o = np.fft.fft2(orig_f)
        F_s = np.fft.fft2(sus_f)
        mag_o = np.abs(F_o)
        mag_s = np.abs(F_s)
        fft_diff = np.abs(mag_o - mag_s)
        mean_fft_diff = float(np.mean(fft_diff))

        # Interpretation
        if mse == 0:
            status = "IDENTICAL IMAGES"
            conclusion = "Images are bit-exact copies. No hidden data detected via comparison."
        elif psnr > 45:
            # Very high PSNR means invisible changes -> Likely Steganography
            status = "SUSPICIOUS: HIGH QUALITY (STEGANOGRAPHY?)"
            conclusion = (
                "Extremely high similarity (PSNR > 45 dB). Differences are invisible to the human eye.\n"
                "However, mathematical differences exist. This is a STRONG indicator of STEGANOGRAPHY.\n"
                "Action: Check the DCT Histogram for distribution anomalies and the Difference Heatmap\n"
                "to locate the hidden payload."
            )
        elif psnr > 35:
            status = "HIGH SIMILARITY"
            conclusion = "Images are very similar. Possible light steganography, subtle editing, or re-saving artifacts."
        elif psnr > 25:
            status = "MODERATE DIFFERENCE"
            conclusion = "Moderate differences detected. Possible editing, recompression, or partial tampering."
        else:
            status = "STRONG DIFFERENCE"
            conclusion = "Strong differences in both spatial and frequency domains. High chance of visible tampering."

        # --- NEW: Save frequency-domain graphs ---
        base_name_sus, _ = os.path.splitext(img_path)
        metrics = {
            "mean_dct_diff": mean_dct_diff,
            "mean_mid_diff": mean_mid_diff,
            "mean_ll_diff": mean_ll_diff,
            "mean_lh_diff": mean_lh_diff,
            "mean_hl_diff": mean_hl_diff,
            "mean_hh_diff": mean_hh_diff,
            "mean_fft_diff": mean_fft_diff,
            "dct_diff": dct_diff,
            "hh_diff": hh_diff,
        }

        try:
            bar_dct_fft_path, bar_dwt_path, dct_heatmap_path, dwt_hh_heatmap_path = \
                save_freq_graphs(metrics, base_name_sus)
            
            # Save DCT Histogram Comparison (New Feature for Steganography)
            dct_hist_path = save_dct_histogram_comparison(dct_orig, dct_sus, base_name_sus)
        except Exception as ge:
            print("Warning: could not save frequency graphs:", ge)
            bar_dct_fft_path = bar_dwt_path = dct_heatmap_path = dwt_hh_heatmap_path = dct_hist_path = None

        # Build report text
        freq_report_path = base_name_sus + "_freq_report.txt"
        lines = []
        lines.append("-----------------------------------------")
        lines.append("IMAGE FORENSICS FREQUENCY-DOMAIN REPORT")
        lines.append("-----------------------------------------")
        lines.append(f"Original image: {original_compare_path}")
        lines.append(f"Suspect image : {img_path}")
        lines.append(f"Date          : {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
        lines.append("")
        lines.append("1. Spatial Domain:")
        lines.append(f"   - MSE  : {mse:.4f}")
        lines.append(f"   - PSNR : {psnr:.2f} dB")
        lines.append("")
        lines.append("2. DCT (Discrete Cosine Transform) Analysis:")
        lines.append(f"   - Mean |DCT diff| overall : {mean_dct_diff:.4f}")
        lines.append(f"   - Max  |DCT diff| overall : {max_dct_diff:.4f}")
        lines.append(f"   - Mean |DCT diff| mid-band: {mean_mid_diff:.4f}")
        lines.append("")
        lines.append("3. DWT (Haar) Analysis:")
        lines.append(f"   - Mean LL diff : {mean_ll_diff:.4f}")
        lines.append(f"   - Mean LH diff : {mean_lh_diff:.4f}")
        lines.append(f"   - Mean HL diff : {mean_hl_diff:.4f}")
        lines.append(f"   - Mean HH diff : {mean_hh_diff:.4f}")
        lines.append("")
        lines.append("4. FFT (Fast Fourier Transform) Analysis:")
        lines.append(f"   - Mean |FFT magnitude diff| : {mean_fft_diff:.4f}")
        lines.append("")
        lines.append("Summary Interpretation:")
        lines.append(f"   - Status     : {status}")
        lines.append(f"   - Conclusion : {conclusion}")
        lines.append("")
        lines.append("Note:")
        lines.append("- Frequency-domain differences can indicate recompression, filtering, splicing or steganographic operations.")
        lines.append("- Combine this with ELA results and visual inspection for stronger conclusions.")
        lines.append("")
        lines.append("Graphical Outputs:")
        if bar_dct_fft_path:
            lines.append(f"- DCT/FFT Difference Bar Chart : {os.path.basename(bar_dct_fft_path)}")
        if bar_dwt_path:
            lines.append(f"- DWT Sub-band Bar Chart       : {os.path.basename(bar_dwt_path)}")
        if dct_heatmap_path:
            lines.append(f"- DCT Difference Heatmap       : {os.path.basename(dct_heatmap_path)}")
        if dwt_hh_heatmap_path:
            lines.append(f"- DWT HH Difference Heatmap    : {os.path.basename(dwt_hh_heatmap_path)}")
        if dct_hist_path:
            lines.append(f"- DCT Histogram Comparison     : {os.path.basename(dct_hist_path)}")
        lines.append("-----------------------------------------")

        with open(freq_report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        messagebox.showinfo(
            "Frequency-Domain Forensics",
            f"Frequency-domain analysis completed.\n\n"
            f"Original : {os.path.basename(original_compare_path)}\n"
            f"Suspect  : {os.path.basename(img_path)}\n"
            f"Report   : {os.path.basename(freq_report_path)}\n\n"
            f"Status: {status}\n\n{conclusion}"
        )
        result_label.config(
            text=f"ELA + Frequency Forensics ready | Freq Report: {os.path.basename(freq_report_path)}",
            fg="blue"
        )
        print(f"Frequency Analysis Done. Report saved: {freq_report_path}")

    except Exception as e:
        print(f"ERROR during Frequency Analysis: {e}")
        messagebox.showerror("Frequency Analysis Error", f"Error during frequency-domain analysis:\n{e}")


# ===========================
#  LSB Steganography Tool
# ===========================

def open_lsb_tool():
    """Open a popup window to generate LSB steganography images."""
    win = Toplevel(root)
    win.title("LSB Steganography Generator")
    win.geometry("500x250")
    win.configure(bg="#f7fafc")

    Label(win, text="Generate LSB Stego Image (Spatial Domain)", font=("Arial", 12, "bold"), bg="#f7fafc").pack(pady=10)

    # State
    cover_path_var = StringVar()

    def select_cover():
        p = filedialog.askopenfilename(
            title="Select Cover Image",
            filetypes=[("Image files", "*.png *.jpg *.bmp")]
        )
        if p:
            cover_path_var.set(p)

    def generate():
        src = cover_path_var.get()
        msg = msg_entry.get()
        if not src or not msg:
            messagebox.showwarning("Missing Info", "Please select a cover image and enter a message.")
            return
        
        out_path = filedialog.asksaveasfilename(
            title="Save Stego Image",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")]
        )
        if not out_path:
            return

        try:
            img_cv = cv2.imread(src)
            if img_cv is None:
                raise ValueError("Could not read cover image.")

            full_msg = msg + "#####"
            binary_msg = ''.join(format(ord(char), '08b') for char in full_msg)
            flat_img = img_cv.flatten()
            
            if len(binary_msg) > len(flat_img):
                raise ValueError("Message too long for this image.")

            for i in range(len(binary_msg)):
                flat_img[i] = (flat_img[i] & ~1) | int(binary_msg[i])
            
            stego_img = flat_img.reshape(img_cv.shape)
            cv2.imwrite(out_path, stego_img)
            messagebox.showinfo("Success", f"Stego image created!\nSaved to: {out_path}")
            win.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed:\n{e}")

    # UI Layout
    f_sel = Frame(win, bg="#f7fafc")
    f_sel.pack(fill=X, padx=20, pady=5)
    Button(f_sel, text="Select Cover Image", command=select_cover).pack(side=LEFT)
    Label(f_sel, textvariable=cover_path_var, bg="#f7fafc", fg="blue").pack(side=LEFT, padx=10)

    f_msg = Frame(win, bg="#f7fafc")
    f_msg.pack(fill=X, padx=20, pady=5)
    Label(f_msg, text="Secret Message:", bg="#f7fafc").pack(side=LEFT)
    msg_entry = Entry(f_msg, width=40)
    msg_entry.pack(side=LEFT, padx=10)

    Button(win, text="Generate Stego Image", command=generate, bg="#4CAF50", fg="white").pack(pady=15)


# ===========================
#  UI Setup
# ===========================

root = Tk()
root.title("Image Forensics: ELA + Frequency-Domain Analysis")
root.geometry("1200x820")
root.configure(bg="#f7fafc")

top_frame = Frame(root, bg="#f7fafc")
top_frame.pack(fill=X, pady=8)

Label(
    top_frame,
    text="IMAGE FORENSICS (ELA + DCT + DWT + FFT)",
    font=("Arial", 16, "bold"),
    bg="#f7fafc"
).pack(side=LEFT, padx=12)

controls_frame = Frame(root, bg="#f7fafc")
controls_frame.pack(fill=X, pady=6)

# Start directory display and change button
start_folder_label = Label(controls_frame, text="Start folder: " + start_dir, bg="#f7fafc")
start_folder_label.grid(row=0, column=0, padx=8, sticky=W)
Button(controls_frame, text="Change Start Folder", command=choose_start_folder, width=18).grid(row=0, column=1, padx=6)

# ELA quality and threshold k
ela_quality_var = IntVar(value=90)
threshold_k_var = DoubleVar(value=1.0)
Label(controls_frame, text="ELA JPEG quality:", bg="#f7fafc").grid(row=0, column=2, padx=6)
Spinbox(controls_frame, from_=10, to=100, textvariable=ela_quality_var, width=6).grid(row=0, column=3)
Label(controls_frame, text="Threshold k (mean + k*std):", bg="#f7fafc").grid(row=0, column=4, padx=6)
Spinbox(controls_frame, from_=0.0, to=5.0, increment=0.1, textvariable=threshold_k_var, width=6).grid(row=0, column=5)

# Original (for frequency compare) info
original_label = Label(controls_frame, text="Original for Compare: (none selected)", bg="#f7fafc")
original_label.grid(row=1, column=0, padx=8, pady=4, sticky=W)
Button(controls_frame, text="Upload Original Image", command=upload_original_image, width=18).grid(row=1, column=1, padx=6, pady=4)

preview_frame = Frame(root, bg="#f7fafc")
preview_frame.pack(pady=6)

panel = Label(preview_frame, text="Suspect Image", width=54, height=22,
              bg="lightgray", anchor=CENTER, justify=CENTER)
panel.grid(row=0, column=0, padx=10)

ela_panel = Label(preview_frame, text="ELA Result", width=54, height=22,
                  bg="lightgray", anchor=CENTER, justify=CENTER)
ela_panel.grid(row=0, column=1, padx=10)

overlay_panel = Label(preview_frame, text="Overlay (suspected areas)", width=54, height=22,
                      bg="lightgray", anchor=CENTER, justify=CENTER)
overlay_panel.grid(row=0, column=2, padx=10)

btn_frame = Frame(root, bg="#f7fafc")
btn_frame.pack(pady=10)

Button(btn_frame, text="Upload Suspect Image", command=upload_image,
       font=("Arial", 12), bg="#008CBA", fg="white", width=20).grid(row=0, column=0, padx=8)

Button(btn_frame, text="Detect Forgery + ELA Report", command=detect_forgery_and_report,
       font=("Arial", 12), bg="#4CAF50", fg="white", width=24).grid(row=0, column=1, padx=8)

Button(btn_frame, text="Frequency Analysis (Original vs Suspect)", command=frequency_analysis_compare,
       font=("Arial", 12), bg="#6C63FF", fg="white", width=32).grid(row=0, column=2, padx=8)

Button(btn_frame, text="LSB Stego Tool", command=open_lsb_tool,
       font=("Arial", 12), bg="#FF9800", fg="white", width=16).grid(row=0, column=3, padx=8)

result_label = Label(root, text="", font=("Arial", 12, "bold"), bg="#f7fafc")
result_label.pack(pady=8)

hint = Label(
    root,
    text=(
        "ELA Interpretation Guide:\n"
        "- DARK areas = Original. BRIGHT areas = Modified (Watermarks/Edits).\n"
        "- Why? Added objects compress differently than the background.\n"
        "- If you see a glowing shape 'behind' the image, it was likely added later."
    ),
    bg="#f7fafc",
    font=("Arial", 10),
    justify=LEFT
)
hint.pack(pady=2)

print("GUI constructed. Launching window...")
root.mainloop()
