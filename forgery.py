# ela_full_report_gui.py
"""
Image Forgery Detection (ELA) - Full Report GUI
Saves:
 - <original_base>_ela_q<q>_result.jpg       (ELA visualization)
 - <original_base>_overlay_boxes.jpg         (overlay on original with boxes)
 - <original_base>_ela_report.txt            (text report with details & human-friendly conclusion)

Requires:
 pip3 install pillow numpy
Run:
 python3 ela_full_report_gui.py
"""

from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageChops, ImageEnhance, ImageDraw
import os
import tempfile
import numpy as np
import datetime

# Globals
img_path = None
img = None

def choose_start_folder():
    """Let user change the starting folder for the file dialog."""
    global start_dir
    d = filedialog.askdirectory(title="Choose start directory for dialogs")
    if d:
        start_dir = d
        start_folder_label.config(text="Start folder: " + start_dir)

def upload_image():
    global img_path, img
    default_dir = start_dir if os.path.isdir(start_dir) else os.path.expanduser("~")

    img_path = filedialog.askopenfilename(
        title="Select an image",
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
    result_label.config(text="Image loaded: " + os.path.basename(img_path), fg="black")
    ela_panel.config(image="", text="ELA Result")
    ela_panel.image = None
    overlay_panel.config(image="", text="Overlay Result")
    overlay_panel.image = None

def make_report_lines(basename, ela_quality, width, height, max_diff, mean_diff, std_diff, threshold_val, flagged_pixels, total_pixels, boxes, percent_flagged):
    """Return formatted list of lines to write to report file."""
    lines = []
    lines.append("-----------------------------------------")
    lines.append("IMAGE FORGERY DETECTION SYSTEM REPORT")
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
    # Human-friendly conclusion using percent_flagged and other metrics
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
    lines.append("Notes:")
    lines.append("- ELA highlights compression inconsistency; it is an evidence tool, not absolute proof.")
    lines.append("- For better accuracy, compare with known original or use multiple forensic techniques.")
    lines.append("-----------------------------------------")
    lines.append(f"Result Status: {conclusion_text}")
    lines.append("-----------------------------------------")
    return lines

def detect_forgery_and_report():
    global img_path, img
    if not img_path or img is None:
        result_label.config(text="Please upload an image first!", fg="red")
        return

    try:
        # Create temp file for JPEG save
        fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)

        # Convert to RGBA first to handle alpha; then to RGB for JPEG
        base_rgba = img.convert("RGBA")
        if base_rgba.mode == "RGBA":
            background = Image.new("RGBA", base_rgba.size, (255, 255, 255, 255))
            background.paste(base_rgba, mask=base_rgba.split()[3])
            rgb_img = background.convert("RGB")
        else:
            rgb_img = base_rgba.convert("RGB")

        # Save JPEG re-encoded
        ela_quality = ela_quality_var.get()
        if not (10 <= ela_quality <= 100):
            ela_quality = 90

        rgb_img.save(temp_path, 'JPEG', quality=ela_quality)

        # Reload saved jpeg and compute difference
        reloaded = Image.open(temp_path).convert("RGB")
        diff_img = ImageChops.difference(rgb_img, reloaded)

        # Convert diff to numpy array (uint8)
        diff_np = np.array(diff_img).astype(np.uint8)  # shape (H,W,3)

        # Compute per-pixel intensity (max across channels)
        diff_gray = diff_np.max(axis=2)  # shape (H,W), 0..255

        # Stats
        max_diff = int(diff_gray.max())
        mean_diff = float(diff_gray.mean())
        std_diff = float(diff_gray.std())

        # Scale differences to use full 0..255
        scale = (255.0 / max_diff) if max_diff != 0 else 1.0
        diff_scaled = np.clip((diff_gray.astype(np.float32) * scale), 0, 255).astype(np.uint8)

        # ELA image preparation and enhancements
        ela_image = Image.fromarray(diff_scaled).convert("RGB")
        ela_image = ImageEnhance.Contrast(ela_image).enhance(1.6)
        ela_image = ImageEnhance.Brightness(ela_image).enhance(1.05)

        # Save ELA image
        base_name, ext = os.path.splitext(img_path)
        ela_out_path = base_name + f"_ela_q{ela_quality}_result.jpg"
        ela_image.save(ela_out_path)

        # Thresholding to detect suspicious pixels
        k = threshold_k_var.get()
        threshold_val = mean_diff + k * std_diff
        threshold_val = min(max(threshold_val, 5), 200)
        mask = (diff_gray >= threshold_val).astype(np.uint8) * 255  # 0 or 255

        total_pixels = mask.size
        flagged_pixels = int((mask > 0).sum())
        percent_flagged = (flagged_pixels / total_pixels) * 100

        # Connected components (simple flood fill) to get bounding boxes
        labeled = np.zeros_like(mask, dtype=np.int32)
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
                        for ny in range(max(0, cy-1), min(height, cy+2)):
                            for nx in range(max(0, cx-1), min(width, cx+2)):
                                if mask[ny, nx] and not visited[ny, nx]:
                                    visited[ny, nx] = 1
                                    stack.append((ny, nx))
                    boxes.append({
                        "bbox": (minx, miny, maxx, maxy),
                        "area_pixels": area_count
                    })

        # Create overlay image: red mask blended on original
        orig_np = np.array(rgb_img).astype(np.uint8)
        overlay_np = orig_np.copy()
        overlay_layer = np.zeros((height, width, 3), dtype=np.uint8)
        overlay_layer[:, :, 0] = mask  # red channel
        alpha = 0.45
        blended = np.clip((overlay_np * (1 - alpha) + overlay_layer * alpha), 0, 255).astype(np.uint8)
        overlay_img = Image.fromarray(blended)

        # Draw bounding boxes (yellow) for each region
        draw = ImageDraw.Draw(overlay_img)
        for b in boxes:
            minx, miny, maxx, maxy = b["bbox"]
            pad = max(2, int(0.01 * max(width, height)))
            draw.rectangle([minx-pad, miny-pad, maxx+pad, maxy+pad], outline=(255, 255, 0), width=2)

        overlay_with_boxes_path = base_name + "_overlay_boxes.jpg"
        overlay_img.save(overlay_with_boxes_path)

        # Compose textual report
        report_lines = make_report_lines(base_name, ela_quality, width, height,
                                         max_diff, mean_diff, std_diff,
                                         threshold_val, flagged_pixels, total_pixels, boxes, percent_flagged)
        report_path = base_name + "_ela_report.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        # Show previews in GUI
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

        # Show final result summary in GUI
        # Create the friendly conclusion string similar to report
        if percent_flagged < 0.05:
            conclusion_text = "ORIGINAL IMAGE"
            human_readable = "No significant tampering found. The image appears to be ORIGINAL or contains only very small noise."
        elif percent_flagged < 1.0:
            conclusion_text = "POSSIBLE MINOR EDITS"
            human_readable = "Small areas flagged. Could be minor edits or compression artifacts. Manual review recommended."
        else:
            conclusion_text = "LIKELY FORGED"
            human_readable = "Significant flagged area detected. The image is likely forged — review overlay images."

        result_label.config(text=f"Done: {os.path.basename(ela_out_path)}  |  Report: {os.path.basename(report_path)}  |  Status: {conclusion_text}", fg="green")
        # Also show a small popup with summary
        messagebox.showinfo("ELA Report Summary",
                            f"ELA done.\nELA image: {os.path.basename(ela_out_path)}\nOverlay: {os.path.basename(overlay_with_boxes_path)}\nReport: {os.path.basename(report_path)}\n\nStatus: {conclusion_text}\n\n{human_readable}")

    except Exception as e:
        messagebox.showerror("Detection Error", f"Error during ELA/report:\n{e}")
        result_label.config(text="Error during ELA/report. See message.", fg="red")
    finally:
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass

# UI Setup
root = Tk()
root.title("Image Forgery Detection (ELA + Full Report)")
root.geometry("1200x780")
root.configure(bg="#f7fafc")

top_frame = Frame(root, bg="#f7fafc")
top_frame.pack(fill=X, pady=8)

Label(top_frame, text="IMAGE FORGERY DETECTION (ELA + DETAILED REPORT)", font=("Arial", 16, "bold"), bg="#f7fafc").pack(side=LEFT, padx=12)

controls_frame = Frame(root, bg="#f7fafc")
controls_frame.pack(fill=X, pady=6)

# Start directory display and change button
start_dir = os.path.expanduser("~")
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

preview_frame = Frame(root, bg="#f7fafc")
preview_frame.pack(pady=6)

panel = Label(preview_frame, text="Original Image", width=54, height=22, bg="lightgray", anchor=CENTER, justify=CENTER)
panel.grid(row=0, column=0, padx=10)

ela_panel = Label(preview_frame, text="ELA Result", width=54, height=22, bg="lightgray", anchor=CENTER, justify=CENTER)
ela_panel.grid(row=0, column=1, padx=10)

overlay_panel = Label(preview_frame, text="Overlay (suspected areas)", width=54, height=22, bg="lightgray", anchor=CENTER, justify=CENTER)
overlay_panel.grid(row=0, column=2, padx=10)

btn_frame = Frame(root, bg="#f7fafc")
btn_frame.pack(pady=10)

Button(btn_frame, text="Upload Image", command=upload_image, font=("Arial", 12), bg="#008CBA", fg="white", width=16).grid(row=0, column=0, padx=8)
Button(btn_frame, text="Detect Forgery + Save Report", command=detect_forgery_and_report, font=("Arial", 12), bg="#4CAF50", fg="white", width=24).grid(row=0, column=1, padx=8)

result_label = Label(root, text="", font=("Arial", 12, "bold"), bg="#f7fafc")
result_label.pack(pady=8)

hint = Label(root, text="Hint: ELA highlights compression differences. Lower threshold k to be more sensitive; increase to reduce false positives.\nOutputs are saved next to the original image file.", bg="#f7fafc", font=("Arial", 10))
hint.pack(pady=2)

root.mainloop()
