# forensics_graphs.py
"""
Helper functions to save graphical representations for image forensics:
- ELA histograms / pie chart
- Frequency-domain (DCT / DWT / FFT) bar charts and heatmaps
"""

import matplotlib
matplotlib.use("Agg")  # so it works even without a GUI display (Kali, servers)

import matplotlib.pyplot as plt
import numpy as np


def save_ela_graphs(diff_gray: np.ndarray, mask: np.ndarray, base_name: str):
    """
    Create:
      - Histogram of ELA difference values
      - Pie chart of suspicious vs normal pixels

    Saves:
      <base_name>_ela_histogram.png
      <base_name>_ela_pie.png

    Returns:
      (hist_path, pie_path)
    """

    # 1) Histogram of ELA differences
    hist_path = base_name + "_ela_histogram.png"
    plt.figure()
    plt.hist(diff_gray.flatten(), bins=50)
    plt.xlabel("Difference value (0–255)")
    plt.ylabel("Pixel count")
    plt.title("ELA Difference Histogram")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()

    # 2) Pie chart: normal vs suspicious pixels
    total_pixels = mask.size
    flagged_pixels = int((mask > 0).sum())
    normal_pixels = total_pixels - flagged_pixels

    pie_path = base_name + "_ela_pie.png"
    plt.figure()
    plt.pie(
        [normal_pixels, flagged_pixels],
        labels=["Normal Pixels", "Suspicious Pixels"],
        autopct="%1.2f%%",
        startangle=90,
    )
    plt.title("ELA – Normal vs Suspicious Pixels")
    plt.tight_layout()
    plt.savefig(pie_path)
    plt.close()

    return hist_path, pie_path


def save_freq_graphs(metrics: dict, base_name: str):
    """
    Create bar charts & heatmaps for frequency-domain differences.

    metrics must contain:
      mean_dct_diff, mean_mid_diff,
      mean_ll_diff, mean_lh_diff, mean_hl_diff, mean_hh_diff,
      mean_fft_diff,
      dct_diff (2D array),
      hh_diff  (2D array)

    Saves:
      <base_name>_freq_bar_dct_fft.png
      <base_name>_freq_bar_dwt.png
      <base_name>_dct_heatmap.png
      <base_name>_dwt_hh_heatmap.png

    Returns:
      (bar_dct_fft_path, bar_dwt_path, dct_heatmap_path, dwt_hh_heatmap_path)
    """

    # --- Bar chart: DCT + FFT ---
    bar_dct_fft_path = base_name + "_freq_bar_dct_fft.png"
    methods = ["DCT overall", "DCT mid-band", "FFT magnitude"]
    values = [
        metrics["mean_dct_diff"],
        metrics["mean_mid_diff"],
        metrics["mean_fft_diff"],
    ]

    plt.figure()
    plt.bar(methods, values)
    plt.ylabel("Mean difference")
    plt.title("Frequency-Domain Differences (DCT / FFT)")
    plt.tight_layout()
    plt.savefig(bar_dct_fft_path)
    plt.close()

    # --- Bar chart: DWT bands ---
    bar_dwt_path = base_name + "_freq_bar_dwt.png"
    bands = ["LL", "LH", "HL", "HH"]
    vals = [
        metrics["mean_ll_diff"],
        metrics["mean_lh_diff"],
        metrics["mean_hl_diff"],
        metrics["mean_hh_diff"],
    ]

    plt.figure()
    plt.bar(bands, vals)
    plt.ylabel("Mean difference")
    plt.title("DWT Sub-band Differences")
    plt.tight_layout()
    plt.savefig(bar_dwt_path)
    plt.close()

    # --- Heatmap: full DCT difference ---
    dct_heatmap_path = base_name + "_dct_heatmap.png"
    dct_diff = metrics["dct_diff"]
    plt.figure()
    plt.imshow(dct_diff, cmap="hot", interpolation="nearest")
    plt.colorbar(label="|DCT diff|")
    plt.title("DCT Difference Heatmap")
    plt.tight_layout()
    plt.savefig(dct_heatmap_path)
    plt.close()

    # --- Heatmap: DWT HH difference ---
    dwt_hh_heatmap_path = base_name + "_dwt_hh_heatmap.png"
    hh_diff = metrics["hh_diff"]
    plt.figure()
    plt.imshow(hh_diff, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="|DWT HH diff|")
    plt.title("DWT HH Sub-band Difference Heatmap")
    plt.tight_layout()
    plt.savefig(dwt_hh_heatmap_path)
    plt.close()

    return bar_dct_fft_path, bar_dwt_path, dct_heatmap_path, dwt_hh_heatmap_path

def save_dct_histogram_comparison(dct_orig: np.ndarray, dct_sus: np.ndarray, base_name: str):
    """
    Create a histogram comparison of DCT coefficients.
    Steganography often alters the distribution of coefficients.
    """
    path = base_name + "_dct_hist_compare.png"
    
    # We use log scale to visualize the differences in smaller coefficients better
    # Flatten arrays and take log(1 + abs(x))
    data_orig = np.log1p(np.abs(dct_orig.flatten()))
    data_sus = np.log1p(np.abs(dct_sus.flatten()))

    plt.figure()
    plt.hist(data_orig, bins=100, alpha=0.5, label='Original', color='blue', log=True)
    plt.hist(data_sus, bins=100, alpha=0.5, label='Suspect', color='red', log=True)
    plt.legend()
    plt.title("DCT Coefficient Distribution (Log Scale)")
    plt.xlabel("Log Magnitude")
    plt.ylabel("Count (Log Scale)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    
    return path

if __name__ == "__main__":
    print("This file is a library module (forensics_graphs.py).")
    print("It does not show a GUI when run directly.")
    print("Please run 'image_forensics_gui.py' instead.")
