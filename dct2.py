import cv2
import numpy as np

# Same JPEG-like Quantization Table
Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])

def extract_text(stego_path):
    # Read stego image in grayscale
    img = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: stego image not found or cannot be opened.")
        return ""

    h, w = img.shape
    binary = ""
    message = ""

    # Go block by block (8x8)
    for y in range(0, h, 8):
        for x in range(0, w, 8):

            block = img[y:y+8, x:x+8]
            if block.shape[0] < 8 or block.shape[1] < 8:
                continue

            # DCT
            dct_block = cv2.dct(np.float32(block))

            # Quantize like in embedding
            q_block = np.round(dct_block / Q)

            # Extract LSB from the same coefficient [4,3]
            bit = int(q_block[4, 3]) & 1
            binary += str(bit)

            # Every 8 bits -> 1 character
            if len(binary) % 8 == 0:
                byte = binary[-8:]
                try:
                    char = chr(int(byte, 2))
                except ValueError:
                    continue

                message += char

                # Check for end marker "#####"
                if message.endswith("#####"):
                    # Remove marker and return clean message
                    return message[:-5]

    return message  # in case marker not found


if __name__ == "__main__":
    stego_name = input("Enter stego image filename (ex: stego.png): ")
    secret = extract_text(stego_name)
    print("Extracted message:", secret)
