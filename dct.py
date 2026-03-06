import cv2
import numpy as np

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

def embed_text(cover_path, message, out_path="stego.png"):
    message += "#####"

    bits = ''.join([format(ord(c), '08b') for c in message])
    bit_id = 0
    bit_len = len(bits)

    img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    stego = img.copy()

    for y in range(0, h, 8):
        for x in range(0, w, 8):

            if bit_id >= bit_len:
                break

            block = np.float32(stego[y:y+8, x:x+8])

            dct_block = cv2.dct(block)
            q_block = np.round(dct_block / Q)

            cy, cx = 4, 3
            coeff = int(q_block[cy, cx])
            bit = int(bits[bit_id])
            q_block[cy, cx] = (coeff & ~1) | bit
            bit_id += 1

            deq = q_block * Q
            stego_block = cv2.idct(deq)
            stego[y:y+8, x:x+8] = np.uint8(np.clip(stego_block, 0, 255))

    cv2.imwrite(out_path, stego)
    print("Message embedded into", out_path)


# -------- USER INPUT --------
img_name = input("Enter cover image filename (ex: dog.png): ")
msg = input("Enter secret message: ")

embed_text(img_name, msg, "stego.png")
