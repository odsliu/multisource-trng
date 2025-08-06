import struct

from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
import ctypes
from rdengine import radioEngine
from rdrandengine import rdrandEngine
import math
import sys
import threading
import time
import hashlib

from pyaudio import *
import wave
import numpy as np

# CHUNK = 2 ** 5
# RATE = 44100
# RPOOL = b''

# p = PyAudio()
#
# stream = p.open(
#     format=paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK
# )

# lib = ctypes.CDLL('D:/pylocal/pythonProject-radio-noise-trng/rdrand.dll')
# lib.rdrand64.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
# lib.rdrand64.restype = ctypes.c_int


def derive_large_key(ikm, length, salt=None, info=b"", hash_algorithm=hashes.SHA256()):
    # Maximum output per chunk (255 * hash digest size)
    max_chunk_size = 255 * hash_algorithm.digest_size  # 8160 for SHA256

    if length <= max_chunk_size:
        # Use standard HKDF for keys within the limit
        return HKDF(
            algorithm=hash_algorithm,
            length=length,
            salt=salt,
            info=info,
        ).derive(ikm)

    # Step 1: Extract PRK using HMAC
    digest_size = hash_algorithm.digest_size
    if salt is None:
        salt = b"\x00" * digest_size  # Default salt if not provided
    prk = hmac.HMAC(salt, hash_algorithm, backend=default_backend())
    prk.update(ikm)
    prk = prk.finalize()

    # Step 2: Derive key in chunks
    derived_key = b""
    chunk_index = 0
    while length > 0:
        # Create unique info for this chunk: base_info + chunk_index (4-byte big-endian)
        chunk_info = info + struct.pack(">I", chunk_index)
        chunk_length = min(length, max_chunk_size)

        # Derive chunk using HKDF-Expand logic
        chunk = b""
        block = b""
        counter = 1
        while len(chunk) < chunk_length:
            h = hmac.HMAC(prk, hash_algorithm, backend=default_backend())
            h.update(block)  # Previous block (T(i-1))
            h.update(chunk_info)
            h.update(struct.pack("B", counter))  # Counter byte
            block = h.finalize()
            chunk += block
            counter += 1
        chunk = chunk[:chunk_length]

        # Append chunk and update state
        derived_key += chunk
        length -= chunk_length
        chunk_index += 1

    return derived_key


def merge_entropy(entropy1: bytes, entropy2: bytes, output_length: int = 32, info: bytes = b'modwdjt') -> bytes:
    """
    Merges two entropy sources using HKDF for random number generation seeding.

    Args:
        entropy1: First entropy source (bytes).
        entropy2: Second entropy source (bytes).
        output_length: Desired output length in bytes (default: 32).
        info: Context-specific application info (default: b'RNG_merge').

    Returns:
        Derived key bytes of length `output_length`.
    """
    # Use entropy2 as salt (if non-empty), else default to None (auto-salt with zeros)
    salt = entropy2 if entropy2 else None

    # Initialize HKDF with SHA-256
    # hkdf = HKDF(
    #     algorithm=hashes.SHA256(),
    #     length=output_length,
    #     salt=salt,
    #     info=info,
    #     backend=default_backend()
    # )
    # a = derive_large_key(entropy1,output_length,salt,info,hashes.SHA256())
    # Derive key using entropy1 as input key material (IKM)
    return derive_large_key(entropy1, output_length, salt, info, hashes.SHA256())


def bytes_to_bits_binary(byte_data):
    bits_data = bin(int.from_bytes(byte_data, byteorder='big'))[2:]
    return bits_data


# def get_rdseed():
#     global RPOOL
#     while True:
#         seed_val = ctypes.c_uint64(0)
#         success = lib.rdrand64(ctypes.byref(seed_val))
#         if not success:
#             raise RuntimeError("RDRAND failed Entropy may not be ready")
#         RPOOL += int(bin(seed_val.value), 2).to_bytes((len(bin(seed_val.value)) + 7) // 8, byteorder='big')


# rdpoolt = threading.Thread(target=get_rdseed)
# rdpoolt.start()


def calc_bits(maximum, minimum):
    x = math.floor(math.log2(maximum - minimum) + 1)
    return x


def process(maximum, minimum):
    global bits
    r = ''
    x = calc_bits(maximum, minimum)
    # print(x)

    while r == '' or int(r, 2) > (maximum - minimum):
        r = ''
        for _ in range(x):
            # print(bits)
            r += bits[0]
            bits = bits[1:]
            # print(r)
    r = int(r, 2)
    return minimum + r


if __name__ == '__main__':
    radio_engine = radioEngine(2 ** 5,44100)
    rdrand_engine = rdrandEngine()
    a = b''
    ma, mi = int(input('max: ')), int(input('min: '))
    num = int(input('n: '))

    LEN = num / 1666

    bits = radio_engine.get_bits(LEN)
    rdbits = rdrand_engine.get_bits()[0]
    bits = merge_entropy(bits, rdbits, output_length=len(bits))

    # for _ in range(num):
    #     a += (bits[0:])
    #     bits = bits[1:]
    #     print(_)

    # for i in range((ma-mi) + 1):
    #     print(f'counts{i + mi}: ' + str(a.count(i + mi)))

    print(sum(a)/num)
    print(a)
    with open('bin.bin', 'wb') as b:
        b.write(bits)

    # import matplotlib.pyplot as plt
    # data = a
    #
    # # Line Plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(data, color='blue', linewidth=1)
    # plt.title(f'Line Plot of {num} Random Numbers')
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.savefig('line_plot.png')  # Optional: Save the figure
    # plt.show()
    #
    # # Scatter Plot
    # plt.figure(figsize=(10, 6))
    # plt.scatter(list(range(num)), data, s=10, color='red', alpha=0.7)
    # plt.title('Scatter Plot of 500 Random Numbers')
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.savefig('scatter_plot.png')
    # plt.show()
    #
    # # Histogram
    # plt.figure(figsize=(10, 8))
    # plt.hist(data, bins=20, color='green', edgecolor='black', alpha=0.7)
    # plt.title('Histogram of 500 Random Numbers')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    # plt.savefig('histogram.png')
    # plt.show()

    # radioEngine.stream.stop_stream()
    # stream.close()
    # p.terminate()
