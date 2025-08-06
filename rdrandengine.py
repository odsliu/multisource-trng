import ctypes
import threading

from engine import Engine


class rdrandEngine(Engine):

    def __init__(self):
        super().__init__('rdrand')
        self.lib = ctypes.CDLL('D:/pylocal/pythonProject-radio-noise-trng/rdrand.dll')
        self.lib.rdrand64.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        self.lib.rdrand64.restype = ctypes.c_int
        self.RPOOL = b''
        self.pthread = threading.Thread(target=lambda:self.preserve())
        self.pthread.start()

    def preserve(self):
        while True:
            seed_val = ctypes.c_uint64(0)
            success = self.lib.rdrand64(ctypes.byref(seed_val))
            if not success:
                raise RuntimeError("RDRAND failed Entropy may not be ready")
            self.RPOOL += int(bin(seed_val.value), 2).to_bytes((len(bin(seed_val.value)) + 7) // 8, byteorder='big')

    def bytes_to_bits_binary(self, byte_data):
        bits_data = bin(int.from_bytes(byte_data, byteorder='big'))[2:]
        return bits_data

    def clean(self):
        self.RPOOL = b''

    def get_bits(self):
        return self.RPOOL,self.clean()