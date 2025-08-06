import wave

from pyaudio import PyAudio, paInt16

from engine import Engine

class radioEngine(Engine):

    def __init__(self,chunk,rate):
        super().__init__('radio')
        self.p = PyAudio()
        self.CHUNK = chunk
        self.RATE = rate

        self.stream = self.p.open(
            format=paInt16, channels=1, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK
        )

    def bytes_to_bits_binary(self,byte_data):
        bits_data = bin(int.from_bytes(byte_data, byteorder='big'))[2:]
        return bits_data

    def get_bits(self,length) -> bytes:
        res = b''

        for _ in range(int(length * self.RATE / self.CHUNK)):  # go for a LEN seconds
            # data = np.frombuffer(stream.read(CHUNK,exception_on_overflow=False), dtype=np.int16)
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            res = res + data

        # print('res: ' , res)
        # print(bytes_to_bits_binary(res),'\n',bytes_to_bits_binary(res).count('0'),bytes_to_bits_binary(res).count('1'))
        # a =  bytes_to_bits_binary(res)
        wf = wave.open('wav3.wav', 'wb')

        wf.setnchannels(1)
        wf.setsampwidth(self.p.get_sample_size(paInt16))
        wf.setframerate(44100)

        wf.writeframes(res)
        wf.close()

        with open('wav3.wav', 'rb') as f:
            aun = f.read()

        binary = self.bytes_to_bits_binary(aun[44:])
        # for i in range(0, len(a), 16):
        #     print(a[i:i + 16])
        # print(a)
        randbits = ''
        pos = 0
        for bit in binary:
            if pos % 16 == 0:
                randbits += bit
            pos += 1
        # print(aun)
        randbytes = int(randbits, 2).to_bytes((len(randbits) + 7) // 8, byteorder='big')

        # randbytes = merge_entropy(randbytes, RPOOL, output_length=len(randbytes))
        return randbytes

