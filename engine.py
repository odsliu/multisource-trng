class Engine:
    def __init__(self,name):
        self._name = name

    def get_bits(self,*args) -> bytes:
        return b''