"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class outputs_2_t(object):
    __slots__ = ["foo"]

    __typenames__ = ["double"]

    __dimensions__ = [None]

    def __init__(self):
        self.foo = 0.0

    def encode(self):
        buf = BytesIO()
        buf.write(outputs_2_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">d", self.foo))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != outputs_2_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return outputs_2_t._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = outputs_2_t()
        self.foo = struct.unpack(">d", buf.read(8))[0]
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if outputs_2_t in parents: return 0
        tmphash = (0x2acd1c65693943de) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff) + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if outputs_2_t._packed_fingerprint is None:
            outputs_2_t._packed_fingerprint = struct.pack(">Q", outputs_2_t._get_hash_recursive([]))
        return outputs_2_t._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)
