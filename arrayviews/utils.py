
def get_bitmap(isnull):
    """Pack is-null bytearray to is-null bitarray.

    Parameters
    ----------
    isnull : np.ndarray
      Specify is-null bytearray where 0 values correspond to nulls.

    Returns
    -------
    bitmap : np.ndarray
      is-null bitarray
    """
    import numpy as np
    bitlst = []
    for i in range(1+len(isnull)//8):
        bits = ''.join('10'[bool(b)] for b in isnull[i*8:i*8+8])
        if len(bits) < 8:
            bits = bits + '0'*(8 - len(bits))
        bits = int(bits[::-1], base=2)
        bitlst.append(bits)
    bitmap = np.array(bitlst, dtype=np.uint8)
    return bitmap
