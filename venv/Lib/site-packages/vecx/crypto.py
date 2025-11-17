
import json
import zlib

def get_checksum(key:str|None)->int:
    # Convert last two characters of key to integer
    if key is None:
        return -1
    return int(key[-2:], 16)

def json_zip(dict):
    if not dict:
        return b''
    json_dict = json.dumps(dict).encode('utf-8')
    compressed_dict = zlib.compress(json_dict)
    return compressed_dict

def json_unzip(compressed_dict):
    if not compressed_dict:
        return {}
    decompressed_dict = zlib.decompress(compressed_dict)
    return json.loads(decompressed_dict.decode('utf-8'))