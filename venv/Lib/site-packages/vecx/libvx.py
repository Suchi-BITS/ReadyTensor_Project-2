import os
import platform
import numpy as np
import cffi
from functools import lru_cache

ffi = cffi.FFI()

# Define the function signatures expected from the shared library
ffi.cdef("""
    typedef struct VecX VecX;

    VecX *vecx_create(const char *key, const char *lib_token, const char *space_type, int version, int dimension);
    void vecx_destroy(VecX *obj);

    void vecx_encrypt_vector(VecX *obj, const float *input, float *output, int dimension);
    void vecx_decrypt_vector(VecX *obj, const float *input, float *output, int dimension);
    
    void vecx_encrypt_vectors(VecX *obj, const float *input, float *output, int dimension, int num_vectors);
    void vecx_decrypt_vectors(VecX *obj, const float *input, float *output, int dimension, int num_vectors);
    void vecx_decrypt_and_calculate_similarities(VecX *obj, const float* query_vector, const float* encrypted_vectors,
                                                  int dimension, int num_vectors, float* out_similarities);

    unsigned char *vecx_encrypt_meta(VecX *obj, const unsigned char *input, int length, int *out_length);
    unsigned char *vecx_decrypt_meta(VecX *obj, const unsigned char *input, int length, int *out_length);
    void vecx_free_buffer(unsigned char *buffer);

    void vecx_calculate_distances(VecX* obj, const float* query_vector, int dimension,
                                  const float* input_vectors, int num_vectors, float* out_distances);
""")



@lru_cache(maxsize=0)
def load_libvx():
    """Loads the correct shared library based on the OS and architecture."""
    system = platform.system().lower()
    arch = platform.machine().lower()

    if system == "linux":
        if arch in ["x86_64", "amd64"]:
            library_name = "libvx_x86_64.so"
        elif arch in ["aarch64", "arm64", "armv8l"]:
            library_name = "libvx_aarch64.so"
        else:
            raise Exception(f"Unsupported Linux architecture: {arch}")
    elif system == "darwin":
        library_name = "libvx_x86_64.dylib" if arch == "x86_64" else "libvx_arm64.dylib"
    elif system == "windows":
        library_name = "libvx_x86_64.dll"
    else:
        raise Exception(f"Unsupported operating system: {system}")
    
    library_path = os.path.join(os.path.dirname(__file__), "libvx", library_name)
    if not os.path.exists(library_path):
        raise Exception(f"Library file not found: {library_path}")
    
    print(f"Loading library: {library_path}")
    return ffi.dlopen(library_path)

class LibVectorX:
    def __init__(self, key:str, lib_token:str, dimension:int, space_type:str="cosine", version:int=1):
        self.key = key.encode('utf-8')
        self.lib_token = lib_token.encode('utf-8')
        self.dimension = dimension
        self.space_type = space_type.encode('utf-8')
        self.version = version
        self.lib = load_libvx()
        
        self.vecx = self.lib.vecx_create(self.key, self.lib_token, self.space_type, self.version, self.dimension)
        if not self.vecx:
            raise RuntimeError("Failed to initialize VectorX instance")
    
    def __del__(self):
        if self.vecx:
            self.lib.vecx_destroy(self.vecx)
    
    def encrypt_vector(self, vector):
        input_array = np.array(vector, dtype=np.float32)
        output_array = np.zeros(self.dimension, dtype=np.float32)
        self.lib.vecx_encrypt_vector(self.vecx, ffi.cast("float*", input_array.ctypes.data), ffi.cast("float*", output_array.ctypes.data), self.dimension)
        return output_array
    
    def decrypt_vector(self, vector):
        input_array = np.array(vector, dtype=np.float32)
        output_array = np.zeros(self.dimension, dtype=np.float32)
        self.lib.vecx_decrypt_vector(self.vecx, ffi.cast("float*", input_array.ctypes.data), ffi.cast("float*", output_array.ctypes.data), self.dimension)
        return output_array
    
    def encrypt_meta(self, meta):
        input_array = np.frombuffer(meta, dtype=np.uint8)
        output_array = np.zeros_like(input_array, dtype=np.uint8)
        self.lib.vecx_encrypt_meta(self.vecx, ffi.cast("unsigned char*", input_array.ctypes.data), ffi.cast("unsigned char*", output_array.ctypes.data), len(input_array))
        return output_array.tobytes()
    
    def decrypt_meta(self, meta):
        input_array = np.frombuffer(meta, dtype=np.uint8)
        output_array = np.zeros_like(input_array, dtype=np.uint8)
        self.lib.vecx_decrypt_meta(self.vecx, ffi.cast("unsigned char*", input_array.ctypes.data), ffi.cast("unsigned char*", output_array.ctypes.data), len(input_array))
        return output_array.tobytes()
    
    def calculate_distances(self, query_vector, vectors):
        query_array = np.array(query_vector, dtype=np.float32)
        vectors_array = np.array(vectors, dtype=np.float32)
        num_vectors = vectors_array.shape[0]

        out_distances = np.empty(num_vectors, dtype=np.float32)  # Allocate buffer in Python

        self.lib.vecx_calculate_distances(
            self.vecx,
            ffi.cast("float*", query_array.ctypes.data),
            self.dimension,
            ffi.cast("float*", vectors_array.ctypes.data),
            num_vectors,
            ffi.cast("float*", out_distances.ctypes.data)  # Pass preallocated buffer
        )

        return out_distances  # No need to free, Python handles it

    def encrypt_meta(self, meta):
        """Encrypt metadata (binary data)."""
        meta_bytes = bytes(meta)  # Ensure meta is in bytes
        out_length = ffi.new("int *")  # Allocate space for output length

        # Use ffi.from_buffer to create a buffer from bytes
        result = self.lib.vecx_encrypt_meta(self.vecx, ffi.from_buffer("unsigned char[]", meta_bytes), len(meta_bytes), out_length)

        if result == ffi.NULL:
            raise RuntimeError("Encryption failed")

        encrypted_data = ffi.buffer(result, out_length[0])[:]  # Convert C buffer to Python bytes
        self.lib.vecx_free_buffer(result)  # Free memory allocated by C++
        
        return encrypted_data

    def decrypt_meta(self, encrypted_meta: bytes) -> bytes:
        """Decrypt metadata (binary data)."""
        out_length = ffi.new("int*")  # Allocate space for output length

        # Use ffi.from_buffer to safely pass bytes to C
        result = self.lib.vecx_decrypt_meta(self.vecx, ffi.from_buffer("unsigned char[]", encrypted_meta), len(encrypted_meta), out_length)

        if result == ffi.NULL:
            raise RuntimeError("Decryption failed.")

        data = ffi.buffer(result, out_length[0])[:]  # Convert C buffer to Python bytes
        self.lib.vecx_free_buffer(result)  # Free allocated memory in C++
        
        return data
    def encrypt_vectors(self, matrix):
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        if matrix.shape[1] != self.dimension:
            raise ValueError(f"Expected shape (*, {self.dimension}), got {matrix.shape}")
        num_vectors = matrix.shape[0]
        output = np.empty_like(matrix)
        self.lib.vecx_encrypt_vectors(
            self.vecx,
            ffi.cast("float*", matrix.ctypes.data),
            ffi.cast("float*", output.ctypes.data),
            self.dimension,
            num_vectors
        )
        return output

    def decrypt_vectors(self, matrix):
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        if matrix.shape[1] != self.dimension:
            raise ValueError(f"Expected shape (*, {self.dimension}), got {matrix.shape}")
        num_vectors = matrix.shape[0]
        output = np.empty_like(matrix)
        self.lib.vecx_decrypt_vectors(
            self.vecx,
            ffi.cast("float*", matrix.ctypes.data),
            ffi.cast("float*", output.ctypes.data),
            self.dimension,
            num_vectors
        )
        return output

    def decrypt_and_calculate_similarities(self, query_vector, encrypted_matrix):
        query = np.asarray(query_vector, dtype=np.float32)
        matrix = np.ascontiguousarray(encrypted_matrix, dtype=np.float32)
        if query.shape[0] != self.dimension:
            raise ValueError("Query vector has incorrect dimension.")
        if matrix.shape[1] != self.dimension:
            raise ValueError("Matrix vectors must match dimension.")
        num_vectors = matrix.shape[0]
        similarities = np.empty(num_vectors, dtype=np.float32)
        self.lib.vecx_decrypt_and_calculate_similarities(
            self.vecx,
            ffi.cast("float*", query.ctypes.data),
            ffi.cast("float*", matrix.ctypes.data),
            self.dimension,
            num_vectors,
            ffi.cast("float*", similarities.ctypes.data)
        )
        return similarities


@lru_cache(maxsize=None)
def get_libvx(key:str, lib_token:str, dimension:int, space_type:str="cosine", version:int=1):
    return LibVectorX(key, lib_token, dimension, space_type, version)
