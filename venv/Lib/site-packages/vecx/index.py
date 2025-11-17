import requests, json, zlib
import numpy as np
import msgpack
from .libvx import get_libvx
from .crypto import get_checksum, json_zip, json_unzip
from .exceptions import raise_exception

class Index:
    def __init__(self, name:str, key:str, token:str, url:str, version:int=1, params=None):
        self.name = name
        self.key = key
        self.token = token
        self.url = url
        self.version = version
        self.checksum = get_checksum(self.key)
        self.lib_token = params["lib_token"]
        self.count = params["total_elements"]
        self.space_type = params["space_type"]
        self.dimension = params["dimension"]
        self.precision = "float16" if params["use_fp16"] else "float32"
        self.M = params["M"]

        if key:
            self.vxlib = get_libvx(key=key, lib_token=self.lib_token, dimension=self.dimension, space_type=self.space_type, version=version)
        else:
            self.vxlib = None

    def __str__(self):
        return self.name
    
    def _normalize_vector(self, vector):
        # Convert to numpy array if not already
        vector = np.array(vector, dtype=np.float32)
        # Check dimension of the vector
        if vector.ndim != 1 or vector.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vector.shape[0]}")
        # Normalize only if using cosine distance
        if self.space_type != "cosine":
            return vector, 1.0
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector, 1.0
        normalized_vector = vector / norm
        return normalized_vector, float(norm)

    def upsert(self, input_array):
        if len(input_array) > 1000:
            raise ValueError("Cannot insert more than 1000 vectors at a time")
        
        vector_batch = []

        vectors = []
        norms = []
        for item in input_array:
            vector, norm = self._normalize_vector(item['vector'])
            vectors.append(vector)
            norms.append(norm)

        vectors = np.vstack(vectors).astype(np.float32)  # shape: (N, dim)

        if self.vxlib:
            encrypted = self.vxlib.encrypt_vectors(vectors)
        else:
            encrypted = vectors

        for i, item in enumerate(input_array):
            meta_data = json_zip(dict=item.get('meta', {}))
            if self.vxlib:
                meta_data = self.vxlib.encrypt_meta(meta_data)

            vector_obj = [
                str(item.get('id', '')),
                meta_data,
                json.dumps(item.get('filter', {})),
                float(norms[i]),
                encrypted[i].tolist()
            ]
            vector_batch.append(vector_obj)

        serialized_data = msgpack.packb(vector_batch, use_bin_type=True, use_single_float=True)
        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/msgpack'
        }
        
        response = requests.post(
            f'{self.url}/index/{self.name}/vector/insert', 
            headers=headers, 
            data=serialized_data
        )

        if response.status_code != 200:
            raise_exception(response.status_code, response.text)
        return "Vectors inserted successfully"

    def query(self, vector, top_k=10, filter=None, ef=128, include_vectors=False, log=False):
        if top_k > 512:
            raise ValueError("top_k cannot be greater than 512")
        if ef > 1024:
            raise ValueError("ef search cannot be greater than 1024")

        # Normalize query vector if using cosine
        vector, norm = self._normalize_vector(vector)
        original_vector = vector

        # Encrypt query vector
        if self.vxlib:
            vector = self.vxlib.encrypt_vector(vector)

        # Prepare search request
        headers = {
            'Authorization': f'{self.token}',
            'Content-Type': 'application/json'
        }

        data = {
            'vector': vector.tolist(),
            'k': top_k,
            'ef': ef,
            'include_vectors': include_vectors
        }

        if filter:
            data['filter'] = json.dumps(filter)

        response = requests.post(
            f'{self.url}/index/{self.name}/search',
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            raise_exception(response.status_code, response.text)

        results = msgpack.unpackb(response.content, raw=False)

        # [similarity, id, meta, filter, norm, vector]
        vectors = []
        processed_results = []

        for result in results:
            similarity = result[0]
            vector_id = result[1]
            meta_data = result[2]
            filter_str = result[3]
            norm_value = result[4]
            vector_data = result[5] if len(result) > 5 else []

            processed = {
                'id': vector_id,
                'similarity': similarity,
                'distance': 1.0 - similarity,
                'meta': json_unzip(self.vxlib.decrypt_meta(meta_data)) if self.vxlib else json_unzip(meta_data),
                'norm': norm_value
            }

            if filter_str:
                processed['filter'] = json.loads(filter_str)

            if (include_vectors or self.vxlib) and vector_data:
                if self.vxlib:
                    vectors.append(np.array(vector_data, dtype=np.float32))
                else:
                    processed['vector'] = list(vector_data)

            processed_results.append(processed)

        # Rescore using batch similarity
        if self.vxlib and vectors:
            decrypted_matrix = np.vstack(vectors).astype(np.float32)
            similarities = self.vxlib.decrypt_and_calculate_similarities(original_vector, decrypted_matrix)

            for i, (result, sim) in enumerate(zip(processed_results, similarities)):
                result['distance'] = 1.0 - float(sim)
                result['similarity'] = float(sim)
                if include_vectors:
                    result['vector'] = self.vxlib.decrypt_vector(decrypted_matrix[i])

            processed_results = sorted(processed_results, key=lambda x: x['distance'])[:top_k]

        if not include_vectors:
            for result in processed_results:
                result.pop('vector', None)

        return processed_results

    
    def delete_vector(self, id):
        checksum = get_checksum(self.key)
        headers = {
            'Authorization': f'{self.token}',
        }
        response = requests.delete(f'{self.url}/index/{self.name}/vector/{id}/delete', headers=headers)
        if response.status_code != 200:
            raise_exception(response.status_code, response.text)
        return response.text + " rows deleted"
    
    # Delete multiple vectors based on a filter
    def delete_with_filter(self, filter):
        checksum = get_checksum(self.key)
        headers = {
            'Authorization': f'{self.token}',
            'Content-Type': 'application/json'
        }
        data = {"filter": filter}
        print(filter)
        response = requests.delete(f'{self.url}/index/{self.name}/vectors/delete', headers=headers, json=data)
        if response.status_code != 200:
            print(response.text)
            raise_exception(response.status_code, response.text)
        return response.text
    
    # Get a single vector by id
    def get_vector(self, id):
        checksum = get_checksum(self.key)
        headers = {
            'Authorization': f'{self.token}',
            'Content-Type': 'application/json'
        }
        
        # Use POST method with the ID in the request body
        response = requests.post(
            f'{self.url}/index/{self.name}/vector/get',
            headers=headers,
            json={'id': id}
        )
        
        if response.status_code != 200:
            raise_exception(response.status_code, response.text)
        
        # Parse the msgpack response
        vector_obj = msgpack.unpackb(response.content, raw=False)
        
        response = {
            'id': vector_obj[0],
            'meta': json_unzip(self.vxlib.decrypt_meta(vector_obj[1])) if self.vxlib else json_unzip(vector_obj[1]),
            'filter': vector_obj[2],
            'norm': vector_obj[3],
            'vector': list(self.vxlib.decrypt_vector(vector_obj[4])) if self.vxlib else list(vector_obj[4])
        }
        
        return response

    def describe(self):
        data = {
            "name": self.name,
            "space_type": self.space_type,
            "dimension": self.dimension,
            "count": self.count,
            "precision": self.precision,
            "M": self.M,
        }
        return data