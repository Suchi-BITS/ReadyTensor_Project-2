import requests
import json
import base64
import numpy as np
import msgpack
from .libvx import get_libvx
from .crypto import get_checksum, json_zip, json_unzip
from .exceptions import raise_exception
from .utils import reciprocal_rank_fusion, validate_rrf_input


class HybridIndex:
    def __init__(self, name: str, key: str, token: str, url: str, version: int = 1, params=None):
        self.name = name
        self.key = key
        self.token = token
        self.url = url
        self.version = version
        self.checksum = get_checksum(self.key)
        self.lib_token = params["lib_token"]
        self.count = params.get("total_elements", 0)
        self.space_type = params["space_type"]
        self.dimension = params["dimension"]
        self.vocab_size = params.get("vocab_size", 30522)
        self.precision = "float16" if params.get("use_fp16", False) else "float32"
        self.M = params.get("M", 16)

        if key:
            # Use vocab_size as sparse_max_size for sparse encryption
            self.vxlib = get_libvx(key=key, lib_token=self.lib_token, space_type=self.space_type, 
                                   version=version, dimension=self.dimension, sparse_max_size=self.vocab_size)
        else:
            self.vxlib = None

    def __str__(self):
        return self.name

    def _normalize_vector(self, vector):
        """Normalize dense vector if using cosine distance"""
        vector = np.array(vector, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vector.shape[0]}")
        
        if self.space_type != "cosine":
            return vector, 1.0
        
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector, 1.0
        
        normalized_vector = vector / norm
        return normalized_vector, float(norm)

    def _calculate_sparse_norm(self, indices, values):
        """Calculate L2 norm of sparse vector"""
        if not indices or not values:
            return 0.0
        return float(np.linalg.norm(values))

    def _encrypt_sparse_vector(self, indices, values):
        """Encrypt sparse vector indices and values using dedicated sparse encryption"""
        if not self.vxlib:
            return indices, values
        
        # Use the new sparse encryption functions
        encrypted_indices, encrypted_values = self.vxlib.encrypt_sparse_vector(indices, values)
        
        return encrypted_indices, encrypted_values

    def _decrypt_sparse_vector(self, encrypted_indices, encrypted_values):
        """Decrypt sparse vector using dedicated sparse decryption"""
        if not self.vxlib:
            # For unencrypted mode, the data is already in the right format
            return encrypted_indices, encrypted_values
        
        # Use the new sparse decryption functions
        indices, values = self.vxlib.decrypt_sparse_vector(encrypted_indices, encrypted_values)
        
        return indices, values

    def upsert(self, input_array):
        """
        Insert hybrid vectors into the index.
        
        Args:
            input_array: List of hybrid vector objects containing:
                - id: unique identifier
                - dense_vector: array of floats
                - sparse_vector: dict with 'indices' and 'values' arrays
                - meta: metadata dictionary (optional)
        """
        if len(input_array) > 1000:
            raise ValueError("Cannot insert more than 1000 vectors at a time")
        
        vector_batch = []
        
        # Process each vector
        for item in input_array:
            vector_id = str(item.get('id', ''))
            dense_vector = item.get('dense_vector', [])
            sparse_vector = item.get('sparse_vector', {})
            meta_data = item.get('meta', {})
            
            # Normalize dense vector
            normalized_dense, dense_norm = self._normalize_vector(dense_vector)
            
            # Extract sparse vector components
            indices = sparse_vector.get('indices', [])
            values = sparse_vector.get('values', [])
            
            # Calculate sparse norm
            sparse_norm = self._calculate_sparse_norm(indices, values)
            
            # Encrypt dense vector if needed
            if self.vxlib:
                encrypted_dense = self.vxlib.encrypt_vector(normalized_dense)
                encrypted_indices, encrypted_values = self._encrypt_sparse_vector(indices, values)
                encrypted_meta = self.vxlib.encrypt_meta(json_zip(meta_data))
            else:
                encrypted_dense = normalized_dense
                encrypted_indices, encrypted_values = indices, values
                encrypted_meta = json_zip(meta_data)
            
            # Encode metadata to base64
            meta_b64 = base64.b64encode(encrypted_meta).decode('utf-8')
            
            # Create hybrid vector object
            hybrid_vector = {
                "id": vector_id,
                "dense_vector": encrypted_dense.tolist(),
                "indices": encrypted_indices,
                "values": encrypted_values,
                "dense_norm": dense_norm,
                "sparse_norm": sparse_norm,
                "meta": meta_b64
            }
            
            vector_batch.append(hybrid_vector)
        
        # Serialize and send request
        serialized_data = msgpack.packb(vector_batch, use_bin_type=True, use_single_float=True)
        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/msgpack'
        }
        
        response = requests.post(
            f'{self.url}/hybrid/{self.name}/add',
            headers=headers,
            data=serialized_data
        )
        
        if response.status_code not in [200, 201]:
            raise_exception(response.status_code, response.text)
        
        return "Hybrid vectors inserted successfully"

    def search(self, dense_vector, sparse_vector, sparse_top_k=50, dense_top_k=50, 
               include_vectors=False, filter_query="", rrf_k=60):
        """
        Search the hybrid index and return fused results.
        
        Args:
            dense_vector: query dense vector
            sparse_vector: dict with 'indices' and 'values' arrays
            sparse_top_k: number of sparse results to retrieve
            dense_top_k: number of dense results to retrieve
            include_vectors: whether to include vectors in response
            filter_query: filter query string
            rrf_k: RRF parameter for fusion
        
        Returns:
            List of fused results sorted by RRF score
        """
        if sparse_top_k > 256:
            raise ValueError("sparse_top_k cannot be greater than 256")
        if dense_top_k > 256:
            raise ValueError("dense_top_k cannot be greater than 256")
        
        # Normalize dense query vector
        normalized_dense, _ = self._normalize_vector(dense_vector)
        
        # Encrypt query vectors if needed
        if self.vxlib:
            encrypted_dense = self.vxlib.encrypt_vector(normalized_dense)
        else:
            encrypted_dense = normalized_dense
        
        # Format sparse vector for API
        sparse_indices = sparse_vector.get('indices', [])
        sparse_values = sparse_vector.get('values', [])
        
        # Encrypt sparse query if needed (just like dense query)
        if self.vxlib and sparse_indices and sparse_values:
            encrypted_query_indices, encrypted_query_values = self._encrypt_sparse_vector(sparse_indices, sparse_values)
            sparse_query = [{"index": idx, "value": val} for idx, val in zip(encrypted_query_indices, encrypted_query_values)]
        else:
            sparse_query = [{"index": idx, "value": val} for idx, val in zip(sparse_indices, sparse_values)]
        
        # Prepare search request
        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/json'
        }
        
        data = {
            'dense_vector': encrypted_dense.tolist(),
            'sparse_vector': sparse_query,
            'sparse_top_k': sparse_top_k,
            'dense_top_k': dense_top_k,
            'include_vectors': include_vectors,
            'filter': filter_query
        }
        
        response = requests.post(
            f'{self.url}/hybrid/{self.name}/search_separate',
            headers=headers,
            json=data
        )
        
        if response.status_code not in [200, 201]:
            raise_exception(response.status_code, response.text)
        
        results = response.json()
        
        # Process results for decryption and fusion
        processed_results = self._process_search_results(results, include_vectors)
        
        # Validate RRF input
        is_valid, error_msg = validate_rrf_input(processed_results)
        if not is_valid:
            raise ValueError(f"RRF validation failed: {error_msg}")
        
        # Apply RRF fusion
        fused_results = reciprocal_rank_fusion(processed_results, k=rrf_k)
        
        # Clean up vector data if not requested
        if not include_vectors:
            for result in fused_results:
                result.pop('vector', None)
        
        return fused_results

    def _process_search_results(self, results, include_vectors=False):
        """Process and decrypt search results from the API"""
        dense_results = results.get('dense_results', [])
        sparse_results = results.get('sparse_results', [])
        metadata = results.get('metadata', [])
        
        # Process dense results
        processed_dense = []
        for result in dense_results:
            processed = {
                'id': result['id'],
                'score': result['score'],
                'rank': result['rank'],
                'vector': None
            }
            
            # Decrypt dense vector if present
            if 'vector' in result and result['vector']:
                if self.vxlib:
                    decrypted_vector = self.vxlib.decrypt_vector(np.array(result['vector'], dtype=np.float32))
                    processed['vector'] = decrypted_vector.tolist()
                else:
                    processed['vector'] = result['vector']
            
            processed_dense.append(processed)
        
        # Process sparse results
        processed_sparse = []
        for result in sparse_results:
            processed = {
                'id': result['id'],
                'score': result['score'],
                'rank': result['rank'],
                'vector': None
            }
            
            # Decrypt sparse vector if present
            if 'vector' in result and result['vector']:
                if self.vxlib:
                    # For encrypted mode, decrypt sparse vectors
                    if isinstance(result['vector'], list) and len(result['vector']) > 0:
                        if isinstance(result['vector'][0], dict) and 'index' in result['vector'][0]:
                            # Standard sparse vector format - extract indices and values
                            indices = [item['index'] for item in result['vector']]
                            values = [item['value'] for item in result['vector']]
                            
                            # Decrypt the sparse vector
                            decrypted_indices, decrypted_values = self._decrypt_sparse_vector(indices, values)
                            
                            # Reconstruct sparse vector format
                            processed['vector'] = [
                                {"index": idx, "value": val} 
                                for idx, val in zip(decrypted_indices, decrypted_values)
                            ]
                        else:
                            # Other format - keep as is
                            processed['vector'] = result['vector']
                    else:
                        processed['vector'] = result['vector']
                else:
                    processed['vector'] = result['vector']
            
            processed_sparse.append(processed)
        
        # Process metadata
        processed_metadata = []
        for meta in metadata:
            processed = {'id': meta['id'], 'meta': ''}
            
            # Decode and decrypt metadata
            if meta.get('meta'):
                try:
                    decoded_meta = base64.b64decode(meta['meta'])
                    if self.vxlib:
                        decrypted_meta = self.vxlib.decrypt_meta(decoded_meta)
                        processed['meta'] = json_unzip(decrypted_meta)
                    else:
                        processed['meta'] = json_unzip(decoded_meta)
                except Exception as e:
                    print(f"Warning: Failed to decrypt metadata for {meta['id']}: {e}")
                    processed['meta'] = {}
            
            processed_metadata.append(processed)
        
        return {
            'dense_results': processed_dense,
            'sparse_results': processed_sparse,
            'metadata': processed_metadata
        }

    def get_vector(self, vector_id):
        """Get a hybrid vector by ID"""
        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/json'
        }
        
        response = requests.get(
            f'{self.url}/hybrid/{self.name}/vector/{vector_id}',
            headers=headers
        )
        
        if response.status_code not in [200, 201]:
            raise_exception(response.status_code, response.text)
        
        result = response.json()
        
        # Decrypt the vector data
        if self.vxlib:
            # Decrypt dense vector
            if result.get('dense_vector'):
                decrypted_dense = self.vxlib.decrypt_vector(np.array(result['dense_vector'], dtype=np.float32))
                result['dense_vector'] = decrypted_dense.tolist()
            
            # Decrypt sparse vector
            if result.get('sparse_vector'):
                if isinstance(result['sparse_vector'], list):
                    # Convert from [{"index": idx, "value": val}] format
                    if result['sparse_vector'] and isinstance(result['sparse_vector'][0], dict):
                        indices = [item['index'] for item in result['sparse_vector']]
                        values = [item['value'] for item in result['sparse_vector']]
                        
                        # Decrypt the sparse vector
                        decrypted_indices, decrypted_values = self._decrypt_sparse_vector(indices, values)
                        result['sparse_vector'] = {
                            "indices": decrypted_indices,
                            "values": decrypted_values
                        }
                elif isinstance(result['sparse_vector'], dict):
                    # Handle case where sparse vector is already in {"indices": [...], "values": [...]} format
                    if 'indices' in result['sparse_vector'] and 'values' in result['sparse_vector']:
                        indices = result['sparse_vector']['indices']
                        values = result['sparse_vector']['values']
                        
                        # Decrypt the sparse vector
                        decrypted_indices, decrypted_values = self._decrypt_sparse_vector(indices, values)
                        result['sparse_vector'] = {
                            "indices": decrypted_indices,
                            "values": decrypted_values
                        }
            
            # Decrypt metadata
            if result.get('meta'):
                try:
                    decoded_meta = base64.b64decode(result['meta'])
                    decrypted_meta = self.vxlib.decrypt_meta(decoded_meta)
                    result['meta'] = json_unzip(decrypted_meta)
                except Exception as e:
                    print(f"Warning: Failed to decrypt metadata: {e}")
                    result['meta'] = {}
        else:
            # Decode metadata for unencrypted mode
            if result.get('meta'):
                try:
                    decoded_meta = base64.b64decode(result['meta'])
                    result['meta'] = json_unzip(decoded_meta)
                except Exception as e:
                    print(f"Warning: Failed to decode metadata: {e}")
                    result['meta'] = {}
            
            # Format sparse vector for consistency
            if result.get('sparse_vector'):
                if isinstance(result['sparse_vector'], list):
                    # Convert from [{"index": idx, "value": val}] to {"indices": [...], "values": [...]}
                    if result['sparse_vector'] and isinstance(result['sparse_vector'][0], dict):
                        indices = [item['index'] for item in result['sparse_vector']]
                        values = [item['value'] for item in result['sparse_vector']]
                        result['sparse_vector'] = {"indices": indices, "values": values}
        
        return result

    def delete_vector(self, vector_id):
        """Delete a hybrid vector by ID"""
        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/json'
        }
        
        response = requests.delete(
            f'{self.url}/hybrid/{self.name}/vector/{vector_id}',
            headers=headers
        )
        
        if response.status_code not in [200, 201]:
            raise_exception(response.status_code, response.text)
        
        return f"Hybrid vector {vector_id} deleted successfully" 