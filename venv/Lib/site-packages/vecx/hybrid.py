import requests
import json
import numpy as np
import msgpack
from .crypto import get_checksum, json_zip, json_unzip
from .exceptions import raise_exception
from typing import List, Dict, Any, Optional, Union

class HybridIndex:
    """
    Hybrid Index class for managing dense + sparse vector operations.
    
    This class handles hybrid vector indexes that support both dense vectors (traditional embeddings)
    and sparse vectors (keyword/token-based) with Reciprocal Rank Fusion (RRF) for combined search.
    
    Note: Encryption is disabled for hybrid indexes.
    """
    
    def __init__(self, name: str, key: str, token: str, url: str, version: int = 1, params: Dict = None):
        self.name = name
        self.key = key
        self.token = token
        self.url = url
        self.version = version
        self.checksum = get_checksum(self.key)
        
        # Hybrid-specific parameters
        self.lib_token = params.get("lib_token") if params else None
        self.count = params.get("total_elements", 0) if params else 0
        self.space_type = params.get("space_type", "cosine") if params else "cosine"
        self.dimension = params.get("dimension", 128) if params else 128
        self.vocab_size = params.get("vocab_size", 30522) if params else 30522
        self.precision = "float16" if params and params.get("use_fp16") else "float32"
        self.M = params.get("M", 16) if params else 16
        
        # Encryption is disabled for hybrid indexes
        self.vxlib = None

    def __str__(self):
        return f"HybridIndex({self.name})"
    
    def _normalize_dense_vector(self, vector: List[float]) -> tuple:
        """Normalize dense vector for cosine similarity if needed."""
        vector = np.array(vector, dtype=np.float32)
        
        # Check dimension
        if vector.ndim != 1 or vector.shape[0] != self.dimension:
            raise ValueError(f"Dense vector dimension mismatch: expected {self.dimension}, got {vector.shape[0]}")
        
        # Normalize only if using cosine distance
        if self.space_type != "cosine":
            return vector, 1.0
            
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector, 1.0
            
        normalized_vector = vector / norm
        return normalized_vector, float(norm)
    
    def _normalize_sparse_vector(self, indices: List[int], values: List[float]) -> tuple:
        """Normalize sparse vector values."""
        if len(indices) != len(values):
            raise ValueError("Sparse vector indices and values must have same length")
        
        # Check vocab size bounds
        for idx in indices:
            if idx < 0 or idx >= self.vocab_size:
                raise ValueError(f"Sparse vector index {idx} out of vocab bounds [0, {self.vocab_size})")
        
        values = np.array(values, dtype=np.float32)
        norm = np.linalg.norm(values)
        
        if norm == 0:
            return indices, values.tolist(), 1.0
            
        if self.space_type == "cosine":
            normalized_values = values / norm
            return indices, normalized_values.tolist(), float(norm)
        else:
            return indices, values.tolist(), float(norm)

    def upsert(self, input_array: List[Dict[str, Any]]) -> str:
        """
        Insert or update hybrid vectors (dense + sparse).
        
        Args:
            input_array: List of hybrid vector objects with structure:
                {
                    "id": str,
                    "dense_vector": List[float],
                    "sparse_indices": List[int],
                    "sparse_values": List[float], 
                    "meta": Dict (optional),
                    "filter": Dict (optional)
                }
        
        Returns:
            Success message string
        """
        if len(input_array) > 1000:
            raise ValueError("Cannot insert more than 1000 vectors at a time")
        
        vector_batch = []
        
        # Process each vector
        for i, item in enumerate(input_array):
            if "dense_vector" not in item:
                raise ValueError("Missing 'dense_vector' in input item")
            
            # Normalize dense vector
            dense_vector, dense_norm = self._normalize_dense_vector(item["dense_vector"])
            
            # Handle sparse vector
            sparse_indices = item.get("sparse_indices", [])
            sparse_values = item.get("sparse_values", [])
            
            if len(sparse_indices) != len(sparse_values):
                raise ValueError(f"Sparse indices and values length mismatch in item {i}")
            
            # Normalize sparse vector
            norm_indices, norm_values, sparse_norm = self._normalize_sparse_vector(
                sparse_indices, sparse_values
            )
            
            # Prepare metadata (no encryption - send as JSON string)
            meta_dict = item.get("meta", {})
            meta_data = json.dumps(meta_dict) if meta_dict else ""
            
            # Create hybrid vector object in expected format
            vector_obj = {
                "id": str(item.get("id", "")),
                "meta": meta_data,
                "filter": json.dumps(item.get("filter", {})),
                "dense_vector": dense_vector.tolist(),
                "dense_norm": float(dense_norm),
                "sparse_norm": float(sparse_norm),
                "indices": norm_indices,
                "values": norm_values
            }
            
            vector_batch.append(vector_obj)
        
        # Send to server
        serialized_data = msgpack.packb(vector_batch, use_bin_type=True, use_single_float=True)
        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/msgpack'
        }
        
        response = requests.post(
            f'{self.url}/hybrid/unified/{self.name}/add',
            headers=headers,
            data=serialized_data
        )
        
        if response.status_code != 200:
            raise_exception(response.status_code, response.text)
        
        return "Hybrid vectors inserted successfully"

    def query(self, 
              dense_vector: List[float],
              sparse_indices: List[int], 
              sparse_values: List[float],
              sparse_top_k: int = 50,
              dense_top_k: int = 50,
              final_top_k: int = 10,
              k_rrf: float = 60.0,
              filter: Optional[Dict] = None,
              include_vectors: bool = False) -> List[Dict[str, Any]]:
        """
        Search hybrid vectors using both dense and sparse components.
        
        Note: Both dense_vector AND sparse components (indices + values) are required.
        Hybrid search combines both modalities for optimal RRF ranking results.
        
        Args:
            dense_vector: Dense query vector (required)
            sparse_indices: Sparse vector indices (required)
            sparse_values: Sparse vector values (required)
            sparse_top_k: Top-K for sparse search
            dense_top_k: Top-K for dense search
            final_top_k: Final top-K after RRF
            k_rrf: RRF parameter (higher = more weight to rank position)
            filter: Optional filter dictionary
            include_vectors: Whether to include vector data in results
            
        Returns:
            List of search results with RRF scores and rankings
            
        Raises:
            ValueError: If dense_vector or sparse components are missing
        """
        if final_top_k > 256:
            raise ValueError("final_top_k cannot be greater than 256")
        
        # Validate inputs - require both dense and sparse components for hybrid search
        if dense_vector is None:
            raise ValueError(
                "Missing dense_vector: Hybrid search requires both dense and sparse components. "
                "Please provide dense_vector parameter."
            )
        
        if sparse_indices is None or sparse_values is None:
            raise ValueError(
                "Missing sparse components: Hybrid search requires both dense and sparse components. "
                "Please provide both sparse_indices and sparse_values parameters."
            )
        
        if not sparse_indices or not sparse_values:
            raise ValueError(
                "Empty sparse components: sparse_indices and sparse_values cannot be empty. "
                "Please provide non-empty sparse_indices and sparse_values lists."
            )
        
        # Prepare query data
        query_data = {
            "sparse_top_k": sparse_top_k,
            "dense_top_k": dense_top_k, 
            "final_top_k": final_top_k,
            "k_rrf": k_rrf,
            "include_vectors": include_vectors
        }
        
        # Handle dense vector (no encryption)
        dense_vec, dense_norm = self._normalize_dense_vector(dense_vector)
        query_data["dense_vector"] = dense_vec.tolist()
        
        # Handle sparse vector
        norm_indices, norm_values, sparse_norm = self._normalize_sparse_vector(
            sparse_indices, sparse_values
        )
        
        # Convert to server expected format
        sparse_vector = [{"index": idx, "value": val} 
                       for idx, val in zip(norm_indices, norm_values)]
        query_data["sparse_vector"] = sparse_vector
        
        # Add filter if provided
        if filter:
            query_data["filter"] = json.dumps(filter)
        
        # Make request
        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f'{self.url}/hybrid/unified/{self.name}/search',
            headers=headers,
            json=query_data
        )
        
        if response.status_code != 200:
            raise_exception(response.status_code, response.text)
        
        # Parse results
        results = response.json()
        processed_results = []
        
        for result in results.get("results", []):
            processed = {
                "id": result["id"],
                "rrf_score": result["rrf_score"],
                "dense_rank": result.get("dense_rank"),
                "sparse_rank": result.get("sparse_rank"),
                "dense_score": result.get("dense_score"),
                "sparse_score": result.get("sparse_score")
            }
            
            # Add metadata if present (no decryption)
            if "meta" in result:
                meta_data = result["meta"]
                # Handle different metadata formats when encryption is disabled
                if isinstance(meta_data, str):
                    # If it's a string, try to parse as JSON directly
                    try:
                        processed["meta"] = json.loads(meta_data) if meta_data else {}
                    except json.JSONDecodeError:
                        processed["meta"] = {}
                elif isinstance(meta_data, bytes):
                    # If it's bytes, use json_unzip (compressed format)
                    processed["meta"] = json_unzip(meta_data)
                else:
                    # If it's already a dict or other format, use as-is
                    processed["meta"] = meta_data if meta_data else {}
            
            # Add filter if present
            if "filter" in result and result["filter"]:
                processed["filter"] = json.loads(result["filter"])
            
            # Handle vectors if requested (no decryption)
            if include_vectors and "dense_vector" in result:
                processed["dense_vector"] = result["dense_vector"]
                
                if "sparse_indices" in result:
                    processed["sparse_indices"] = result["sparse_indices"]
                    processed["sparse_values"] = result["sparse_values"]
            
            processed_results.append(processed)
        
        return processed_results

    def get_vector(self, vector_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific hybrid vector by ID.
        
        Args:
            vector_id: The ID of the vector to retrieve
            
        Returns:
            Dictionary containing the vector data
        """
        headers = {
            'Authorization': self.token
        }
        
        response = requests.get(
            f'{self.url}/hybrid/unified/{self.name}/vector/{vector_id}',
            headers=headers
        )
        
        if response.status_code != 200:
            raise_exception(response.status_code, response.text)
        
        vector_data = response.json()
        
        # Process the response (no decryption)
        processed = {
            "id": vector_data["id"],
            "dense_norm": vector_data.get("dense_norm"),
            "sparse_norm": vector_data.get("sparse_norm")
        }
        
        # Add metadata (no decryption)
        if "meta" in vector_data:
            meta_data = vector_data["meta"]
            # Handle different metadata formats when encryption is disabled
            if isinstance(meta_data, str):
                # If it's a string, try to parse as JSON directly
                try:
                    processed["meta"] = json.loads(meta_data) if meta_data else {}
                except json.JSONDecodeError:
                    processed["meta"] = {}
            elif isinstance(meta_data, bytes):
                # If it's bytes, use json_unzip (compressed format)
                processed["meta"] = json_unzip(meta_data)
            else:
                # If it's already a dict or other format, use as-is
                processed["meta"] = meta_data if meta_data else {}
        
        # Add dense vector (no decryption)
        if "dense_vector" in vector_data:
            processed["dense_vector"] = vector_data["dense_vector"]
        
        # Add sparse components
        if "sparse_indices" in vector_data:
            processed["sparse_indices"] = vector_data["sparse_indices"]
            processed["sparse_values"] = vector_data["sparse_values"]
        
        # Add filter if present
        if "filter" in vector_data and vector_data["filter"]:
            processed["filter"] = json.loads(vector_data["filter"])
        
        return processed

    def delete_vector(self, vector_id: str) -> str:
        """
        Delete a specific hybrid vector by ID.
        
        Args:
            vector_id: The ID of the vector to delete
            
        Returns:
            Success message
        """
        headers = {
            'Authorization': self.token
        }
        
        response = requests.delete(
            f'{self.url}/hybrid/unified/{self.name}/vector/{vector_id}',
            headers=headers
        )
        
        if response.status_code != 200:
            raise_exception(response.status_code, response.text)
        
        return f"Vector {vector_id} deleted successfully"

    def describe(self) -> Dict[str, Any]:
        """
        Get hybrid index information and statistics.
        
        Returns:
            Dictionary with index metadata
        """
        return {
            "name": self.name,
            "type": "hybrid",
            "space_type": self.space_type,
            "dimension": self.dimension,
            "vocab_size": self.vocab_size,
            "count": self.count,
            "precision": self.precision,
            "M": self.M,
            "encryption": False  # Hybrid indexes have encryption disabled
        } 