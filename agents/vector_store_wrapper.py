import os
from typing import List, Dict, Any, Optional
from openai import OpenAI


class VectorStoreWrapper:
    """
    Wrapper class for OpenAI Vector Stores functionality.
    
    Provides methods to create, manage, and search vector stores.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the VectorStoreWrapper.
        
        Args:
            api_key (Optional[str]): OpenAI API key. If not provided, will use OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self.client = OpenAI(api_key=self.api_key)
    
    def create_vector_store(self, name: str, file_ids: Optional[List[str]] = None) -> str:
        """
        Create a new vector store.
        
        Args:
            name (str): Name for the vector store
            file_ids (Optional[List[str]]): List of file IDs to add to the vector store
            
        Returns:
            str: Vector store ID
        """
        vector_store_data = {"name": name}
        if file_ids:
            vector_store_data["file_ids"] = file_ids
            
        vector_store = self.client.vector_stores.create(**vector_store_data)
        return vector_store.id
    
    def upload_file(self, file_path: str, purpose: str = "assistants") -> str:
        """
        Upload a file to OpenAI for use in vector stores.
        
        Args:
            file_path (str): Path to the file to upload
            purpose (str): Purpose of the file (default: "assistants")
            
        Returns:
            str: File ID
        """
        with open(file_path, "rb") as file:
            uploaded_file = self.client.files.create(file=file, purpose=purpose)
        return uploaded_file.id
    
    def add_files_to_vector_store(self, vector_store_id: str, file_ids: List[str]) -> None:
        """
        Add files to an existing vector store.
        
        Args:
            vector_store_id (str): ID of the vector store
            file_ids (List[str]): List of file IDs to add
        """
        for file_id in file_ids:
            self.client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=file_id
            )
    
    def search_vector_store(
        self, 
        vector_store_id: str, 
        query: str, 
        limit: int = 20,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search a vector store for relevant content.
        
        Args:
            vector_store_id (str): ID of the vector store to search
            query (str): Search query
            limit (int): Maximum number of results to return (default: 20)
            filter_metadata (Optional[Dict[str, Any]]): Metadata filters to apply
            
        Returns:
            List[Dict[str, Any]]: List of search results with content and metadata
        """
        search_params = {
            "query": query,
            "max_num_results": limit
        }
        
            
        results = self.client.vector_stores.search(
            vector_store_id=vector_store_id,
            **search_params
        )
        
        return [{"content": result.content } for result in results.data]
    
    
    def list_vector_stores(self) -> List[Dict[str, Any]]:
        """
        List all vector stores.
        
        Returns:
            List[Dict[str, Any]]: List of vector stores with id, name, and metadata
        """
        vector_stores = self.client.vector_stores.list()
        return [
            {
                "id": vs.id, 
                "name": vs.name, 
                "created_at": vs.created_at,
                "file_counts": vs.file_counts
            } 
            for vs in vector_stores.data
        ]
    
    def delete_vector_store(self, vector_store_id: str) -> bool:
        """
        Delete a vector store.
        
        Args:
            vector_store_id (str): ID of the vector store to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.client.vector_stores.delete(vector_store_id)
            return True
        except Exception:
            return False
    
    def get_vector_store_status(self, vector_store_id: str) -> Dict[str, Any]:
        """
        Get the status and details of a vector store.
        
        Args:
            vector_store_id (str): ID of the vector store
            
        Returns:
            Dict[str, Any]: Vector store details including status and file counts
        """
        vector_store = self.client.vector_stores.retrieve(vector_store_id)
        return {
            "id": vector_store.id,
            "name": vector_store.name,
            "status": vector_store.status,
            "file_counts": vector_store.file_counts,
            "created_at": vector_store.created_at,
            "last_active_at": vector_store.last_active_at
        }
    
    def get_store_id_by_name(self, name: str) -> Optional[str]:
        """
        Get the vector store ID by name.
        
        Args:
            name (str): Name of the vector store to find
            
        Returns:
            Optional[str]: Vector store ID if found, None otherwise
        """
        vector_stores = self.list_vector_stores()
        for vs in vector_stores:
            if vs["name"] == name:
                return vs["id"]
        return None
    
    def search_for_file(self, vector_store_id: str, query: str) -> Optional[str]:
        """
        Search vector store and return the filename that most closely matches the query.
        
        Args:
            vector_store_id (str): ID of the vector store to search
            query (str): Search query
            
        Returns:
            Optional[str]: Filename of the most relevant result, None if no results
        """
        results = self.search_vector_store(vector_store_id, query, limit=1)
        
        if results and len(results) > 0:
            # Extract filename from metadata if available
            metadata = results[0].get("metadata", {})
            filename = metadata.get("filename") or metadata.get("file_name") or metadata.get("source")
            return filename
        
        return None