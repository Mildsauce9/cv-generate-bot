# One-time bulk processing


class GitHubEmbeddingPipeline:
    def process_repository(self, repo_url):
        # Extract code files
        files = self.extract_code_files(repo_url)
        
        # Chunk code intelligently
        chunks = self.chunk_by_function_and_class(files)
        
        # Generate embeddings in batches
        embeddings = self.batch_embed(chunks)
        
        # Store with rich metadata
        self.store_with_metadata(embeddings, chunks, repo_url)