import re
from typing import List
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    title: str
    content: str

class DocumentChunker:
    def __init__(self, file_path: str, file_name:str):
        with open(file_path, "r", encoding='utf-8') as file:
            self.content = file.read()
        self.max_chunk_size = 1024
        self.file_name = file_name
        self.chunks = []

    def extract_sections(self) -> List[DocumentChunk]:
        """Extract sections from markdown-style documentation."""
        # Split the text into major sections first (denoted by === or ---)

        chunks = []
        major_sections = re.split(r'\n[=\-]{3,}\n', self.content)
        
        current_major_section = ""

        for section in major_sections:
            if not section.strip():
                continue
                
            # Extract title if present (first line)
            lines = section.strip().split('\n')
            title = lines[0].strip('# ')
            current_major_section = title
            
            # Split by Q&A patterns
            qa_pairs = re.split(r'\*\*Q:', section)
            
            for qa in qa_pairs:
                if not qa.strip():
                    continue
                    
                if '**' in qa:  # This is a Q&A pair
                    # Extract question and answer
                    question = qa.split('**')[0].strip()
                    answer = ''.join(qa.split('**')[1:]).strip()
                    
                    # Create chunk with metadata
                    chunk = DocumentChunk(
                        title=f"Q: {question}",
                        content=f"Q: {question}\nA: {answer}",
                    )
                    chunks.append(chunk)
                else:
                    # This is introduction or non-Q&A content
                    if len(qa.strip()) < self.max_chunk_size:
                        chunk = DocumentChunk(
                            title=current_major_section,
                            content=qa.strip(),
                        )
                        chunks.append(chunk)
        return chunks
        


    def merge_small_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Merge chunks that are too small."""
        current_chunk = None
        
        for chunk in chunks:
            if not current_chunk:
                current_chunk = chunk
                continue
                
            if len(current_chunk.content) + len(chunk.content) <= self.max_chunk_size:
                # Merge with next chunk
                current_chunk.content += f"\n\n{chunk.content}"
            else:
                self.chunks.append(current_chunk)
                current_chunk = chunk
                
        if current_chunk:
            self.chunks.append(current_chunk)
            
    
    def create_chunks(self) -> List[DocumentChunk]:
        """Create all chunks from document."""
        self.chunks = []
        extracted_chunks = self.extract_sections()
        self.merge_small_chunks(extracted_chunks)
        return self.chunks

def main():

    
    chunker = DocumentChunker(file_path = "./metadata.txt", file_name = "metadata.txt")
    chunks = chunker.create_chunks()
    
    # Print results
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Title: {chunk.title}")
        print(f"Content: {chunk.content}...")  # Print first 100 chars
        print("-" * 80)

if __name__ == "__main__":
    main()