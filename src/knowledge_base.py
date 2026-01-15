import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict
import hashlib
import logging

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """ChromaDB-based knowledge base for academic papers"""

    def __init__(self, persist_directory: str):
        embedder = embedding_functions.ONNXMiniLM_L6_V2()
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name="academic_papers",
            metadata={"description": "Academic papers from multiple sources"},
            embedding_function=embedder
        )

        logger.info(f"Knowledge base initialized: {self.collection.count()} papers loaded")

    def add_papers(self, papers: List[Dict]) -> int:
        """Add papers to knowledge base"""
        if not papers:
            return 0

        documents = []
        metadatas = []
        ids = []

        for paper in papers:
            doc_text = f"""
            Title: {paper.get('title', '')}
            Authors: {', '.join(paper.get('authors', [])[:5])}
            Abstract: {paper.get('abstract', '')}
            Year: {paper.get('year', 'N/A')}
            Source: {paper.get('source', '')}
            """

            documents.append(doc_text)

            metadatas.append({
                "title": paper.get("title", "")[:500],
                "source": paper.get("source", ""),
                "citation": paper.get("citation", ""),
                "year": str(paper.get("year", "")),
                "paper_id": paper.get("id", "")
            })

            paper_id = hashlib.md5(
                paper.get("title", "").encode()
            ).hexdigest()
            ids.append(paper_id)

        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(papers)} papers to knowledge base")
            return len(papers)
        except Exception as e:
            logger.error(f"Failed to add papers: {e}")
            return 0

    def search(self, query: str, n_results: int = 10) -> List[Dict]:
        """Search knowledge base"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )

            papers = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i]
                    papers.append({
                        "content": doc,
                        "title": metadata.get("title", ""),
                        "source": metadata.get("source", ""),
                        "citation": metadata.get("citation", ""),
                        "year": metadata.get("year", ""),
                        "distance": results["distances"][0][i] if "distances" in results else None
                    })

            logger.info(f"Found {len(papers)} relevant papers in knowledge base")
            return papers

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def clear(self):
        """Clear knowledge base"""
        try:
            self.client.delete_collection("academic_papers")
            self.collection = self.client.get_or_create_collection(
                name="academic_papers"
            )
            logger.info("Knowledge base cleared")
        except Exception as e:
            logger.error(f"Failed to clear: {e}")

    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        return {
            "total_papers": self.collection.count(),
            "collection_name": self.collection.name
        }
