import requests
import arxiv
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AcademicSearchEngine:
    """Unified academic database search engine"""

    def __init__(self, config):
        self.config = config
        self.cache_dir = config.CACHE_DIR

    def search_all(self, query: str, papers_per_source: int = 5) -> Dict[str, List[Dict]]:
        """Search all academic databases"""
        logger.info(f"🔍 Searching academic databases for: {query}")

        results = {
            "arxiv": self._search_arxiv(query, papers_per_source),
            "semantic_scholar": self._search_semantic_scholar(query, papers_per_source),
            "crossref": self._search_crossref(query, papers_per_source),
            "openalex": self._search_openalex(query, papers_per_source)
        }

        total = sum(len(papers) for papers in results.values())
        logger.info(f"✅ Found {total} papers across all sources")

        return results

    def _search_arxiv(self, query: str, limit: int) -> List[Dict]:
        """Search arXiv"""
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=limit,
                sort_by=arxiv.SortCriterion.Relevance
            )

            papers = []
            for result in client.results(search):
                papers.append({
                    "id": result.entry_id.split("/")[-1],
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "abstract": result.summary,
                    "published": result.published.strftime("%Y-%m-%d"),
                    "url": result.pdf_url,
                    "source": "arXiv",
                    "citation": f"arXiv:{result.entry_id.split('/')[-1]}",
                    "year": result.published.year
                })

            logger.info(f"📄 arXiv: {len(papers)} papers")
            return papers

        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return []

    def _search_semantic_scholar(self, query: str, limit: int) -> List[Dict]:
        """Search Semantic Scholar"""
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            headers = {}
            if self.config.SEMANTIC_SCHOLAR_API_KEY:
                headers["x-api-key"] = self.config.SEMANTIC_SCHOLAR_API_KEY

            params = {
                "query": query,
                "limit": limit,
                "fields": "paperId,title,abstract,year,authors,citationCount,publicationDate,externalIds,fieldsOfStudy"
            }

            last_error = None
            for attempt in range(3):
                try:
                    response = requests.get(url, params=params, headers=headers, timeout=20)
                    response.raise_for_status()
                    data = response.json()
                    break
                except requests.RequestException as exc:
                    last_error = exc
                    if attempt < 2:
                        logger.warning("Semantic Scholar attempt %s failed: %s", attempt + 1, exc)
                    else:
                        raise
            
            if last_error:
                logger.warning("Semantic Scholar search recovered after retry.")
            papers = []

            for paper in data.get("data", []):
                papers.append({
                    "id": paper.get("paperId"),
                    "title": paper.get("title"),
                    "authors": [a.get("name") for a in paper.get("authors", [])],
                    "abstract": paper.get("abstract", "No abstract available"),
                    "year": paper.get("year"),
                    "citation_count": paper.get("citationCount", 0),
                    "fields": paper.get("fieldsOfStudy", []),
                    "source": "Semantic Scholar",
                    "citation": f"S2:{paper.get('paperId', '')[:10]}",
                    "published": paper.get("publicationDate", "")
                })

            logger.info(f"🔬 Semantic Scholar: {len(papers)} papers")
            return papers

        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return []

    def _search_crossref(self, query: str, limit: int) -> List[Dict]:
        """Search CrossRef"""
        try:
            url = "https://api.crossref.org/works"
            headers = {
                "User-Agent": f"PaperMind/2.0 (mailto:{self.config.CROSSREF_EMAIL})"
            }
            params = {
                "query": query,
                "rows": limit,
                "select": "DOI,title,author,published-print,abstract,container-title"
            }

            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            papers = []

            for item in data.get("message", {}).get("items", []):
                year = None
                if "published-print" in item:
                    date_parts = item["published-print"].get("date-parts", [[]])[0]
                    year = date_parts[0] if date_parts else None

                papers.append({
                    "id": item.get("DOI"),
                    "title": item.get("title", [""])[0],
                    "authors": [
                        f"{a.get('given', '')} {a.get('family', '')}"
                        for a in item.get("author", [])
                    ],
                    "abstract": item.get("abstract", "No abstract available"),
                    "journal": item.get("container-title", [""])[0],
                    "year": year,
                    "source": "CrossRef",
                    "citation": f"DOI:{item.get('DOI', '')}"
                })

            logger.info(f"📚 CrossRef: {len(papers)} papers")
            return papers

        except Exception as e:
            logger.error(f"CrossRef search failed: {e}")
            return []

    def _search_openalex(self, query: str, limit: int) -> List[Dict]:
        """Search OpenAlex"""
        try:
            url = "https://api.openalex.org/works"
            params = {
                "search": query,
                "per-page": limit,
                "select": "id,title,authorships,abstract,publication_year,cited_by_count,referenced_works"
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            papers = []

            for work in data.get("results", []):
                papers.append({
                    "id": work.get("id", "").split("/")[-1],
                    "openalex_id": work.get("id"),
                    "title": work.get("title"),
                    "authors": [
                        a.get("author", {}).get("display_name")
                        for a in work.get("authorships", [])
                    ],
                    "abstract": work.get("abstract", "No abstract available"),
                    "year": work.get("publication_year"),
                    "cited_by": work.get("cited_by_count", 0),
                    "source": "OpenAlex",
                    "citation": f"OpenAlex:{work.get('id', '').split('/')[-1]}",
                    "references": work.get("referenced_works", [])
                })

            logger.info(f"🌐 OpenAlex: {len(papers)} papers")
            return papers

        except Exception as e:
            logger.error(f"OpenAlex search failed: {e}")
            return []

    def flatten_results(self, results: Dict[str, List[Dict]]) -> List[Dict]:
        """Flatten and deduplicate results"""
        all_papers = []
        seen_titles = set()

        for papers in results.values():
            for paper in papers:
                title = paper.get("title", "").lower().strip()
                if title and title not in seen_titles and len(title) > 10:
                    seen_titles.add(title)
                    all_papers.append(paper)

        all_papers.sort(
            key=lambda x: (x.get("citation_count", 0) or x.get("cited_by", 0), x.get("year", 0) or 0),
            reverse=True
        )

        return all_papers
