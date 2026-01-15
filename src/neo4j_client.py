import hashlib
import logging
from typing import Dict, List, Optional

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j client for storing and querying citation graphs."""

    def __init__(self, uri: str, username: str, password: str, database: str):
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database

    def close(self):
        self._driver.close()

    def upsert_query_graph(self, query: str, papers: List[Dict]) -> str:
        query_id = hashlib.md5(query.strip().lower().encode()).hexdigest()
        paper_rows = []
        citation_rows = []
        seen = {}

        for paper in papers:
            paper_id = self._paper_id(paper)
            if not paper_id:
                continue
            if paper_id in seen:
                continue
            seen[paper_id] = True

            paper_rows.append({
                "id": paper_id,
                "title": paper.get("title", ""),
                "source": paper.get("source", ""),
                "year": paper.get("year", ""),
                "citation": paper.get("citation", ""),
                "cited_by": paper.get("citation_count") or paper.get("cited_by") or 0
            })

        openalex_ids = {p.get("openalex_id"): self._paper_id(p) for p in papers if p.get("openalex_id")}

        for paper in papers:
            source_openalex = paper.get("openalex_id")
            if not source_openalex:
                continue
            source_id = openalex_ids.get(source_openalex)
            if not source_id:
                continue
            for ref in paper.get("references", []) or []:
                target_id = openalex_ids.get(ref)
                if target_id:
                    citation_rows.append({"source": source_id, "target": target_id})

        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MERGE (q:Query {id: $query_id})
                SET q.text = $query, q.updated_at = timestamp()
                """,
                query_id=query_id,
                query=query
            )
            if paper_rows:
                session.run(
                    """
                    UNWIND $papers AS row
                    MERGE (p:Paper {id: row.id})
                    SET p.title = row.title,
                        p.source = row.source,
                        p.year = row.year,
                        p.citation = row.citation,
                        p.cited_by = row.cited_by
                    """,
                    papers=paper_rows
                )
                session.run(
                    """
                    MATCH (q:Query {id: $query_id})
                    UNWIND $papers AS row
                    MATCH (p:Paper {id: row.id})
                    MERGE (q)-[:RETURNED]->(p)
                    """,
                    query_id=query_id,
                    papers=paper_rows
                )
            if citation_rows:
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (a:Paper {id: row.source})
                    MATCH (b:Paper {id: row.target})
                    MERGE (a)-[:CITES]->(b)
                    """,
                    rows=citation_rows
                )

        return query_id

    def fetch_query_graph(self, query_id: str) -> Dict:
        with self._driver.session(database=self._database) as session:
            paper_records = session.run(
                """
                MATCH (q:Query {id: $query_id})-[:RETURNED]->(p:Paper)
                RETURN p.id AS id, p.title AS title, p.source AS source,
                       p.year AS year, p.citation AS citation, p.cited_by AS cited_by
                """,
                query_id=query_id
            ).data()

            edges = session.run(
                """
                MATCH (q:Query {id: $query_id})-[:RETURNED]->(p:Paper)
                WITH collect(p) AS papers
                MATCH (a:Paper)-[:CITES]->(b:Paper)
                WHERE a IN papers AND b IN papers
                RETURN a.id AS source, b.id AS target
                """,
                query_id=query_id
            ).data()

        nodes = []
        for row in paper_records:
            cited_by = row.get("cited_by") or 0
            nodes.append({
                "id": row["id"],
                "label": row.get("title") or "Untitled",
                "group": "paper",
                "size": min(30, 6 + int(cited_by) ** 0.5),
                "meta": f"{row.get('citation', '')} | {row.get('source', '')} | {row.get('year', 'N/A')}"
            })

        return {
            "nodes": nodes,
            "links": [{"source": edge["source"], "target": edge["target"]} for edge in edges]
        }

    @staticmethod
    def _paper_id(paper: Dict) -> Optional[str]:
        if paper.get("openalex_id"):
            return paper["openalex_id"]
        if paper.get("id"):
            return f"{paper.get('source', 'Paper')}:{paper['id']}"
        if paper.get("citation"):
            return paper["citation"]
        title = paper.get("title", "").strip().lower()
        if title:
            return hashlib.md5(title.encode()).hexdigest()
        return None
