from langchain_groq import ChatGroq
from groq import Groq
from pydantic.v1.error_wrappers import ValidationError
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class QAEngine:
    """Question answering engine with academic sources"""

    def __init__(self, config, knowledge_base, search_engine):
        self.config = config
        self.kb = knowledge_base
        self.search_engine = search_engine
        self.llm = self._create_llm()

        logger.info(f"QA Engine initialized with {config.DEFAULT_MODEL}")

    def _create_llm(self):
        """Create LLM instance"""
        provider = self.config.DEFAULT_MODEL
        model_config = self.config.MODELS[provider]

        if provider == "groq":
            try:
                return ChatGroq(
                    model=model_config["name"],
                    temperature=model_config["temperature"],
                    max_tokens=model_config["max_tokens"],
                    groq_api_key=self.config.GROQ_API_KEY
                )
            except (ValidationError, TypeError) as e:
                logger.warning("ChatGroq init failed, falling back to Groq SDK: %s", e)
                return GroqSDKWrapper(
                    api_key=self.config.GROQ_API_KEY,
                    model=model_config["name"],
                    temperature=model_config["temperature"],
                    max_tokens=model_config["max_tokens"]
                )
        if provider == "openai":
            return ChatOpenAI(
                model=model_config["name"],
                temperature=model_config["temperature"],
                max_tokens=model_config["max_tokens"],
                api_key=self.config.OPENAI_API_KEY
            )
        if provider == "anthropic":
            return ChatAnthropic(
                model=model_config["name"],
                temperature=model_config["temperature"],
                max_tokens=model_config["max_tokens"],
                api_key=self.config.ANTHROPIC_API_KEY
            )
        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_config["name"],
                temperature=model_config["temperature"],
                google_api_key=self.config.GOOGLE_API_KEY
            )
        raise ValueError(f"Unsupported provider: {provider}")

    def answer(self, question: str, use_kb: bool = True, search_new: bool = True) -> Dict:
        """Generate answer with academic sources"""
        logger.info(f"Processing question: {question[:100]}...")

        sources = []

        if use_kb:
            kb_papers = self.kb.search(question, n_results=5)
            sources.extend(kb_papers)
            logger.info(f"Found {len(kb_papers)} papers in knowledge base")

        if search_new:
            new_results = self.search_engine.search_all(question, papers_per_source=3)
            new_papers = self.search_engine.flatten_results(new_results)[:10]

            if new_papers:
                self.kb.add_papers(new_papers)

            for paper in new_papers:
                sources.append({
                    "content": f"Title: {paper['title']}\nAbstract: {paper['abstract']}",
                    "title": paper["title"],
                    "source": paper["source"],
                    "citation": paper["citation"],
                    "year": paper.get("year", "N/A"),
                    "openalex_id": paper.get("openalex_id"),
                    "references": paper.get("references", []),
                    "citation_count": paper.get("citation_count"),
                    "cited_by": paper.get("cited_by")
                })

            logger.info(f"Found {len(new_papers)} new papers from live search")

        if not sources:
            return {
                "answer": "I couldn't find any relevant academic papers for your question. Try rephrasing or using different keywords.",
                "sources": [],
                "total_sources": 0
            }

        context = self._format_context(sources[:15])

        prompt = f"""You are an expert research assistant. Answer the following question based ONLY on the provided academic papers.

CRITICAL RULES:
1. Cite every claim with [Source: Citation]
2. Use multiple papers to provide comprehensive answers
3. Compare findings across papers when relevant
4. If papers contradict, mention both viewpoints
5. Be specific about which paper says what

Context from Academic Papers:
{context}

Question: {question}

Provide a detailed, well-cited answer:"""

        try:
            response = self.llm.invoke(prompt)
            answer = response.content

            return {
                "answer": answer,
                "sources": sources,
                "total_sources": len(sources),
                "kb_sources": len([s for s in sources if "distance" in s]),
                "new_sources": len([s for s in sources if "distance" not in s])
            }

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": sources,
                "total_sources": len(sources)
            }

    def _format_context(self, sources: List[Dict]) -> str:
        """Format sources as context"""
        context_parts = []

        for i, source in enumerate(sources, 1):
            context_parts.append(
                f"[Paper {i}] {source['citation']}\n"
                f"Source: {source['source']}\n"
                f"{source['content'][:800]}...\n"
            )

        return "\n---\n".join(context_parts)


class GroqSDKWrapper:
    """Minimal Groq SDK wrapper with a LangChain-like invoke API."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required for Groq SDK.")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        content = ""
        if response and response.choices:
            content = response.choices[0].message.content or ""
        return type("GroqResponse", (), {"content": content})
