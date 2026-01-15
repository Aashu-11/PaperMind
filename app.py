import json
import streamlit as st
import streamlit.components.v1 as components
from src.config import config
from src.academic_search import AcademicSearchEngine
from src.knowledge_base import KnowledgeBase
from src.qa_engine import QAEngine
from src.neo4j_client import Neo4jClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="PaperMind - Academic Q&A",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-card {
        background: #fff4d6;
        border-left: 4px solid #f59e0b;
        color: #111827;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session():
    """Initialize session state"""
    if "initialized" not in st.session_state:
        st.session_state.search_engine = AcademicSearchEngine(config)
        st.session_state.kb = KnowledgeBase(str(config.CHROMA_DIR))
        st.session_state.neo4j = None
        if config.NEO4J_URI and config.NEO4J_USERNAME and config.NEO4J_PASSWORD:
            st.session_state.neo4j = Neo4jClient(
                config.NEO4J_URI,
                config.NEO4J_USERNAME,
                config.NEO4J_PASSWORD,
                config.NEO4J_DATABASE
            )
        st.session_state.qa_engine = QAEngine(
            config,
            st.session_state.kb,
            st.session_state.search_engine
        )
        st.session_state.chat_history = []
        st.session_state.initialized = True
        logger.info("Session initialized")


def validate_api_keys():
    """Check if API keys are configured"""
    providers = {
        "Groq": config.GROQ_API_KEY,
        "OpenAI": config.OPENAI_API_KEY,
        "Anthropic": config.ANTHROPIC_API_KEY,
        "Google": config.GOOGLE_API_KEY
    }

    active = [name for name, key in providers.items() if key]

    if not active:
        st.error("⚠️ No API keys configured! Please set up at least one in .env file.")
        st.code("""
# Create .env file with:
GROQ_API_KEY=gsk_...  # Recommended - Free tier available!
        """)
        return False

    return True


def record_graph(query: str, sources):
    """Persist graph data to Neo4j if configured."""
    client = st.session_state.get("neo4j")
    if not client:
        return None
    try:
        query_id = client.upsert_query_graph(query, sources)
        st.session_state.last_graph_query_id = query_id
        return query_id
    except Exception as e:
        logger.warning("Neo4j write failed: %s", e)
        return None


def render_header():
    """Render main header"""
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0;">🔬 PaperMind</h1>
        <p style="margin:0.5rem 0 0 0; font-size:1.2rem;">
            Academic Research Q&A Engine
        </p>
        <p style="margin:0.3rem 0 0 0; opacity:0.9;">
            Powered by arXiv • Semantic Scholar • CrossRef • OpenAlex
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar"""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        model = st.selectbox(
            "AI Model",
            options=["groq", "openai", "anthropic", "google"],
            index=0,
            help="Select your AI model provider"
        )

        if model != config.DEFAULT_MODEL:
            config.DEFAULT_MODEL = model
            st.session_state.qa_engine = QAEngine(
                config,
                st.session_state.kb,
                st.session_state.search_engine
            )
            st.success(f"Switched to {model.title()}")

        st.markdown("---")

        stats = st.session_state.kb.get_stats()
        st.markdown("### 📊 Knowledge Base")
        st.metric("Papers Indexed", stats["total_papers"])

        if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
            st.session_state.kb.clear()
            st.success("Knowledge base cleared!")
            st.rerun()

        st.markdown("---")

        st.markdown("### 📡 API Status")
        apis = {
            "Groq": "✅" if config.GROQ_API_KEY else "❌"
        }

        for name, status in apis.items():
            st.text(f"{status} {name}")

        st.markdown("---")

        st.markdown("""
        ### 📖 How It Works

        1. **Ask any research question**
        2. **Searches 4 academic databases**
        3. **AI synthesizes answer**
        4. **Every claim is cited**
        """)


def render_chat():
    """Render chat interface"""
    st.markdown("### 💬 Ask Research Questions")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander(f"📚 View {msg['total_sources']} Sources"):
                    for i, source in enumerate(msg["sources"][:10], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>[{i}] {source['title']}</strong><br>
                            <small>{source['citation']} | {source['source']} | {source['year']}</small>
                        </div>
                        """, unsafe_allow_html=True)

    if prompt := st.chat_input("Ask anything about academic research..."):
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching academic databases and generating answer..."):
                response = st.session_state.qa_engine.answer(
                    prompt,
                    use_kb=True,
                    search_new=True
                )

            st.markdown(response["answer"])

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response["sources"],
                "total_sources": response["total_sources"]
            })
            record_graph(prompt, response["sources"])
            st.session_state.last_graph_sources = response["sources"]
            st.session_state.last_graph_query = prompt

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sources", response["total_sources"])
            with col2:
                st.metric("From Knowledge Base", response.get("kb_sources", 0))
            with col3:
                st.metric("New Searches", response.get("new_sources", 0))


def render_direct_search():
    """Render direct database search"""
    st.markdown("### 🔍 Direct Database Search")
    st.info("Search academic databases without AI synthesis - see raw results")

    search_query = st.text_input("Enter search query:", key="direct_search_query")
    col1, col2 = st.columns([1, 1])

    with col1:
        search_clicked = st.button("Search", type="primary")
    with col2:
        clear_clicked = st.button("Clear Results")

    if clear_clicked:
        st.session_state.pop("last_search_results", None)
        st.session_state.pop("last_search_query", None)

    if search_clicked:
        if not search_query.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching..."):
                results = st.session_state.search_engine.search_all(
                    search_query,
                    papers_per_source=config.PAPERS_PER_SOURCE
                )
                flat_results = st.session_state.search_engine.flatten_results(results)

            st.session_state.last_search_results = flat_results
            st.session_state.last_search_query = search_query
            st.session_state.last_graph_sources = flat_results
            st.session_state.last_graph_query = search_query
            record_graph(search_query, flat_results)

    if "last_search_results" in st.session_state:
        flat_results = st.session_state.last_search_results
        query_label = st.session_state.get("last_search_query", "")
        st.success(f"Found {len(flat_results)} papers for \"{query_label}\"")

        if flat_results:
            if st.button("➕ Add All to Knowledge Base", use_container_width=True):
                added = st.session_state.kb.add_papers(flat_results)
                st.success(f"Added {added} papers to knowledge base")

        for i, paper in enumerate(flat_results, 1):
            authors = ", ".join([a for a in paper.get("authors", []) if a])
            if not authors:
                authors = "Unknown authors"

            st.markdown(f"""
            <div class="source-card">
                <strong>[{i}] {paper.get('title', 'Untitled')}</strong><br>
                <small>{paper.get('citation', '')} | {paper.get('source', '')} | {paper.get('year', 'N/A')}</small><br>
                <small>{authors}</small>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Abstract"):
                st.write(paper.get("abstract", "No abstract available"))


def render_citation_graph():
    """Render D3.js citation graph"""
    st.markdown("### 🕸️ Citations Graph")
    st.info("Interactive graph of the latest query results stored in Neo4j.")

    query_id = st.session_state.get("last_graph_query_id")
    client = st.session_state.get("neo4j")

    if not client:
        st.warning("Neo4j is not configured. Add NEO4J_* values in .env and restart.")
        return

    if not query_id:
        st.warning("No graph data yet. Ask a question or run a search first.")
        return

    graph = client.fetch_query_graph(query_id)
    if not graph["nodes"]:
        st.warning("No citation edges available for this query yet.")
        return

    graph_data = json.dumps(graph)

    components.html(
        f"""
        <div id="graph" style="width:100%; height:620px; background:#0f172a; border-radius:12px;"></div>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <script>
        const data = {graph_data};
        const width = document.getElementById("graph").clientWidth;
        const height = 620;

        const svg = d3.select("#graph")
          .append("svg")
          .attr("width", width)
          .attr("height", height);

        const color = d3.scaleOrdinal()
          .domain(["paper"])
          .range(["#22d3ee"]);

        const simulation = d3.forceSimulation(data.nodes)
          .force("link", d3.forceLink(data.links).id(d => d.id).distance(140))
          .force("charge", d3.forceManyBody().strength(-320))
          .force("center", d3.forceCenter(width / 2, height / 2));

        const link = svg.append("g")
          .attr("stroke", "#38bdf8")
          .attr("stroke-opacity", 0.5)
          .selectAll("line")
          .data(data.links)
          .enter().append("line")
          .attr("stroke-width", 1.5);

        const node = svg.append("g")
          .selectAll("circle")
          .data(data.nodes)
          .enter().append("circle")
          .attr("r", d => d.size)
          .attr("fill", d => color(d.group))
          .attr("stroke", "#0f172a")
          .attr("stroke-width", 1.5)
          .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

        const label = svg.append("g")
          .selectAll("text")
          .data(data.nodes)
          .enter().append("text")
          .text(d => d.label.length > 50 ? d.label.slice(0, 50) + "..." : d.label)
          .attr("font-size", "12px")
          .attr("fill", "#e2e8f0");

        node.append("title")
          .text(d => d.meta ? d.meta : d.label);

        simulation.on("tick", () => {{
          link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

          node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);

          label
            .attr("x", d => d.x + 8)
            .attr("y", d => d.y + 4);
        }});

        function dragstarted(event, d) {{
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        }}

        function dragged(event, d) {{
          d.fx = event.x;
          d.fy = event.y;
        }}

        function dragended(event, d) {{
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }}
        </script>
        """,
        height=640
    )


def main():
    """Main application"""
    render_header()

    if not validate_api_keys():
        return

    initialize_session()
    render_sidebar()

    tab1, tab2, tab3 = st.tabs(["💬 Q&A Chat", "🔍 Direct Search", "🕸️ Citations Graph"])

    with tab1:
        render_chat()

        st.markdown("---")
        st.markdown("#### 💡 Try These Questions")

        sample_questions = [
            "What are the latest advances in transformer architectures?",
            "Explain diffusion models for image generation",
            "What is retrieval augmented generation (RAG)?",
            "How do large language models work?",
            "What are the current challenges in computer vision?",
            "Explain few-shot learning techniques"
        ]

        cols = st.columns(3)
        for i, question in enumerate(sample_questions):
            with cols[i % 3]:
                if st.button(question, key=f"sample_{i}", use_container_width=True):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": question
                    })
                    response = st.session_state.qa_engine.answer(
                        question,
                        use_kb=True,
                        search_new=True
                    )
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"],
                        "total_sources": response["total_sources"]
                    })
                    st.rerun()

    with tab2:
        render_direct_search()
    with tab3:
        render_citation_graph()


if __name__ == "__main__":
    main()
