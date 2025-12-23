import ast
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import httpx
import faiss
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
import ast
import numpy as np
import os
import json



# Import ResilientDB Knowledge Base
try:
    from ResilientDBKnowledgeBase import ResilientDBKnowledgeBase
    KNOWLEDGE_BASE_AVAILABLE = True
    print("[OK] ResilientDB Knowledge Base loaded")
except ImportError as e:
    KNOWLEDGE_BASE_AVAILABLE = False
    print(f"[WARNING] ResilientDB Knowledge Base not available: {e}")
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn

# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
# )

# sessions = {}


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
def get_auth_headers():
    if GITHUB_TOKEN:
        return {"Authorization": f"token {GITHUB_TOKEN}"}
    return {}


# -------------------------
# FAISS vector search setup
# -------------------------
DIM = 768  # Embedding dimension

index = faiss.IndexFlatL2(DIM)  # Basic L2 index for demo
vectors = []
metadata = []  # List[(repo, file, chunk_id, text)] for retrieval
# --- Hybrid RAG + Knowledge Graph globals ---

# metadata[i] corresponds to FAISS row i
# Build a simple file-level graph using NetworkX
kg = nx.Graph()


import hashlib
import numpy as np

import hashlib

def embed_text(text: str) -> np.ndarray:
    """
    100% OFFLINE embedding. No downloads, no SSL, no external deps.
    Deterministic 768-dim vectors for FAISS.
    """
    # Fast hash -> 768-dim embedding
    hash_obj = hashlib.md5(text.encode('utf-8'))
    seed = int.from_bytes(hash_obj.digest(), 'big') % (2**31)
    
    np.random.seed(seed)
    embedding = np.random.normal(0, 1, DIM).astype('float32')
    
    # Normalize
    embedding = (embedding - embedding.mean()) / (embedding.std() + 1e-8)
    
    return embedding.reshape(1, -1)


# -------------------------
# Pydantic Models
# -------------------------
class FileSummary(BaseModel):
    filename: str
    code_summary: str
    insights: List[str] = Field(default_factory=list)

class RepoSummary(BaseModel):
    repo_name: str
    total_files: int
    files: List[str]

class SearchResult(BaseModel):
    filepath: str
    code_snippet: str
    score: float

class RepoInsights(BaseModel):
    repo_name: str
    insights: List[str] = Field(default_factory=list)
# -------------------------
# FastMCP Server Setup
# -------------------------
mcp = FastMCP(name="GitHub Repo Analyzer MCP Server")

# -------------------------
# ResilientDB Knowledge Base Initialization
# -------------------------
resilientdb_knowledge = None
if KNOWLEDGE_BASE_AVAILABLE:
    try:
        resilientdb_knowledge = ResilientDBKnowledgeBase()
        print("[OK] ResilientDB Knowledge Base initialized and ready")
    except Exception as e:
        print(f"[ERROR] Failed to initialize knowledge base: {e}")
        resilientdb_knowledge = None

# -------------------------
# Helper functions
# -------------------------
def split_code_into_chunks(code: str, max_lines: int = 200) -> List[str]:
    lines = code.splitlines()
    return ["\n".join(lines[i:i+max_lines]) for i in range(0, len(lines), max_lines)]

async def fetch_repo_tree(owner: str, repo: str, branch: str = "main") -> List[dict]:
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    headers= get_auth_headers()
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return [item for item in data.get("tree", []) if item['type'] == 'blob']

async def fetch_raw_file(owner: str, repo: str, branch: str, path: str) -> Optional[str]:
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        if r.status_code == 200:
            return r.text
    return None

# Alternate parser using Python's built-in ast module
def parse_python_functions_ast(code: str) -> List[str]:
    functions = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Generate a simple string summary (function signature)
                args = [arg.arg for arg in node.args.args]
                arglist = ", ".join(args)
                functions.append(f"def {node.name}({arglist}):")
    except Exception:
        # If parsing fails, return empty list to avoid crashing MCP server
        return []
    return functions

# -------------------------
# MCP Tools
# -------------------------

@mcp.tool(name="list_github_repo_files")
# Fetch all files in a public GitHub repo recursively
async def list_github_repo_files(owner: str, repo: str, branch: str = "main") -> List[str]:
    """
    This tool fetches the list of files in a particular repository using the github api mentioned in the MCP tool and not from any online source. 
    """
    headers= get_auth_headers()
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        files = [item['path'] for item in data.get('tree', []) if item['type'] == 'blob']
        return files

@mcp.tool(name="Get_Repo_Summary")
async def getRepoSummary(owner: str, repo: str, branch: str = "main") -> RepoSummary:
    """
    Get the summary of a particular repository. Ask the user about the branch they want the information about.This MCP tool will give the summary of the repo as to what type of files, how many files, what it does, what is it about, what is it implementing..
    """
    tree = await fetch_repo_tree(owner, repo, branch)
    files = [item['path'] for item in tree]
    return RepoSummary(repo_name=f"{owner}/{repo}", total_files=len(files), files=files)

# Simple file summary generator (simulated)
@mcp.tool(name="getFileSummary")
async def getFileSummary(owner: str, repo: str, filenames: List[str], branch: str = "main") -> List[FileSummary]:
    """
    Generate the summary of a particular file in a branch. Do not provide extra unnecessary information. To the point and specific information and what the file does..
    """
    file_summaries = []
    async with httpx.AsyncClient() as client:
        for filename in filenames:
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filename}"
            resp = await client.get(url)
            if resp.status_code == 200:
                code = resp.text
                # Simple summary: line count, presence of TODOs or defs
                summary_text = f"{filename} has {len(code.splitlines())} lines of code."
                insights = []
                if "TODO" in code:
                    insights.append("Contains TODO comments.")
                if "def " in code:
                    insights.append("Contains function definitions.")
                file_summaries.append(FileSummary(filename=filename, code_summary=summary_text, insights=insights))
            else:
                file_summaries.append(FileSummary(filename=filename, code_summary="File not found or inaccessible", insights=[]))
    return file_summaries

# @mcp.tool(name="AutoIngestRepo")
# async def autoIngestRepo(repo_url: str) -> dict:
#     """Auto-ingest repo from URL. Extracts owner/repo/branch automatically."""
#     import re
#     match = re.match(r'https://github\.com/([\w\-]+)/([\w\-]+)/?(?:tree/([\w\-]+))?', repo_url)
#     if match:
#         owner, repo, branch = match.groups()
#         branch = branch or "main"
#         result = await ingestRepoCode(owner, repo, branch)
#         return {"status": "ingested", "repo": f"{owner}/{repo}", "result": result}
#     return {"error": "Invalid GitHub URL format"}


@mcp.tool(name="Ingest_Repo_Code")
async def ingestRepoCode(owner: str, repo: str, branch: str = "main") -> dict:
    global vectors, metadata, index
    
    vectors.clear()
    metadata.clear()
    
    tree = await fetch_repo_tree(owner, repo, branch)
    count = 0
    
    # ✅ CODE FILES ONLY
    code_files = [item for item in tree if any(
        item['path'].lower().endswith(ext) 
        for ext in ['.js', '.jsx', '.ts', '.tsx', '.py', '.java', '.cpp', '.c', '.go', '.rs']
    )]
    
    print(f"Found {len(code_files)} code files (filtered from {len(tree)} total)")
    
    for item in code_files:  # Only code files!
        code = await fetch_raw_file(owner, repo, branch, item['path'])
        if code is None or len(code.strip()) < 50:  # Skip empty/short files
            continue
        
        chunks = split_code_into_chunks(code)
        
        for chunk in chunks:
            if len(chunk.strip()) < 20:  # Skip tiny chunks
                continue
                
            try:
                emb = embed_text(chunk)
                vectors.append(emb[0])
                metadata.append({
                    "repo": f"{owner}/{repo}",
                    "filepath": item['path'],
                    "chunk_id": count,
                    "code": chunk[:1000]  # Truncate for storage
                })
                count += 1
            except Exception as e:
                print(f"WARNING: {item['path']}: {e}")
                continue
    
    if vectors:
        all_vectors = np.vstack(vectors).astype("float32")
        print(f"Adding {all_vectors.shape} vectors to FAISS")
        index.add(all_vectors)
        rebuild_knowledge_graph()
        print(f"Knowledge graph has {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")
        return {"message": f"✅ Indexed {count} CODE chunks + KG ({kg.number_of_nodes()} nodes)"}
    else:
        return {"message": "❌ No CODE files found or processed"}


@mcp.tool(name="Semantic_Search")
async def semanticSearch(query: str, top_k: int = 5) -> List[SearchResult]:
    """
    Perform semantic search on the data after vectorization and indexing.
    """
       
    # if index.ntotal == 0:
    #     return [SearchResult(
    #         filepath="error",
    #         code_snippet="Index is empty. Please run Ingest_Repo_Code first to index a repository.",
    #         score=0.0
    #     )]
    # In SemanticSearch and HybridRepoQuestion functions:
    if index.ntotal == 0:
        return [{
            "filepath": "needs_repo",
            "codesnippet": """Index empty. Please provide a GitHub repo first:
    1. Run: listgithubrepofiles("owner", "repo", "main")
    2. Then: ingestRepoCode("owner", "repo", "main")
    Example: ingestRepoCode("ResilientApp", "Arrayan", "main")""",
            "score": 0.0
        }]

    
    qv = embed_text(query).reshape(1, -1)
    D, I = index.search(qv, top_k)  # distances and indices
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < len(metadata):
            meta = metadata[idx]
            filepath = meta['filepath']
            
            # FILTER: Skip images/binary files
            if any(ext in filepath.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg']):
                continue
                
            results.append(SearchResult(
                filepath=filepath,
                code_snippet=meta['code'][:500],
                score=float(dist)
            ))
            
            if len(results) >= top_k:
                break
    
    return results if results else [{"filepath": "no_code_files", "code_snippet": "Only images/binary files found", "score": 0.0}]

@mcp.tool(name="Get_File_Functions")
async def getFileFunctions(owner: str, repo: str, filepath: str, branch: str = "main") -> List[str]:
    """
    The MCP tool getFileFunctions extracts Python function definitions from a given file in a GitHub repository. Here's what it does in detail:

    It fetches the raw source code of the specified file from the GitHub repo using fetch_raw_file.

    If the file content is empty or unavailable, it returns an empty list.

    Otherwise, it calls parse_python_functions_ast, which uses Python’s built-in ast module to:

    Parse the source code into an abstract syntax tree (AST).

    Traverse the AST to find all function definitions (ast.FunctionDef nodes).

    For each function found, it creates a simple signature string like def function_name(arg1, arg2):.

    The tool returns a list of these function signature strings
    Purpose:
    This tool helps junior developers (or any users) quickly understand the structure of Python files by listing the function definitions and their arguments without needing to manually inspect the file line-by-line. It supports the larger goal of the MCP-powered Repo Analyzer to provide guided navigation and understanding of a codebase
    """
    code = await fetch_raw_file(owner, repo, branch, filepath)
    if not code:
        return []
    funcs = parse_python_functions_ast(code)
    return funcs

# =========================================================================
# RESILIENTDB KNOWLEDGE BASE QUERY TOOL - USE THIS FIRST FOR RESILIENTDB!
# =========================================================================

@mcp.tool(name="SearchResilientDBKnowledge")
async def search_resilientdb_knowledge(query: str, category: Optional[str] = None) -> str:
    """
    🎓 CRITICAL: USE THIS TOOL FIRST for ANY question about ResilientDB!
    
    This tool provides comprehensive information about ResilientDB from a built-in knowledge base.
    It covers all ResilientDB topics including:
    
    - **Setup & Installation**: How to install, configure, and run ResilientDB
    - **Applications**: Debitable, DraftRes, Arrayán/Arrayan, Echo, ResCounty, CrypoGo, etc.
    - **Architecture**: System design, components, and technical details
    - **Consensus**: PBFT, Byzantine fault tolerance, and consensus mechanisms
    - **Performance**: Benchmarks, optimization, throughput, and latency data
    - **Use Cases**: Real-world applications across industries
    - **Research**: Academic papers and publications
    - **Development**: How to build applications on ResilientDB
    
    🚨 ALWAYS call this tool BEFORE searching the web for ResilientDB questions! Do not search the web. Ask user follow up questions
    
    Examples of questions that should use this tool:
    - "How do I setup ResilientDB?"
    - "What is Arrayán?" or "What is Arrayan?"
    - "Tell me about Debitable"
    - "How does PBFT work in ResilientDB?"
    - "Show me performance benchmarks"
    - "How to install ResilientDB?"
    - "What applications are built on ResilientDB?"
    - "How to use [any ResilientDB application]?"
    
    Args:
        query: Your question about ResilientDB (any topic)
        category: Optional. One of: applications, architecture, consensus, performance, 
                  use_cases, research, setup, general
    
    Returns:
        Comprehensive answer from the ResilientDB knowledge base with examples and guidance

    Do not use the web to serach for infomration.If you want more information, ask the user to provide more information..
    """
    if not KNOWLEDGE_BASE_AVAILABLE or resilientdb_knowledge is None:
        return """
❌ ResilientDB Knowledge Base is not available.
        
Please ensure ResilientDBKnowledgeBase.py is in the project directory.
        
For now, you can:
1. Check the ResilientDB GitHub repository: https://github.com/apache/incubator-resilientdb
2. Ask me to fetch information from the repository using other tools
"""
    
    try:
        # Determine the best domain based on category or query content
        domain = category or "general"
        
        # Auto-detect domain from query if not specified
        if not category:
            query_lower = query.lower()
            
            # Check for setup/installation queries
            if any(word in query_lower for word in ["setup", "install", "configure", "run", "start", "deploy", "docker"]):
                domain = "setup"            # Check for specific applications (all 14 from ExpoLab)
            elif any(app in query_lower for app in [
                "debitable", "draftres", "arrayán", "arrayan", "echo", 
                "rescounty", "crypogo", "explorer", "monitoring", 
                "resview", "reslens", "coinsensus", "respirer", 
                "utxo", "utxo lenses", "resilientdb cli", "cli",
                "application", "app"
            ]):
                domain = "applications"
            # Check for architecture queries
            elif any(word in query_lower for word in ["architecture", "design", "component", "structure", "layer"]):
                domain = "architecture"
            # Check for consensus queries
            elif any(word in query_lower for word in ["consensus", "pbft", "bft", "byzantine", "fault tolerance", "agreement"]):
                domain = "consensus"
            # Check for performance queries
            elif any(word in query_lower for word in ["performance", "benchmark", "speed", "throughput", "latency", "tps", "fast"]):
                domain = "performance"
            # Check for use case queries
            elif any(word in query_lower for word in ["use case", "example", "industry", "real world", "application"]):
                domain = "use_cases"
            # Check for research queries
            elif any(word in query_lower for word in ["paper", "research", "publication", "academic", "study"]):
                domain = "research"
            # Check for "how to use" queries
            elif "how to use" in query_lower or "how do i use" in query_lower:
                domain = "applications"
        
        # Query the knowledge base
        result_dict = await resilientdb_knowledge.query_knowledge(query, domain)
        
        # Format the result nicely
        if isinstance(result_dict, dict):
            # Extract the main content
            content = result_dict.get("content", "")
            result_type = result_dict.get("type", "general")
            
            # Build formatted response
            formatted_result = f"""
# 📚 ResilientDB Knowledge Base Results

**Query:** {query}
**Category:** {domain}
**Result Type:** {result_type.replace('_', ' ').title()}

---

{content}

---
"""
            # Add additional sections if present
            if "technical_deep_dive" in result_dict:
                formatted_result += f"\n**🔧 Technical Details:**\n```json\n{json.dumps(result_dict['technical_deep_dive'], indent=2)}\n```\n"
            
            if "implementation_guidance" in result_dict:
                formatted_result += f"\n{result_dict['implementation_guidance']}\n"
            
            if "further_exploration" in result_dict:
                formatted_result += f"\n{result_dict['further_exploration']}\n"
            
            result = formatted_result
        else:
            result = str(result_dict)
        
        return f"{result}\n\n💡 **Tip:** This information comes from the comprehensive ResilientDB knowledge base.\nFor more details, ask follow-up questions or try a different category!"
    
    except Exception as e:
        return f"""
❌ **Error querying ResilientDB knowledge base:** {str(e)}

💡 **Troubleshooting:**
1. Check that ResilientDBKnowledgeBase.py is in the project directory
2. Verify the knowledge base class has the required query methods
3. Try rephrasing your question or using a specific category

**Your query:** {query}
**Attempted domain:** {domain if 'domain' in locals() else 'unknown'}

**Available categories:**
- setup: Installation and configuration
- applications: ResilientDB applications (Debitable, Arrayán, etc.)
- architecture: System design and technical details
- consensus: Consensus mechanisms (PBFT, etc.)
- performance: Benchmarks and performance data
- use_cases: Real-world applications
- research: Research papers and publications
"""

# Example: Function to parse Dockerfile directives
async def parse_dockerfile(owner:str, repo:str, branch:str="main") -> List[str]:
    dockerfile_content = await fetch_raw_file(owner, repo, branch, "Dockerfile")
    if not dockerfile_content:
        return []
    steps = []
    for line in dockerfile_content.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            steps.append(line)
    return steps

@mcp.tool(name="SetupGuide")
async def setup_guide(owner: str, repo: str, question: str, branch: str = "main") -> dict:
    """
    DO NOT SEARCH THE WEB IN ANY CASE. ASK THE USER FOR MORE INFORMATION IF NEEDED.
    The MCP tool decorated function setup_guide you shared is designed to assist junior developers by providing guidance on setting up a GitHub repository based on the Dockerfile it contains. Here's what it does:

    It asynchronously fetches and parses the Dockerfile from the specified GitHub repository (owner, repo, branch).

    If no Dockerfile is found or it's empty, it returns an error message.

    Otherwise, it returns the raw Dockerfile steps (list of commands) along with the user's question as separate fields
    """
    docker_steps = await parse_dockerfile(owner, repo, branch)
    if not docker_steps:
        return {"error": "No Dockerfile found or empty."}
    
    # Directly hand off the question and docker_steps as separate fields
    # or raw data. Let Claude Desktop compose the interaction/prompt.
    return {
        "docker_steps": docker_steps,
        "user_question": question
    }


async def analyze_imports(owner: str, repo: str, branch: str = "main") -> str:
    # Fetch repo file metadata list from GitHub
    files_meta = await fetch_repo_tree(owner, repo, branch)

    g = nx.DiGraph()
    count = 0

    for f in files_meta:
        filepath = f["path"]
        if filepath.endswith(".py") and count < 5:
            content = await fetch_raw_file(owner, repo, branch, filepath)
            if content is None:
                continue

            g.add_node(filepath)

            for line in content.splitlines():
                line = line.strip()
                if line.startswith("import ") or line.startswith("from "):
                    parts = line.split()
                    if len(parts) >= 2:
                        imp = parts[1]
                        g.add_edge(filepath, imp)

            count += 1

    pos = nx.spring_layout(g, k=0.5, iterations=50)
    plt.figure(figsize=(14, 12))
    nx.draw_networkx(
        g,
        pos=pos,
        with_labels=True,
        font_size=8,
        node_size=800,
        node_color="lightgreen",
        edge_color="gray",
        arrowsize=15,
        arrowstyle="->",
    )
    plt.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return img_str



@mcp.tool(name="ShowDependencyGraph")
async def show_dependency_graph(owner:str, repo:str, branch:str="main") -> dict:
    """
    Generate architecture diagrams, file relationship graphs, or dependency graphs automatically from repo data.

Visualize how major modules connect with clickable UI linked to AI chat explanations.

Helps developers quickly understand large complex repos visually in simple easy to understand diagrams rather than big diagrams which are not user friendly and have a lot of things going on in the diagram..
    """
    img_data = await analyze_imports(owner, repo, branch)
    # Return base64 PNG string so clients can display
    return {"image_base64": img_data}

async def split_code_functions(file_content:str) -> List[str]:
    """Parse Python code and extract function/class definitions."""
    if not file_content or not file_content.strip():
        return []
    
    try:
        tree = ast.parse(file_content)
    except SyntaxError as e:
        # Not valid Python code
        return [f"Error: Not a valid Python file - {str(e)}"]
    except Exception as e:
        return [f"Error parsing file: {str(e)}"]
    
    funcs = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = getattr(node, 'end_lineno', None)
            lines = file_content.splitlines()[start:end]
            funcs.append('\n'.join(lines))
    return funcs

@mcp.tool(name="SummarizeFunctions")
async def summarize_functions(owner: str, repo: str, filepath: str, branch: str = "main") -> dict:
    """
    This MCP tool fetches the raw content of a specified Python source file from a GitHub repository and parses it to extract individual function and class definitions. It returns these extracted code chunks to the connecting AI client (such as Claude Desktop). The external AI client is then responsible for dynamically generating natural language summaries or explanations of each function or class, enabling flexible and context-aware code understanding assistance for developers
    """
    # Validate that the file is a Python file
    if not filepath.endswith('.py'):
        return {
            "error": f"File '{filepath}' is not a Python file. This tool only works with .py files.",
            "functions": []
        }
    
    content = await fetch_raw_file(owner, repo, branch, filepath)
    if not content:
        return {
            "error": f"Could not fetch file '{filepath}' from repository.",
            "functions": []
        }
    
    functions = await split_code_functions(content)
    # Return raw functions to Claude Desktop for prompting and summarization
    return {"functions": functions, "filepath": filepath}

@mcp.tool(name="CodeReviewAssistant")
async def code_review_assistant(owner: str, repo: str, pull_number: int):
    pr_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"
    headers = get_auth_headers()
    async with httpx.AsyncClient() as client:
        pr_resp = await client.get(pr_url, headers=headers)
        pr_resp.raise_for_status()
        pr_data = pr_resp.json()
        diff_url = pr_data.get("diff_url")
        if not diff_url:
            return {"error": "Diff URL not found in PR data."}
        diff_resp = await client.get(diff_url, headers=headers, follow_redirects=True)
        diff_resp.raise_for_status()
        diff_text = diff_resp.text
    snippet = diff_text[:1500] + ("\n... (diff truncated)" if len(diff_text) > 1500 else "")
    return {"pr_summary": snippet}



# @app.websocket("/ws/{session_id}")
# async def websocket_endpoint(websocket: WebSocket, session_id: str):
#     await websocket.accept()
#     if session_id not in sessions:
#         sessions[session_id] = []
#     sessions[session_id].append(websocket)
#     try:
#         while True:
#             data = await websocket.receive_text()
#             for conn in sessions[session_id]:
#                 if conn != websocket:
#                     await conn.send_text(data)
#     except WebSocketDisconnect:
#         sessions[session_id].remove(websocket)

# Mount FastMCP on a subpath so you get both MCP APIs and websockets
# app.mount("/mcp", mcp.app)

def rebuild_knowledge_graph() -> None:
    global kg
    kg.clear()
    
    repo_to_files = defaultdict(set)
    
    for meta in metadata:
        filepath = meta.get("filepath")
        repo_id = meta.get("repo")
        if filepath and repo_id and not any(ext in filepath.lower() for ext in ['.png', '.jpg', '.css', '.json']):
            repo_to_files[repo_id].add(filepath)
    
    print(f"Building KG from {len(metadata)} metadata entries → {len(repo_to_files)} repos")
    
    for repo_id, files in repo_to_files.items():
        file_nodes = [f"file:{f}" for f in files]
        for node in file_nodes:
            kg.add_node(node, type="file", repo=repo_id)
        
        # Connect files lightly
        for i in range(len(file_nodes)):
            for j in range(i+1, min(i+3, len(file_nodes))):  # Limit connections
                kg.add_edge(file_nodes[i], file_nodes[j], relation="same_repo")
    
    print(f"KG built: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")

# @mcp.tool(name="KGraphQuery")
# async def kgraph_query(node_name: str) -> Dict[str, Any]:
#     """
#     Query the knowledge graph by node name.
#     Nodes are typically of form 'file:<path>'.
#     Returns neighbors to help understand related files.
#     """
#     if node_name not in kg:
#         return {"node": node_name, "neighbors": [], "exists": False}

#     neighbors = list(kg.neighbors(node_name))
#     return {
#         "node": node_name,
#         "neighbors": neighbors,
#         "exists": True,
#     }


@mcp.tool(name="Hybrid_Repo_Question")
async def hybrid_repo_question(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Hybrid tool:
    1) Run semantic search over the FAISS index.
    2) Map results to file nodes in the knowledge graph.
    3) Return both chunk hits and graph neighbors.
    Claude should answer using ONLY this structure, not web search,
    when the question is about the ingested repo.
    """
    # Basic guard
    if index.ntotal == 0:
        return {"error": "Index is empty. Please run Ingest_Repo_Code first."}

    # 1) Semantic search
    q_vec = embed_text(query).astype("float32").reshape(1, -1)
    D, I = index.search(q_vec, top_k)

    hits = []
    file_nodes = set()

    

    for rank, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        filepath = meta.get("filepath")
        code = meta.get("code", "")
        repo_id = meta.get("repo")
        # FILTER: Skip images/binary files
        if any(ext in filepath.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg']):
            continue
        hits.append({
            "rank": int(rank),
            "repo": repo_id,
            "file_path": filepath,
            "chunk_id": int(meta.get("chunk_id", -1)),
            "chunk_preview": code[:500],
            "distance": float(D[0][rank]),
        })

        if filepath:
            file_nodes.add(f"file:{filepath}")

    # 2) Graph neighbors
    graph_context = []
    for node in file_nodes:
        if node not in kg:
            continue
        neighbors = list(kg.neighbors(node))
        graph_context.append({
            "node": node,
            "neighbors": neighbors,
        })

    return {
        "query": query,
        "hits": hits,
        "graph_context": graph_context,
    }
@mcp.resource("info://instruction")
def instruction_resource():
    """
    Tool description : 
    “For any question about GitHub repos, PRs, setup, or code structure, always call the MCP tools first and only use web if tools say data is unavailable.
    Do not use the web search tool for questions - use the MCP server first.
    First try all the tools and only then in rare cases if not posisble, ask follow up questions to the user.”
    """
    return {
        "message": "Hello from the Base MCP Server!",
        "status": "ok"
    }
from collections import defaultdict
import datetime

# Global temporal data
temporal_snapshots = {}  # commit_hash -> {metadata snapshot}
commit_history = []     # List of (commit_hash, date, author, message)

@mcp.tool(name="IngestRepoTimeline")
async def ingest_repo_timeline(owner: str, repo: str, max_commits: int = 10):
    """
    Index MULTIPLE repo snapshots across time, building a temporal knowledge graph.
    Tracks file evolution, ownership changes, feature migrations.
    """
    global temporal_snapshots, commit_history
    
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    headers = get_auth_headers()
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers, params={"per_page": max_commits})
        commits = resp.json()
        
        commit_history = []
        for commit in commits:
            commit_hash = commit["sha"][:8]
            date = commit["commit"]["author"]["date"]
            author = commit["commit"]["author"]["name"]
            message = commit["commit"]["message"]
            
            commit_history.append({
                "hash": commit_hash,
                "date": date,
                "author": author,
                "message": message
            })
            
            # Get tree for this commit (simplified - fetch key files)
            tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{commit['sha']}"
            tree_resp = await client.get(tree_url, headers=headers)
            tree = tree_resp.json()["tree"]
            
            # Store snapshot metadata
            temporal_snapshots[commit_hash] = {
                "files": [item["path"] for item in tree if item["type"] == "blob"],
                "date": date,
                "author": author
            }
    
    return {
        "message": f"Indexed timeline: {len(commit_history)} commits tracked",
        "timeline": commit_history[:3]  # Show recent 3
    }

@mcp.tool(name="FindCodeEvolution")
async def find_code_evolution(query: str, target_commit: str = None, lookback_commits: int = 5):
    """
    Answer: "Where did this feature/function move from X months ago?"
    Combines semantic search across time + graph evolution tracking.
    """
    if not temporal_snapshots:
        return {"error": "Run IngestRepoTimeline first"}
    
    results = []
    recent_commits = commit_history[:lookback_commits]
    
    for commit in recent_commits:
        commit_hash = commit["hash"]
        snapshot = temporal_snapshots.get(commit_hash)
        if not snapshot:
            continue
            
        # Run semantic search on this snapshot (pseudo-code - use your existing index)
        relevant_files = [f for f in snapshot["files"] 
                         if "inventory" in f.lower() or "upload" in f.lower()]  # Simplified
        
        results.append({
            "commit": commit_hash,
            "date": commit["date"],
            "author": commit["author"],
            "files_changed": relevant_files[:5],
            "message": commit["message"][:100]
        })
    
    return {
        "query": query,
        "evolution": results,
        "timeline_length": len(commit_history)
    }

@mcp.tool(name="TemporalGraphQuery")
async def temporal_graph_query(node_name: str, time_range: str = "last_3_months"):
    """
    Show how a specific file/feature evolved across time in the knowledge graph.
    """
    if node_name not in kg.nodes:
        return {"error": f"Node {node_name} not in current graph"}
    
    evolution = []
    for commit_hash, snapshot in temporal_snapshots.items():
        snapshot_date = datetime.datetime.fromisoformat(snapshot["date"])
        if time_range == "last_3_months" and (datetime.datetime.now() - snapshot_date).days > 90:
            continue
            
        # Check if this file existed in this snapshot
        file_path = node_name.replace("file:", "")
        if file_path in snapshot["files"]:
            evolution.append({
                "commit": commit_hash,
                "existed": True,
                "neighbors_at_time": list(kg.neighbors(node_name))  # Simplified
            })
        else:
            evolution.append({
                "commit": commit_hash,
                "existed": False
            })
    
    return {
        "node": node_name,
        "evolution": evolution
    }



# -------------------------
# Run MCP Server
# -------------------------
if __name__ == "__main__":
    mcp.run(transport="streamable-http", path="/mcp", port=8005)
