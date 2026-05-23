import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services.loom_service.substrate.neural_viewer import NeuralViewer
from backend.services.loom_service.orchestration.loomServer_Service import LoomServerService
from backend.services.LLM_service.model_Factory import ModelFactory
from backend.config.envConfig import setup_logger, log_service
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = setup_logger("AIServerRoute")
router = APIRouter(prefix="/ai", tags=["AI"])

# Service Singletons
_loom_service = LoomServerService()

class AskRequest(BaseModel):
    master_path: str
    query: str
    depth: int = 4
    width: int = 5 # Increased for better RAG performance
    provider: str = None # Override default provider

@router.post("/ask")
async def ask_loom(request: AskRequest):
    """
    SECTION 9: DocLoom-Powered RAG (Research Grade)
    Uses the Loom graph activation to provide deep context to the LLM.
    """
    log_service(logger, f"AI Query: {request.query}", "info")
    
    try:
        # 1. Retrieval: Activate Loom Graph
        viewer = NeuralViewer(request.master_path)
        if not viewer.data:
            raise HTTPException(status_code=404, detail="Loom graph not found")
            
        activation_result = _loom_service.activate_loom(
            viewer, 
            request.query, 
            depth=request.depth, 
            width=request.width
        )
        
        if not activation_result["success"]:
            raise HTTPException(status_code=500, detail=activation_result["reason"])

        # 2. Semantic Context Assembly (Pass only actual knowledge, not structural nodes)
        semantic_types = {"shard", "shard_heading", "table", "image"}
        context_units = [u for u in activation_result["activation_path"] if u["type"] in semantic_types]
        
        context_text = "\n\n".join([f"SOURCE [{u['id']}]: {u['content']}" for u in context_units])
        
        if not context_text:
            context_text = "No direct semantic matches found in the Loom. Try a broader search."
        
        # 3. LLM Generation
        model = ModelFactory.get_model(provider=request.provider)
        
        prompt = ChatPromptTemplate.from_template("""
        You are DocLoom AI, a high-precision research assistant. 
        You have access to a specialized "Knowledge Loom" which provides semantically connected context.
        
        USER QUERY: {query}
        
        LOOM CONTEXT:
        {context}
        
        INSTRUCTIONS:
        1. Answer the query using ONLY the provided context.
        2. If the context doesn't contain the answer, state that you don't know based on the current Loom.
        3. Cite the Node IDs when mentioning specific facts.
        4. Maintain a professional, research-grade tone.
        
        ANSWER:
        """)
        
        chain = prompt | model | StrOutputParser()
        
        answer = chain.invoke({
            "query": request.query,
            "context": context_text
        })
        
        return {
            "success": True,
            "answer": answer,
            "citations": [u['id'] for u in context_units],
            "entry_hub": activation_result["entry_point"]
        }
        
    except Exception as e:
        log_service(logger, f"Error in ask_loom: {str(e)}", "error")
        raise HTTPException(status_code=500, detail=str(e))
