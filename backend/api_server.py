#!/usr/bin/env python3
"""
Biomni Backend API Server

This FastAPI server provides a REST API interface for the Biomni A1 agent,
enabling the TypeScript frontend to interact with the biomedical AI agent.
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import uvicorn
from biomni.config import BiomniConfig
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder

config = BiomniConfig()

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv not installed, continue without it
    pass

# Add the biomni package to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from biomni.agent.a1 import A1
from biomni.config import default_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Biomni API Server",
    description="REST API for Biomni Biomedical AI Agent",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # For local development
        "http://localhost:5173",  # For Vite dev server
        "https://biomni-frontend.vercel.app",
        "https://mybioai.net",
        "https://www.mybioai.net",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "ngrok-skip-browser-warning",
        "Accept",
        "Origin",
        "X-Requested-With",
    ],
)

# Global agent instance
agent: Optional[A1] = None
active_sessions: Dict[str, Dict[str, Any]] = {}

# Prompt library service (initialized lazily when needed)
prompt_library_service = None


def get_prompt_library_service():
    """Get or initialize the prompt library service."""
    global prompt_library_service
    if prompt_library_service is None:
        try:
            # Import here to avoid dependency issues if supabase is not installed
            from backend.services.prompt_library_service import PromptLibraryService

            prompt_library_service = PromptLibraryService()
            logger.info("Prompt library service initialized")
        except ImportError as e:
            logger.warning(f"Prompt library service not available: {e}")
            return None
        except ValueError as e:
            logger.warning(f"Prompt library service configuration error: {e}")
            return None
    return prompt_library_service


# Pydantic models for API requests/responses
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the agent")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    use_tool_retriever: Optional[bool] = Field(True, description="Whether to use tool retriever")
    self_critic: Optional[bool] = Field(False, description="Whether to enable self-critic mode")
    model: Optional[str] = Field(None, description="LLM model to use: 'claude-sonnet-4-20250514' or 'gpt-5'")
    source: Optional[str] = Field(None, description="LLM source: 'Anthropic' or 'OpenAI'")


class ChatResponse(BaseModel):
    session_id: str = Field(..., description="Session ID")
    response: str = Field(..., description="Agent response")
    log: List[str] = Field(..., description="Execution log")
    timestamp: datetime = Field(..., description="Response timestamp")
    status: str = Field(..., description="Response status")


class StreamResponse(BaseModel):
    session_id: str = Field(..., description="Session ID")
    output: str = Field(..., description="Stream output")
    step: int = Field(..., description="Step number")
    timestamp: datetime = Field(..., description="Timestamp")
    is_complete: bool = Field(..., description="Whether stream is complete")


class ConfigurationRequest(BaseModel):
    llm: Optional[str] = Field(None, description="LLM model to use (e.g., gpt-5, gpt-4, claude-sonnet-4-20250514)")
    source: Optional[str] = Field(None, description="LLM source provider (OpenAI, Anthropic, Gemini, etc.)")
    use_tool_retriever: Optional[bool] = Field(None, description="Whether to use tool retriever")
    timeout_seconds: Optional[int] = Field(None, description="Timeout for code execution")
    base_url: Optional[str] = Field(None, description="Base URL for custom model")
    api_key: Optional[str] = Field(None, description="API key for the LLM")


class ToolInfo(BaseModel):
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    module: str = Field(..., description="Tool module")
    parameters: Dict[str, Any] = Field(..., description="Tool parameters")


class DataLakeInfo(BaseModel):
    name: str = Field(..., description="Data lake item name")
    description: str = Field(..., description="Data lake item description")
    path: str = Field(..., description="Data lake item path")


class SoftwareInfo(BaseModel):
    name: str = Field(..., description="Software name")
    description: str = Field(..., description="Software description")


class SystemInfo(BaseModel):
    tools: List[ToolInfo] = Field(..., description="Available tools")
    data_lake: List[DataLakeInfo] = Field(..., description="Data lake items")
    software: List[SoftwareInfo] = Field(..., description="Available software")
    configuration: Dict[str, Any] = Field(..., description="Current configuration")


# ==================== PROMPT LIBRARY MODELS ====================


class PromptVariable(BaseModel):
    name: str = Field(..., description="Variable name")
    var_type: str = Field(..., description="Variable type (string, number, boolean, array)", alias="type")
    description: str = Field(..., description="Variable description")
    default: Optional[Any] = Field(None, description="Default value")
    required: bool = Field(True, description="Whether this variable is required")

    class Config:
        populate_by_name = True  # Allows using both 'var_type' and 'type'


class ModelConfig(BaseModel):
    model: str = Field(
        "claude-sonnet-4-20250514", description="LLM model name (e.g., gpt-5, gpt-4, claude-sonnet-4-20250514)"
    )
    temperature: float = Field(0.7, description="Temperature setting")
    max_tokens: int = Field(8192, description="Maximum tokens")
    source: Optional[str] = Field("Anthropic", description="Model source provider (OpenAI, Anthropic, Gemini, etc.)")
    top_p: Optional[float] = Field(None, description="Top-p sampling")
    frequency_penalty: Optional[float] = Field(None, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(None, description="Presence penalty")


class ToolBindings(BaseModel):
    enabled_modules: List[str] = Field(default_factory=list, description="Enabled tool modules")
    specific_tools: List[str] = Field(default_factory=list, description="Specific tools to use")
    use_tool_retriever: bool = Field(True, description="Whether to use tool retriever")


class OutputTemplate(BaseModel):
    format: str = Field("markdown", description="Output format (markdown, json, csv, table)")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="JSON schema for output")
    field_mapping: Dict[str, str] = Field(default_factory=dict, description="Field name mappings")


class CreatePromptRequest(BaseModel):
    model_config = {"populate_by_name": True}  # Allows using both 'llm_config' and 'model_config'

    title: str = Field(..., description="Prompt title")
    prompt_template: str = Field(..., description="Prompt template with {variable} placeholders")
    category: str = Field(
        ...,
        description="Category: genomics, protein_analysis, drug_discovery, literature_review, etc.",
    )
    description: Optional[str] = Field(None, description="Prompt description")
    tags: List[str] = Field(default_factory=list, description="Tags for discovery")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    variables: List[PromptVariable] = Field(default_factory=list, description="Variable definitions")
    llm_config: Optional[ModelConfig] = Field(None, description="Model configuration", alias="model_config")
    tool_bindings: Optional[ToolBindings] = Field(None, description="Tool bindings")
    output_template: Optional[OutputTemplate] = Field(None, description="Output template")


class UpdatePromptRequest(BaseModel):
    title: Optional[str] = Field(None, description="Prompt title")
    prompt_template: Optional[str] = Field(None, description="Prompt template")
    category: Optional[str] = Field(None, description="Category")
    description: Optional[str] = Field(None, description="Description")
    tags: Optional[List[str]] = Field(None, description="Tags")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    variables: Optional[List[PromptVariable]] = Field(None, description="Variables")
    llm_config: Optional[ModelConfig] = Field(None, description="Model configuration", alias="model_config")
    tool_bindings: Optional[ToolBindings] = Field(None, description="Tool bindings")
    output_template: Optional[OutputTemplate] = Field(None, description="Output template")
    is_active: Optional[bool] = Field(None, description="Active status")
    create_version: bool = Field(False, description="Create new version instead of updating")


class ExecutePromptRequest(BaseModel):
    model_config = {"populate_by_name": True}

    prompt_id: str = Field(..., description="Prompt ID to execute")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Variable values for rendering")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")
    override_llm_config: Optional[ModelConfig] = Field(
        None, description="Override model configuration", alias="override_model_config"
    )


class PromptResponse(BaseModel):
    model_config = {"populate_by_name": True}

    id: str = Field(..., description="Prompt ID")
    title: str = Field(..., description="Prompt title")
    description: Optional[str] = Field(None, description="Description")
    category: str = Field(..., description="Category")
    tags: List[str] = Field(default_factory=list, description="Tags")
    prompt_template: str = Field(..., description="Prompt template")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    variables: List[Dict[str, Any]] = Field(default_factory=list, description="Variables")
    llm_config: Dict[str, Any] = Field(default_factory=dict, description="Model config", alias="model_config")
    tool_bindings: Dict[str, Any] = Field(default_factory=dict, description="Tool bindings")
    output_template: Dict[str, Any] = Field(default_factory=dict, description="Output template")
    version: int = Field(1, description="Version number")
    is_predefined: bool = Field(False, description="Is predefined system prompt")
    usage_count: int = Field(0, description="Usage count")
    created_by: Optional[str] = Field(None, description="Creator user ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class ExecutePromptResponse(BaseModel):
    execution_id: str = Field(..., description="Execution ID")
    prompt_id: str = Field(..., description="Prompt ID")
    rendered_prompt: str = Field(..., description="Rendered prompt with variables")
    response: str = Field(..., description="Agent response")
    log: List[str] = Field(default_factory=list, description="Execution log")
    tools_used: List[str] = Field(default_factory=list, description="Tools used")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(..., description="Execution timestamp")


def initialize_agent(config: Optional[ConfigurationRequest] = None) -> A1:
    """Initialize the Biomni agent with the given configuration."""
    global agent

    if agent is not None:
        logger.info("Agent already initialized, reinitializing with new config")

    # Use provided config or defaults
    if config:
        agent = A1(
            llm=config.llm,
            source=config.source,
            use_tool_retriever=config.use_tool_retriever,
            timeout_seconds=config.timeout_seconds,
            base_url=config.base_url,
            api_key=config.api_key,
        )
    else:
        agent = A1()

    logger.info("Biomni agent initialized successfully")
    return agent


async def initialize_async_agent(config: Optional[ConfigurationRequest] = None) -> A1:
    """Initialize the async Biomni agent with the given configuration."""
    global agent

    if agent is not None:
        logger.info("Agent already initialized, reinitializing with new config")

    # Use provided config or defaults
    if config:
        agent = A1(
            llm=config.llm,
            source=config.source,
            use_tool_retriever=config.use_tool_retriever,
            timeout_seconds=config.timeout_seconds,
            base_url=config.base_url,
            api_key=config.api_key,
        )
    else:
        agent = A1()

    await agent.async_init()
    await agent.configure_async()
    logger.info("Async Biomni agent initialized successfully")
    return agent


@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup."""
    global agent
    try:
        # Initialize the agent with async capabilities
        agent = initialize_agent()
        await agent.async_init()
        await agent.configure_async()
        logger.info("Biomni API server started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Biomni API Server", "version": "1.0.0", "status": "running", "docs": "/docs"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    current_model = None
    if agent is not None and hasattr(agent.llm, "model"):
        current_model = agent.llm.model

    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
        "async_agent_initialized": agent is not None and hasattr(agent, "async_llm"),
        "current_model": current_model,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/models")
async def get_available_models():
    """Get list of available models that users can select."""
    return {
        "models": [
            {
                "id": "claude-sonnet-4-20250514",
                "name": "Claude Sonnet 4",
                "source": "Anthropic",
                "description": "Latest Claude Sonnet model - best for complex biomedical reasoning",
                "is_default": True,
            },
            {
                "id": "gpt-5",
                "name": "GPT-5",
                "source": "OpenAI",
                "description": "OpenAI's GPT-5 model - powerful general-purpose AI",
                "is_default": False,
            },
        ],
        "default_model": "claude-sonnet-4-20250514",
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the agent and get a response using async processing."""
    global agent

    if agent is None or not hasattr(agent, "async_llm"):
        raise HTTPException(status_code=500, detail="Async agent not initialized")

    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid4())

        # Update agent configuration if needed
        if request.use_tool_retriever is not None:
            agent.use_tool_retriever = request.use_tool_retriever

        # Handle dynamic model selection
        if request.model or request.source:
            # User wants to use a different model for this request
            model = (
                request.model
                if request.model
                else agent.llm.model
                if hasattr(agent.llm, "model")
                else "claude-sonnet-4-20250514"
            )
            source = request.source if request.source else ("OpenAI" if model.startswith("gpt-") else "Anthropic")

            # Create a temporary agent with the requested model
            from biomni.llm import get_async_llm

            temp_llm = await get_async_llm(model=model, source=source, temperature=agent.llm.temperature)

            # Temporarily swap the LLM
            original_llm = agent.llm
            original_async_llm = agent.async_llm
            agent.llm = temp_llm
            agent.async_llm = temp_llm

            try:
                # Execute with the temporary model
                log, response = await agent.go_async(request.message)
            finally:
                # Restore original LLM
                agent.llm = original_llm
                agent.async_llm = original_async_llm
        else:
            # Use default agent configuration
            log, response = await agent.go_async(request.message)

        # Store session info
        if session_id not in active_sessions:
            active_sessions[session_id] = {"created_at": datetime.now(), "message_count": 0}

        active_sessions[session_id]["message_count"] += 1

        return ChatResponse(
            session_id=session_id, response=response, log=log, timestamp=datetime.now(), status="success"
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/api/chat/stream/{session_id}")
async def chat_stream(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming chat responses."""
    global agent

    if agent is None or not hasattr(agent, "async_llm"):
        await websocket.close(code=1008, reason="Async agent not initialized")
        return

    await websocket.accept()
    logger.info(f"WebSocket connected for session {session_id}")

    try:
        while True:
            try:
                # Receive message from client with timeout to prevent hanging
                data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                message_data = json.loads(data)
                message = message_data.get("message", "")

                if not message:
                    continue

                # Update agent configuration if provided
                use_tool_retriever = message_data.get("use_tool_retriever", True)
                if use_tool_retriever is not None:
                    agent.use_tool_retriever = use_tool_retriever

                # Handle dynamic model selection for streaming
                model = message_data.get("model")
                source = message_data.get("source")

                if model or source:
                    # User wants to use a different model
                    selected_model = (
                        model
                        if model
                        else agent.llm.model
                        if hasattr(agent.llm, "model")
                        else "claude-sonnet-4-20250514"
                    )
                    selected_source = (
                        source if source else ("OpenAI" if selected_model.startswith("gpt-") else "Anthropic")
                    )

                    # Create temporary LLM
                    from biomni.llm import get_async_llm

                    temp_llm = await get_async_llm(
                        model=selected_model, source=selected_source, temperature=agent.llm.temperature
                    )

                    # Swap LLMs temporarily
                    original_llm = agent.llm
                    original_async_llm = agent.async_llm
                    agent.llm = temp_llm
                    agent.async_llm = temp_llm

                    # Stream the response using async generator with better error handling
                    step_count = 0
                    try:
                        async for step in agent.go_stream_async(message):
                            try:
                                step_count += 1
                                response_data = StreamResponse(
                                    session_id=session_id,
                                    output=step.get("output", ""),
                                    step=step_count,
                                    timestamp=datetime.now(),
                                    is_complete=False,
                                )

                                # Send with timeout to prevent blocking
                                await asyncio.wait_for(websocket.send_text(response_data.json()), timeout=10.0)

                                # Small delay to prevent overwhelming the client
                                await asyncio.sleep(0.01)
                            except TimeoutError:
                                logger.warning(f"Send timeout for session {session_id}")
                                break
                            except Exception as send_error:
                                logger.error(f"Error sending step {step_count}: {send_error}")
                                break

                        # Send completion signal
                        try:
                            completion_data = StreamResponse(
                                session_id=session_id,
                                output="",
                                step=step_count + 1,
                                timestamp=datetime.now(),
                                is_complete=True,
                            )
                            await asyncio.wait_for(websocket.send_text(completion_data.json()), timeout=10.0)
                        except Exception as completion_error:
                            logger.error(f"Error sending completion: {completion_error}")
                    finally:
                        # Restore original LLM if it was swapped
                        if model or source:
                            agent.llm = original_llm
                            agent.async_llm = original_async_llm
                else:
                    # Use default model - no swapping needed
                    step_count = 0
                    try:
                        async for step in agent.go_stream_async(message):
                            try:
                                step_count += 1
                                response_data = StreamResponse(
                                    session_id=session_id,
                                    output=step.get("output", ""),
                                    step=step_count,
                                    timestamp=datetime.now(),
                                    is_complete=False,
                                )

                                # Send with timeout to prevent blocking
                                await asyncio.wait_for(websocket.send_text(response_data.json()), timeout=10.0)

                                # Small delay to prevent overwhelming the client
                                await asyncio.sleep(0.01)
                            except TimeoutError:
                                logger.warning(f"Send timeout for session {session_id}")
                                break
                            except Exception as send_error:
                                logger.error(f"Error sending step {step_count}: {send_error}")
                                break

                        # Send completion signal
                        try:
                            completion_data = StreamResponse(
                                session_id=session_id,
                                output="",
                                step=step_count + 1,
                                timestamp=datetime.now(),
                                is_complete=True,
                            )
                            await asyncio.wait_for(websocket.send_text(completion_data.json()), timeout=10.0)
                        except Exception as completion_error:
                            logger.error(f"Error sending completion: {completion_error}")

                    except Exception as stream_error:
                        logger.error(f"Error in agent streaming: {stream_error}")
                        # Send error message to client
                        try:
                            error_response = StreamResponse(
                                session_id=session_id,
                                output=f"Error: {str(stream_error)}",
                                step=step_count + 1,
                                timestamp=datetime.now(),
                                is_complete=True,
                            )
                            await websocket.send_text(error_response.json())
                        except Exception:
                            pass

            except TimeoutError:
                # Timeout waiting for message - this is normal, continue the loop
                continue
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received for session {session_id}")
                continue

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket stream: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"WebSocket cleanup completed for session {session_id}")


@app.post("/api/configure")
async def configure_agent(request: ConfigurationRequest):
    """Configure the agent with new settings."""
    global agent

    try:
        # Configure the agent with async capabilities
        agent = initialize_agent(request)
        await agent.async_init()
        await agent.configure_async()
        return {"message": "Agent configured successfully", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error configuring agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/system/info", response_model=SystemInfo)
async def get_system_info():
    """Get information about available tools, data lake, and software."""
    global agent

    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Get tools
        tools = []
        for module_name, module_tools in agent.module2api.items():
            for tool in module_tools:
                if tool.get("name") != "run_python_repl":  # Skip internal tool
                    tools.append(
                        ToolInfo(
                            name=tool.get("name", ""),
                            description=tool.get("description", ""),
                            module=module_name,
                            parameters=tool.get("parameters", {}),
                        )
                    )

        # Get data lake items
        data_lake_path = os.path.join(agent.path, "data_lake")
        data_lake_items = []
        if os.path.exists(data_lake_path):
            for item in os.listdir(data_lake_path):
                description = agent.data_lake_dict.get(item, f"Data lake item: {item}")
                data_lake_items.append(
                    DataLakeInfo(name=item, description=description, path=os.path.join(data_lake_path, item))
                )

        # Get software
        software = []
        for lib_name, lib_desc in agent.library_content_dict.items():
            software.append(SoftwareInfo(name=lib_name, description=lib_desc))

        # Get current configuration
        config = {
            "llm": agent.llm.model,
            "source": getattr(agent, "source", "Unknown"),
            "use_tool_retriever": agent.use_tool_retriever,
            "temperature": agent.llm.temperature,
            "timeout_seconds": agent.timeout_seconds,
            "path": agent.path,
        }

        return SystemInfo(tools=tools, data_lake=data_lake_items, software=software, configuration=config)

    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions")
async def get_sessions():
    """Get information about active sessions."""
    return {
        "sessions": active_sessions,
        "total_sessions": len(active_sessions),
        "timestamp": datetime.now().isoformat(),
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"message": f"Session {session_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/tools/add")
async def add_custom_tool(tool_data: Dict[str, Any]):
    """Add a custom tool to the agent."""
    global agent

    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # This would need to be implemented based on how you want to add tools
        # For now, we'll return a placeholder
        return {"message": "Custom tool addition not yet implemented", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error adding custom tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tools/custom")
async def get_custom_tools():
    """Get list of custom tools."""
    global agent

    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        custom_tools = agent.list_custom_tools()
        return {"custom_tools": custom_tools, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error getting custom tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/upload")
async def upload_data_file(file: UploadFile = File(...), description: str = Form(...), name: str = Form(...)):
    """Upload a custom data file to the data lake."""
    global agent

    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Validate file size (100MB limit)
        if file.size and file.size > 100 * 1024 * 1024:  # 100MB
            raise HTTPException(status_code=400, detail="File size exceeds 100MB limit")

        # Validate file type
        allowed_extensions = {
            ".csv",
            ".json",
            ".txt",
            ".xlsx",
            ".xls",
            ".tsv",
            ".fasta",
            ".fastq",
            ".bam",
            ".sam",
            ".vcf",
            ".bed",
            ".gtf",
            ".gff",
            ".fa",
            ".fna",
            ".faa",
            ".dat",
            ".xml",
            ".yaml",
            ".yml",
        }

        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported. Allowed: {', '.join(allowed_extensions)}",
            )

        # Create data lake directory if it doesn't exist
        data_lake_path = os.path.join(agent.path, "data_lake")
        os.makedirs(data_lake_path, exist_ok=True)

        # Generate unique filename to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{name}"
        file_path = os.path.join(data_lake_path, safe_filename)

        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Add to agent's data lake dictionary
        if not hasattr(agent, "data_lake_dict"):
            agent.data_lake_dict = {}

        agent.data_lake_dict[safe_filename] = description

        logger.info(f"File uploaded successfully: {safe_filename}")

        return {
            "message": "File uploaded successfully",
            "filename": safe_filename,
            "path": file_path,
            "size": file.size,
            "description": description,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/data/{filename}")
async def delete_data_file(filename: str):
    """Delete a custom data file from the data lake."""
    global agent

    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        data_lake_path = os.path.join(agent.path, "data_lake")
        file_path = os.path.join(data_lake_path, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Remove file
        os.remove(file_path)

        # Remove from agent's data lake dictionary
        if hasattr(agent, "data_lake_dict") and filename in agent.data_lake_dict:
            del agent.data_lake_dict[filename]

        logger.info(f"File deleted successfully: {filename}")

        return {"message": "File deleted successfully", "filename": filename, "timestamp": datetime.now().isoformat()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== PROMPT LIBRARY ENDPOINTS ====================


@app.post("/api/prompts", response_model=PromptResponse)
async def create_prompt(request: CreatePromptRequest, user_id: str = "anonymous"):
    """Create a new prompt template."""
    service = get_prompt_library_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Prompt library service not available. Please configure Supabase.")

    try:
        # Convert Pydantic models to dicts
        llm_config = request.llm_config.dict() if request.llm_config else None
        tool_bindings = request.tool_bindings.dict() if request.tool_bindings else None
        output_template = request.output_template.dict() if request.output_template else None
        variables = [v.dict() for v in request.variables] if request.variables else []

        prompt = await service.create_prompt(
            title=request.title,
            prompt_template=request.prompt_template,
            category=request.category,
            created_by=user_id,
            description=request.description,
            tags=request.tags,
            system_prompt=request.system_prompt,
            variables=variables,
            model_config=llm_config,
            tool_bindings=tool_bindings,
            output_template=output_template,
        )

        return prompt

    except Exception as e:
        logger.error(f"Error creating prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prompts", response_model=List[PromptResponse])
async def list_prompts(
    user_id: str = "anonymous",
    category: Optional[str] = None,
    tags: Optional[str] = None,  # Comma-separated tags
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """List user's prompts with filters."""
    service = get_prompt_library_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Prompt library service not available. Please configure Supabase.")

    try:
        tag_list = tags.split(",") if tags else None

        prompts = await service.list_prompts(
            user_id=user_id,
            category=category,
            tags=tag_list,
            search_query=search,
            limit=limit,
            offset=offset,
        )

        return prompts

    except Exception as e:
        logger.error(f"Error listing prompts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prompts/{prompt_id}", response_model=PromptResponse)
async def get_prompt(prompt_id: str):
    """Get a specific prompt by ID."""
    service = get_prompt_library_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Prompt library service not available. Please configure Supabase.")

    try:
        prompt = await service.get_prompt_by_id(prompt_id)
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")

        return prompt

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/prompts/{prompt_id}", response_model=PromptResponse)
async def update_prompt(prompt_id: str, request: UpdatePromptRequest, user_id: str = "anonymous"):
    """Update a prompt template."""
    service = get_prompt_library_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Prompt library service not available. Please configure Supabase.")

    try:
        # Build update fields from request
        update_fields = {}
        if request.title is not None:
            update_fields["title"] = request.title
        if request.prompt_template is not None:
            update_fields["prompt_template"] = request.prompt_template
        if request.category is not None:
            update_fields["category"] = request.category
        if request.description is not None:
            update_fields["description"] = request.description
        if request.tags is not None:
            update_fields["tags"] = request.tags
        if request.system_prompt is not None:
            update_fields["system_prompt"] = request.system_prompt
        if request.variables is not None:
            update_fields["variables"] = [v.dict() for v in request.variables]
        if request.llm_config is not None:
            update_fields["model_config"] = request.llm_config.dict()
        if request.tool_bindings is not None:
            update_fields["tool_bindings"] = request.tool_bindings.dict()
        if request.output_template is not None:
            update_fields["output_template"] = request.output_template.dict()
        if request.is_active is not None:
            update_fields["is_active"] = request.is_active

        prompt = await service.update_prompt(
            prompt_id=prompt_id,
            updated_by=user_id,
            create_version=request.create_version,
            **update_fields,
        )

        return prompt

    except Exception as e:
        logger.error(f"Error updating prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str, hard_delete: bool = False):
    """Delete a prompt template."""
    service = get_prompt_library_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Prompt library service not available. Please configure Supabase.")

    try:
        success = await service.delete_prompt(prompt_id, soft_delete=not hard_delete)
        if not success:
            raise HTTPException(status_code=404, detail="Prompt not found")

        return {"message": "Prompt deleted successfully", "prompt_id": prompt_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/prompts/execute", response_model=ExecutePromptResponse)
async def execute_prompt(request: ExecutePromptRequest, user_id: str = "anonymous"):
    """Execute a prompt template with the agent."""
    global agent
    service = get_prompt_library_service()

    if agent is None or not hasattr(agent, "async_llm"):
        raise HTTPException(status_code=500, detail="Async agent not initialized")

    if service is None:
        raise HTTPException(status_code=503, detail="Prompt library service not available. Please configure Supabase.")

    try:
        # Get the prompt
        prompt = await service.get_prompt_by_id(request.prompt_id)
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")

        # Render the prompt with variables
        rendered_prompt = await service.render_prompt(request.prompt_id, request.variables)

        # Get model config (use override if provided)
        model_config = (
            request.override_model_config.dict() if request.override_model_config else prompt.get("model_config", {})
        )

        # Configure agent if needed
        tool_bindings = prompt.get("tool_bindings", {})
        if tool_bindings.get("use_tool_retriever") is not None:
            agent.use_tool_retriever = tool_bindings["use_tool_retriever"]

        # Log the execution start
        execution_id = await service.execute_prompt(
            prompt_id=request.prompt_id,
            user_id=user_id,
            variables=request.variables,
            session_id=request.session_id,
            override_model_config=model_config,
        )

        # Execute the agent
        start_time = datetime.now()
        log, response = await agent.go_async(rendered_prompt)
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

        # Extract tools used from log
        tools_used = []
        for log_entry in log:
            if "Tool:" in log_entry or "Function:" in log_entry:
                # Simple extraction - could be improved
                tools_used.append(log_entry)

        # Update execution with results
        await service.update_execution_result(
            execution_id=execution_id,
            response=response,
            execution_log=log,
            tools_used=tools_used,
            processing_time_ms=processing_time,
        )

        return ExecutePromptResponse(
            execution_id=execution_id,
            prompt_id=request.prompt_id,
            rendered_prompt=rendered_prompt,
            response=response,
            log=log,
            tools_used=tools_used,
            processing_time_ms=processing_time,
            timestamp=datetime.now(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing prompt: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prompts/{prompt_id}/versions", response_model=List[PromptResponse])
async def get_prompt_versions(prompt_id: str):
    """Get all versions of a prompt."""
    service = get_prompt_library_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Prompt library service not available. Please configure Supabase.")

    try:
        versions = await service.get_prompt_versions(prompt_id)
        return versions

    except Exception as e:
        logger.error(f"Error getting prompt versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prompts/{prompt_id}/analytics")
async def get_prompt_analytics(prompt_id: str):
    """Get analytics for a specific prompt."""
    service = get_prompt_library_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Prompt library service not available. Please configure Supabase.")

    try:
        analytics = await service.get_prompt_analytics(prompt_id)
        return analytics

    except Exception as e:
        logger.error(f"Error getting prompt analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/prompts/{prompt_id}/favorite")
async def add_favorite(prompt_id: str, user_id: str = "anonymous"):
    """Add a prompt to favorites."""
    service = get_prompt_library_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Prompt library service not available. Please configure Supabase.")

    try:
        favorite = await service.add_favorite(user_id, prompt_id)
        return {"message": "Prompt added to favorites", "favorite": favorite}

    except Exception as e:
        logger.error(f"Error adding favorite: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/prompts/{prompt_id}/favorite")
async def remove_favorite(prompt_id: str, user_id: str = "anonymous"):
    """Remove a prompt from favorites."""
    service = get_prompt_library_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Prompt library service not available. Please configure Supabase.")

    try:
        success = await service.remove_favorite(user_id, prompt_id)
        if not success:
            raise HTTPException(status_code=404, detail="Favorite not found")

        return {"message": "Prompt removed from favorites", "prompt_id": prompt_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing favorite: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/user/prompts/favorites", response_model=List[PromptResponse])
async def get_user_favorites(user_id: str = "anonymous"):
    """Get user's favorite prompts."""
    service = get_prompt_library_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Prompt library service not available. Please configure Supabase.")

    try:
        favorites = await service.get_user_favorites(user_id)
        return favorites

    except Exception as e:
        logger.error(f"Error getting user favorites: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/user/prompts/stats")
async def get_user_prompt_stats(user_id: str = "anonymous"):
    """Get user's prompt statistics."""
    service = get_prompt_library_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Prompt library service not available. Please configure Supabase.")

    try:
        stats = await service.get_user_prompt_stats(user_id)
        return stats

    except Exception as e:
        logger.error(f"Error getting user prompt stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the server
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
