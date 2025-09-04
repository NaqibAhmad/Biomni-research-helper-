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


# Pydantic models for API requests/responses
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the agent")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    use_tool_retriever: Optional[bool] = Field(True, description="Whether to use tool retriever")
    self_critic: Optional[bool] = Field(False, description="Whether to enable self-critic mode")


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
    llm: Optional[str] = Field(None, description="LLM model to use")
    source: Optional[str] = Field(None, description="LLM source provider")
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
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
        "async_agent_initialized": agent is not None and hasattr(agent, "async_llm"),
        "timestamp": datetime.now().isoformat(),
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

        # Store session info
        if session_id not in active_sessions:
            active_sessions[session_id] = {"created_at": datetime.now(), "message_count": 0}

        active_sessions[session_id]["message_count"] += 1

        # Execute the agent asynchronously
        log, response = await agent.go_async(request.message)

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

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            message = message_data.get("message", "")

            if not message:
                continue

            # Update agent configuration if provided
            use_tool_retriever = message_data.get("use_tool_retriever", True)
            if use_tool_retriever is not None:
                agent.use_tool_retriever = use_tool_retriever

            # Stream the response using async generator
            step_count = 0
            async for step in agent.go_stream_async(message):
                step_count += 1
                response_data = StreamResponse(
                    session_id=session_id,
                    output=step["output"],
                    step=step_count,
                    timestamp=datetime.now(),
                    is_complete=False,
                )

                await websocket.send_text(response_data.json())

            # Send completion signal
            completion_data = StreamResponse(
                session_id=session_id, output="", step=step_count + 1, timestamp=datetime.now(), is_complete=True
            )

            await websocket.send_text(completion_data.json())

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket stream: {e}")
        try:
            await websocket.send_text(
                json.dumps({"error": str(e), "session_id": session_id, "timestamp": datetime.now().isoformat()})
            )
        except:
            pass


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


if __name__ == "__main__":
    # Run the server
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
