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
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder

config = BiomniConfig()

# Configure logging first (before loading .env so we can log about it)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    # Try loading from backend directory first, then project root
    backend_dir = Path(__file__).parent
    project_root = backend_dir.parent

    # Load .env from multiple possible locations
    env_loaded = False
    for env_path in [backend_dir / ".env", project_root / ".env", Path.cwd() / ".env"]:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            logger.info(f"Loaded environment variables from: {env_path}")
            env_loaded = True
            break

    if not env_loaded:
        # Try default location (current directory)
        load_dotenv(override=True)
        logger.info("Attempted to load .env from default location")

except ImportError:
    # dotenv not installed, continue without it
    logger.warning("python-dotenv not installed. Environment variables must be set manually.")
    pass

# Add the biomni package to the Python path
backend_dir = Path(__file__).parent
project_root = backend_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_dir))  # Also add backend directory for middleware imports

from biomni.agent.a1 import A1
from biomni.config import default_config

# Import authentication middleware
try:
    # Try absolute import first (from project root)
    from middleware.auth import get_current_user, get_optional_user

    logger.info("Authentication middleware loaded successfully")
except ImportError as e:
    # Try relative import as fallback (from backend directory)
    try:
        from middleware.auth import get_current_user, get_optional_user

        logger.info("Authentication middleware loaded successfully (relative import)")
    except ImportError as e2:
        # Try direct import from the file
        try:
            import importlib.util

            auth_path = backend_dir / "middleware" / "auth.py"
            if auth_path.exists():
                spec = importlib.util.spec_from_file_location("auth_middleware", auth_path)
                auth_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(auth_module)
                get_current_user = auth_module.get_current_user
                get_optional_user = auth_module.get_optional_user
                logger.info("Authentication middleware loaded successfully (direct file import)")
            else:
                raise ImportError(f"Auth file not found at {auth_path}")
        except Exception as e3:
            # Fallback if middleware not found (development mode)
            logger.warning(f"Authentication middleware not found. Absolute import error: {e}")
            logger.warning(f"Relative import error: {e2}")
            logger.warning(f"Direct file import error: {e3}")
            logger.warning("Using anonymous user fallback. Authentication will not work.")

    async def get_optional_user() -> str:
        return "anonymous"

    async def get_current_user() -> str:
        return "anonymous"


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
        "http://127.0.0.1:3000",  # Alternative localhost
        "http://127.0.0.1:5173",  # Alternative localhost for Vite
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

# Services (initialized lazily when needed)
prompt_library_service = None
feedback_service = None
chat_service = None
file_storage_service = None
user_settings_service = None
streaming_service = None


def get_prompt_library_service():
    """Get or initialize the prompt library service."""
    global prompt_library_service
    if prompt_library_service is None:
        try:
            # Import here to avoid dependency issues if supabase is not installed
            from services.prompt_library_service import PromptLibraryService

            prompt_library_service = PromptLibraryService()
            logger.info("Prompt library service initialized")
        except ImportError as e:
            logger.warning(f"Prompt library service not available: {e}")
            return None
        except ValueError as e:
            logger.warning(f"Prompt library service configuration error: {e}")
            return None
    return prompt_library_service


def get_feedback_service():
    """Get or initialize the feedback service."""
    global feedback_service
    if feedback_service is None:
        try:
            from services.feedback_service import FeedbackService

            feedback_service = FeedbackService()
            logger.info("Feedback service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize feedback service: {e}")
            raise HTTPException(status_code=503, detail="Feedback service not available. Please configure Supabase.")
    return feedback_service


def get_chat_service():
    """Get or initialize the chat service."""
    global chat_service
    if chat_service is None:
        try:
            from services.chat_service import ChatService

            chat_service = ChatService()
            logger.info("[SERVICE] Chat service initialized successfully")
        except Exception as e:
            logger.error(f"[SERVICE] Chat service initialization failed: {e}", exc_info=True)
            return None
    else:
        logger.debug("[SERVICE] Using existing chat service instance")
    return chat_service


def get_file_storage_service():
    """Get or initialize the file storage service."""
    global file_storage_service
    if file_storage_service is None:
        try:
            from services.file_storage_service import FileStorageService

            file_storage_service = FileStorageService()
            logger.info("File storage service initialized")
        except Exception as e:
            logger.warning(f"File storage service not available: {e}")
            return None
    return file_storage_service


def get_user_settings_service():
    """Get or initialize the user settings service."""
    global user_settings_service
    if user_settings_service is None:
        try:
            from services.user_settings_service import UserSettingsService

            user_settings_service = UserSettingsService()
            logger.info("User settings service initialized")
        except Exception as e:
            logger.warning(f"User settings service not available: {e}")
            return None
    return user_settings_service


def get_streaming_service():
    """Get or initialize the streaming service."""
    global streaming_service
    if streaming_service is None:
        try:
            from services.streaming_service import StreamingService

            streaming_service = StreamingService()
            logger.info("Streaming service initialized")
        except Exception as e:
            logger.warning(f"Streaming service not available: {e}")
            return None
    return streaming_service


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


# ==================== CHAT HISTORY MODELS ====================


class MessageResponse(BaseModel):
    id: str = Field(..., description="Message ID")
    session_id: str = Field(..., description="Session ID")
    message_type: str = Field(..., description="Message type (user or assistant)")
    content: str = Field(..., description="Message content")
    model_used: Optional[str] = Field(None, description="Model used for generation")
    timestamp: str = Field(..., description="Message timestamp")
    tokens_used: Optional[int] = Field(None, description="Tokens used")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SessionResponse(BaseModel):
    id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    start_time: str = Field(..., description="Session start time")
    last_activity_time: str = Field(..., description="Last activity time")
    end_time: Optional[str] = Field(None, description="Session end time")
    message_count: int = Field(..., description="Number of messages")
    is_active: bool = Field(..., description="Whether session is active")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session metadata")


class SessionWithMessagesResponse(BaseModel):
    session: SessionResponse = Field(..., description="Session information")
    messages: List[MessageResponse] = Field(..., description="Session messages")


class SessionListResponse(BaseModel):
    sessions: List[SessionResponse] = Field(..., description="List of sessions")
    total: int = Field(..., description="Total number of sessions")


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
    execution_id: Optional[str] = Field(None, description="Execution ID (None for anonymous users)")
    prompt_id: str = Field(..., description="Prompt ID")
    rendered_prompt: str = Field(..., description="Rendered prompt with variables")
    response: str = Field(..., description="Agent response")
    log: List[str] = Field(default_factory=list, description="Execution log")
    tools_used: List[str] = Field(default_factory=list, description="Tools used")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(..., description="Execution timestamp")


# ==================== FEEDBACK MODELS ====================


class FeedbackSubmitRequest(BaseModel):
    """Request model for submitting feedback"""

    # Metadata
    date: Optional[str] = Field(None, description="Date of the output")
    output_id: str = Field(..., description="Output/Response ID")
    prompt: str = Field(..., description="The user's prompt/query that generated the response")
    response: str = Field(..., description="The AI response/result that is being rated")

    # Task Type
    task_types: List[str] = Field(default_factory=list, description="Task types (multiple allowed)")
    task_type_other: Optional[str] = Field(None, description="Other task type if specified")

    # Task Understanding
    query_interpreted_correctly: Optional[str] = Field(None, description="Yes/No")
    followed_instructions: Optional[str] = Field(None, description="Yes/Partial/No")
    task_understanding_notes: Optional[str] = Field(None, description="Additional notes")
    save_to_library: Optional[str] = Field(None, description="Yes/No, see below")

    # Scientific Quality
    accuracy: Optional[str] = Field(None, description="Good/Mixed/Poor")
    completeness: Optional[str] = Field(None, description="Yes/Partial/No")
    scientific_quality_notes: Optional[str] = Field(None, description="Notes/Corrections")

    # Technical Performance
    tools_invoked_correctly: Optional[str] = Field(None, description="Yes/No/Not Sure")
    outputs_usable: Optional[str] = Field(None, description="Yes/Partial/No")
    latency_acceptable: Optional[str] = Field(None, description="Yes/No")
    technical_performance_notes: Optional[str] = Field(None, description="Additional notes")

    # Output Clarity & Usability
    readable_structured: Optional[str] = Field(None, description="Yes/Partial/No")
    formatting_issues: Optional[str] = Field(None, description="None/Minor/Major")
    output_clarity_notes: Optional[str] = Field(None, description="Additional notes")

    # Prompt Handling & Logic
    prompt_followed_instructions: Optional[str] = Field(None, description="Yes/Partial/No")
    prompt_handling_notes: Optional[str] = Field(None, description="Additional notes")
    logical_consistency: Optional[str] = Field(None, description="Strong/Mixed/Weak")
    logical_consistency_notes: Optional[str] = Field(None, description="Additional notes")

    # Overall Rating
    overall_rating: Optional[str] = Field(
        None, description="Excellent/Good but needs tweaks/Needs significant improvement"
    )
    overall_notes: Optional[str] = Field(None, description="Additional notes")


class FeedbackResponse(BaseModel):
    """Response model for feedback operations"""

    success: bool = Field(..., description="Whether the operation was successful")
    feedback_id: Optional[str] = Field(None, description="Feedback ID")
    message: str = Field(..., description="Response message")
    submitted_at: Optional[str] = Field(None, description="Submission timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")


def initialize_agent(config: Optional[ConfigurationRequest] = None, user_id: Optional[str] = None) -> A1:
    """Initialize the Biomni agent with the given configuration.

    If user_id is provided and no config is provided, loads user's saved preferences.
    Priority: provided config > user saved settings > defaults
    """
    global agent

    if agent is not None:
        logger.info("Agent already initialized, reinitializing with new config")

    # Load user settings if no config provided but user_id is available
    if not config and user_id and user_id != "anonymous":
        settings_svc = get_user_settings_service()
        if settings_svc:
            try:
                settings = settings_svc.get_user_settings(user_id, create_if_missing=False)
                if settings:
                    llm_prefs = settings.get("llm_preferences", {})
                    tool_prefs = settings.get("tool_preferences", {})

                    # Create config from user settings
                    config = ConfigurationRequest(
                        llm=llm_prefs.get("model"),
                        source=llm_prefs.get("source"),
                        use_tool_retriever=tool_prefs.get("use_tool_retriever"),
                        timeout_seconds=None,  # Can be added to user settings later if needed
                    )
                    logger.info(f"Loaded agent configuration from user settings for {user_id}")
            except Exception as e:
                logger.warning(f"Failed to load user settings: {e}")

    # Use provided config or defaults
    if config:
        # Auto-detect source from model if source is not provided or is "Unknown"
        source = config.source
        if not source or source == "Unknown":
            if config.llm:
                # Auto-detect source based on model name
                if config.llm.startswith("gpt-") or config.llm.startswith("o1-") or config.llm.startswith("o3-"):
                    source = "OpenAI"
                elif (
                    config.llm.startswith("claude-")
                    or "sonnet" in config.llm.lower()
                    or "opus" in config.llm.lower()
                    or "haiku" in config.llm.lower()
                ):
                    source = "Anthropic"
                elif "gemini" in config.llm.lower():
                    source = "Gemini"
                elif "groq" in config.llm.lower():
                    source = "Groq"
                else:
                    # Default to Anthropic if can't determine
                    source = "Anthropic"
            else:
                source = "Anthropic"  # Default fallback

        agent = A1(
            llm=config.llm,
            source=source,
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
        # Auto-detect source from model if source is not provided or is "Unknown"
        source = config.source
        if not source or source == "Unknown":
            if config.llm:
                # Auto-detect source based on model name
                if config.llm.startswith("gpt-") or config.llm.startswith("o1-") or config.llm.startswith("o3-"):
                    source = "OpenAI"
                elif (
                    config.llm.startswith("claude-")
                    or "sonnet" in config.llm.lower()
                    or "opus" in config.llm.lower()
                    or "haiku" in config.llm.lower()
                ):
                    source = "Anthropic"
                elif "gemini" in config.llm.lower():
                    source = "Gemini"
                elif "groq" in config.llm.lower():
                    source = "Groq"
                else:
                    # Default to Anthropic if can't determine
                    source = "Anthropic"
            else:
                source = "Anthropic"  # Default fallback

        agent = A1(
            llm=config.llm,
            source=source,
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
    if agent is not None:
        # Handle different LLM types (ChatOpenAI uses model_name, ChatAnthropic uses model)
        if hasattr(agent.llm, "model"):
            current_model = agent.llm.model
        elif hasattr(agent.llm, "model_name"):
            current_model = agent.llm.model_name
        elif hasattr(agent.llm, "model_id"):
            current_model = agent.llm.model_id

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
async def chat(request: ChatRequest, user_id: str = Depends(get_optional_user)):
    """Send a message to the agent and get a response using async processing."""
    global agent

    if agent is None or not hasattr(agent, "async_llm"):
        raise HTTPException(status_code=500, detail="Async agent not initialized")

    # Get services
    chat_svc = get_chat_service()

    # Debug logging
    logger.info(
        f"[CHAT] Received request - user_id: {user_id}, session_id: {request.session_id}, chat_svc: {chat_svc is not None}"
    )

    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid4())
        logger.info(f"[CHAT] Using session_id: {session_id}")

        # Determine model and source
        model = (
            request.model
            if request.model
            else agent.llm.model
            if hasattr(agent.llm, "model")
            else "claude-sonnet-4-20250514"
        )
        source = request.source if request.source else ("OpenAI" if model.startswith("gpt-") else "Anthropic")

        # Create or get session in database (only for authenticated users)
        if chat_svc and user_id and user_id != "anonymous":
            try:
                logger.info(f"[CHAT] Attempting to create/get session {session_id} for authenticated user {user_id}")
                session_result = chat_svc.get_or_create_session(
                    session_id=session_id,
                    user_id=user_id,
                    metadata={"model": model, "source": source, "use_tool_retriever": request.use_tool_retriever},
                )
                logger.info(
                    f"[CHAT] Successfully created/retrieved session {session_id} for user {user_id}: {session_result.get('id') if session_result else 'None'}"
                )
            except Exception as e:
                logger.error(f"[CHAT] Failed to create session in database: {e}", exc_info=True)
                # Continue processing even if session creation fails (non-critical for now)
        elif not chat_svc:
            logger.warning(f"[CHAT] Chat service not available - cannot create session or store messages")
        elif not user_id or user_id == "anonymous":
            logger.info(f"[CHAT] Skipping session creation for anonymous user")

        # Store user message in database (only for authenticated users)
        if chat_svc and user_id and user_id != "anonymous":
            try:
                logger.info(
                    f"[CHAT] Attempting to store user message for session {session_id} (message length: {len(request.message)})"
                )
                message_result = chat_svc.store_message(
                    session_id=session_id,
                    message_type="user",
                    content=request.message,  # Store full message
                    user_id=user_id,
                )
                logger.info(
                    f"[CHAT] Successfully stored user message for session {session_id}: message_id={message_result.get('id') if message_result else 'None'}"
                )
            except Exception as e:
                logger.error(f"[CHAT] Failed to store user message: {e}", exc_info=True)
                # Log error but continue processing
        elif not chat_svc:
            logger.warning(f"[CHAT] Chat service not available - cannot store user message")
        elif not user_id or user_id == "anonymous":
            logger.info(f"[CHAT] Skipping user message storage for anonymous user")

        # Update agent configuration if needed
        if request.use_tool_retriever is not None:
            agent.use_tool_retriever = request.use_tool_retriever

        # Track processing time
        start_time = datetime.now()

        # Handle dynamic model selection
        logger.info(f"[CHAT] Received model selection - model: {request.model}, source: {request.source}")
        if request.model or request.source:
            # Create a temporary agent with the requested model
            from biomni.llm import get_async_llm

            logger.info(f"[CHAT] Swapping to model: {model}, source: {source}")
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

        # Calculate processing time
        processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Store assistant message in database (only for authenticated users)
        if chat_svc and user_id and user_id != "anonymous":
            try:
                logger.info(
                    f"[CHAT] Attempting to store assistant message for session {session_id} (response length: {len(response)})"
                )
                message_result = chat_svc.store_message(
                    session_id=session_id,
                    message_type="assistant",
                    content=response,  # Store full response
                    model_used=model,
                    processing_time_ms=processing_time_ms,
                    user_id=user_id,
                )
                logger.info(
                    f"[CHAT] Successfully stored assistant message for session {session_id}: message_id={message_result.get('id') if message_result else 'None'}"
                )
            except Exception as e:
                logger.error(f"[CHAT] Failed to store assistant message: {e}", exc_info=True)
                # Log error but continue
        elif not chat_svc:
            logger.warning(f"[CHAT] Chat service not available - cannot store assistant message")
        elif not user_id or user_id == "anonymous":
            logger.info(f"[CHAT] Skipping assistant message storage for anonymous user")

        # Update session metadata
        if chat_svc:
            try:
                session = chat_svc.get_session(session_id)
                message_count = session.get("message_count", 0) + 2 if session else 2
                chat_svc.update_session_metadata(session_id=session_id, message_count=message_count)
            except Exception as e:
                logger.warning(f"Failed to update session metadata: {e}")

        # Store session info in memory for backward compatibility
        if session_id not in active_sessions:
            active_sessions[session_id] = {"created_at": datetime.now(), "message_count": 0}
        active_sessions[session_id]["message_count"] += 1

        return ChatResponse(
            session_id=session_id, response=response, log=log, timestamp=datetime.now(), status="success"
        )

    except Exception as e:
        # Skip query update - analytics removed from dashboard
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

    # Track WebSocket connection
    connection_id = str(uuid4())
    streaming_svc = get_streaming_service()
    user_id = None

    # Try to extract user_id from query params, token query param, or headers
    user_id = None
    try:
        # Check query parameters
        query_params = dict(websocket.query_params)
        user_id = query_params.get("user_id")
        token = query_params.get("token")  # Check for token in query params (WebSocket workaround)

        logger.info(f"[WEBSOCKET] Query params - user_id: {user_id}, token present: {token is not None}")

        # If we have a token in query params, decode it
        if token and not user_id:
            try:
                from middleware.auth import decode_jwt_token, extract_user_id_from_payload

                payload = decode_jwt_token(token)
                if payload:
                    user_id = extract_user_id_from_payload(payload)
                    logger.info(f"[WEBSOCKET] Successfully extracted user_id from query token: {user_id}")
                else:
                    logger.warning(f"[WEBSOCKET] Failed to decode JWT token from query params")
            except Exception as e:
                logger.error(f"[WEBSOCKET] Could not extract user_id from query token: {e}", exc_info=True)

        # If not in query, try to get from auth token in headers
        if not user_id:
            auth_header = websocket.headers.get("authorization") or websocket.headers.get("Authorization")
            logger.info(
                f"[WEBSOCKET] Authorization header present: {auth_header is not None}, header value: {auth_header[:50] + '...' if auth_header and len(auth_header) > 50 else auth_header}"
            )
            if auth_header and auth_header.startswith("Bearer "):
                try:
                    from middleware.auth import decode_jwt_token, extract_user_id_from_payload

                    token = auth_header.replace("Bearer ", "")
                    payload = decode_jwt_token(token)
                    if payload:
                        user_id = extract_user_id_from_payload(payload)
                        logger.info(f"[WEBSOCKET] Successfully extracted user_id from header token: {user_id}")
                    else:
                        logger.warning(f"[WEBSOCKET] Failed to decode JWT token from header")
                except Exception as e:
                    logger.error(
                        f"[WEBSOCKET] Could not extract user_id from WebSocket header token: {e}", exc_info=True
                    )
            else:
                logger.warning(f"[WEBSOCKET] No valid Authorization header found")
    except Exception as e:
        logger.error(f"[WEBSOCKET] Error extracting user_id from WebSocket: {e}", exc_info=True)

    # Get services for session and message management
    chat_svc = get_chat_service()
    streaming_svc = get_streaming_service()

    logger.info(
        f"[WEBSOCKET] Connection setup - user_id: {user_id}, chat_svc: {chat_svc is not None}, streaming_svc: {streaming_svc is not None}"
    )

    # Create or ensure chat session exists in database (only for authenticated users)
    if chat_svc and user_id and user_id != "anonymous":
        try:
            # Validate user_id format (should be UUID)
            from uuid import UUID
            from utils.user_utils import ensure_user_exists

            UUID(user_id)  # Validate UUID format
            # Check if user exists in database
            if not ensure_user_exists(chat_svc.client, user_id):
                logger.warning(f"User {user_id} not found in database, rejecting WebSocket connection")
                await websocket.close(code=1008, reason="User not authenticated")
                return

            logger.info(f"[WEBSOCKET] Attempting to create/get session {session_id} for user {user_id}")
            session_result = chat_svc.get_or_create_session(
                session_id=session_id,
                user_id=user_id,
                metadata={"connection_type": "websocket"},
            )
            logger.info(
                f"[WEBSOCKET] Successfully created/retrieved chat session {session_id} for user: {user_id}, session_record_id: {session_result.get('id') if session_result else 'None'}"
            )
        except ValueError as e:
            logger.error(f"Invalid user_id for WebSocket: {e}")
            await websocket.close(code=1008, reason="Invalid user authentication")
            return
        except Exception as e:
            logger.error(f"Failed to create/ensure chat session {session_id}: {e}")
            await websocket.close(code=1011, reason="Session creation failed")
            return
    elif not user_id or user_id == "anonymous":
        logger.info("Rejecting WebSocket connection from anonymous user")
        await websocket.close(code=1008, reason="Authentication required")
        return

    # Create streaming session record (only for authenticated users)
    if streaming_svc and user_id and user_id != "anonymous":
        try:
            streaming_svc.create_streaming_session(
                connection_id=connection_id,
                user_id=user_id,
                chat_session_id=session_id,
                metadata={
                    "client_host": websocket.client.host if websocket.client else None,
                    "client_port": websocket.client.port if websocket.client else None,
                },
            )
            logger.info(f"Created streaming session for connection {connection_id} (user: {user_id})")
        except Exception as e:
            logger.error(f"Failed to create streaming session record: {e}")
            # Continue anyway - streaming can work without the record

    try:
        while True:
            try:
                # Receive message from client with timeout to prevent hanging
                data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                message_data = json.loads(data)
                message = message_data.get("message", "")

                if not message:
                    continue

                # Ensure session exists before storing messages (only for authenticated users)
                # Session should already exist from connection, but this is a safety check
                if chat_svc and user_id and user_id != "anonymous":
                    try:
                        chat_svc.get_or_create_session(
                            session_id=session_id,
                            user_id=user_id,
                            metadata={"connection_type": "websocket", "last_activity": datetime.now().isoformat()},
                        )
                    except Exception as e:
                        logger.error(f"Failed to ensure session exists: {e}")
                        # Continue - session might already exist

                # Store user message in database (only for authenticated users)
                if chat_svc and user_id and user_id != "anonymous":
                    try:
                        logger.info(
                            f"[WEBSOCKET] Attempting to store user message for session {session_id} (message length: {len(message)})"
                        )
                        message_result = chat_svc.store_message(
                            session_id=session_id,
                            message_type="user",
                            content=message,
                            user_id=user_id,
                        )
                        logger.info(
                            f"[WEBSOCKET] Successfully stored user message for session {session_id}: message_id={message_result.get('id') if message_result else 'None'}"
                        )
                    except Exception as e:
                        logger.error(f"[WEBSOCKET] Failed to store user message: {e}", exc_info=True)
                        # Continue processing even if storage fails
                elif not chat_svc:
                    logger.warning(f"[WEBSOCKET] Chat service not available - cannot store user message")
                elif not user_id or user_id == "anonymous":
                    logger.info(f"[WEBSOCKET] Skipping user message storage for anonymous user")

                # Update agent configuration if provided
                use_tool_retriever = message_data.get("use_tool_retriever", True)
                if use_tool_retriever is not None:
                    agent.use_tool_retriever = use_tool_retriever

                # Handle dynamic model selection for streaming
                model = message_data.get("model")
                source = message_data.get("source")

                logger.info(f"[WEBSOCKET] Received model selection - model: {model}, source: {source}")

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

                    logger.info(f"[WEBSOCKET] Swapping to model: {selected_model}, source: {selected_source}")

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
                    full_response = ""  # Accumulate full response for storage
                    try:
                        async for step in agent.go_stream_async(message):
                            try:
                                step_count += 1
                                step_output = step.get("output", "")
                                full_response += step_output  # Accumulate response

                                response_data = StreamResponse(
                                    session_id=session_id,
                                    output=step_output,
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

                            # Store assistant message after streaming completes (only for authenticated users)
                            if chat_svc and full_response and user_id and user_id != "anonymous":
                                try:
                                    model = (
                                        selected_model
                                        if model or source
                                        else (
                                            agent.llm.model
                                            if hasattr(agent.llm, "model")
                                            else "claude-sonnet-4-20250514"
                                        )
                                    )
                                    logger.info(
                                        f"[WEBSOCKET] Attempting to store assistant message for session {session_id} (response length: {len(full_response)})"
                                    )
                                    message_result = chat_svc.store_message(
                                        session_id=session_id,
                                        message_type="assistant",
                                        content=full_response,
                                        model_used=model,
                                        user_id=user_id,
                                    )
                                    logger.info(
                                        f"[WEBSOCKET] Successfully stored assistant message for session {session_id}: message_id={message_result.get('id') if message_result else 'None'}"
                                    )
                                except Exception as store_error:
                                    logger.error(
                                        f"[WEBSOCKET] Failed to store assistant message: {store_error}", exc_info=True
                                    )
                            elif not chat_svc:
                                logger.warning(
                                    f"[WEBSOCKET] Chat service not available - cannot store assistant message"
                                )
                            elif not user_id or user_id == "anonymous":
                                logger.info(f"[WEBSOCKET] Skipping assistant message storage for anonymous user")
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
                    full_response = ""  # Accumulate full response for storage
                    try:
                        async for step in agent.go_stream_async(message):
                            try:
                                step_count += 1
                                step_output = step.get("output", "")
                                full_response += step_output  # Accumulate response

                                response_data = StreamResponse(
                                    session_id=session_id,
                                    output=step_output,
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

                            # Store assistant message after streaming completes (only for authenticated users)
                            if chat_svc and full_response and user_id and user_id != "anonymous":
                                try:
                                    model = (
                                        agent.llm.model if hasattr(agent.llm, "model") else "claude-sonnet-4-20250514"
                                    )
                                    chat_svc.store_message(
                                        session_id=session_id,
                                        message_type="assistant",
                                        content=full_response,
                                        model_used=model,
                                        user_id=user_id,
                                    )
                                    logger.info(f"Stored assistant message for session {session_id}")
                                except Exception as store_error:
                                    logger.error(f"Failed to store assistant message: {store_error}")
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
        # Update streaming session status
        if streaming_svc:
            try:
                streaming_svc.end_streaming_session(connection_id)
            except Exception as e:
                logger.warning(f"Failed to update streaming session on disconnect: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket stream: {e}")
        logger.error(traceback.format_exc())
        # Update streaming session status with error
        if streaming_svc:
            try:
                streaming_svc.update_streaming_status(connection_id, "error", error_message=str(e))
            except Exception as update_error:
                logger.warning(f"Failed to update streaming session on error: {update_error}")
    finally:
        # Ensure streaming session is marked as ended
        if streaming_svc:
            try:
                streaming_svc.end_streaming_session(connection_id)
            except Exception as e:
                logger.warning(f"Failed to end streaming session in finally: {e}")
        logger.info(f"WebSocket cleanup completed for session {session_id}")


@app.post("/api/configure")
async def configure_agent(request: ConfigurationRequest, user_id: str = Depends(get_optional_user)):
    """Configure the agent with new settings and save to user preferences."""
    global agent

    try:
        # Configure the agent with async capabilities
        agent = initialize_agent(request, user_id=user_id if user_id and user_id != "anonymous" else None)
        await agent.async_init()
        await agent.configure_async()

        # Save configuration to user settings if user is authenticated
        if user_id and user_id != "anonymous":
            settings_svc = get_user_settings_service()
            if settings_svc:
                try:
                    # Prepare LLM preferences
                    llm_prefs = {}
                    if request.llm:
                        llm_prefs["model"] = request.llm
                    if request.source:
                        llm_prefs["source"] = request.source
                    if hasattr(agent.llm, "temperature"):
                        llm_prefs["temperature"] = agent.llm.temperature

                    # Prepare tool preferences
                    tool_prefs = {}
                    if request.use_tool_retriever is not None:
                        tool_prefs["use_tool_retriever"] = request.use_tool_retriever

                    # Update settings
                    updates = {}
                    if llm_prefs:
                        # Merge with existing preferences
                        existing = settings_svc.get_llm_preferences(user_id)
                        existing.update(llm_prefs)
                        updates["llm_preferences"] = existing
                    if tool_prefs:
                        # Merge with existing preferences
                        existing = settings_svc.get_tool_preferences(user_id)
                        existing.update(tool_prefs)
                        updates["tool_preferences"] = existing

                    if updates:
                        settings_svc.update_user_settings(user_id, updates)
                        logger.info(f"Saved agent configuration to user settings for {user_id}")
                except Exception as e:
                    logger.warning(f"Failed to save configuration to user settings: {e}")
                    # Don't fail the configuration if settings save fails

        return {"message": "Agent configured successfully", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error configuring agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/settings")
async def get_user_settings(user_id: str = Depends(get_optional_user)):
    """Get user settings."""
    if not user_id or user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required")

    settings_svc = get_user_settings_service()
    if not settings_svc:
        raise HTTPException(status_code=503, detail="User settings service not available")

    try:
        settings = settings_svc.get_user_settings(user_id)
        return {"settings": settings, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error getting user settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/settings")
async def update_user_settings(settings_updates: Dict[str, Any], user_id: str = Depends(get_optional_user)):
    """Update user settings (partial update supported)."""
    if not user_id or user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required")

    settings_svc = get_user_settings_service()
    if not settings_svc:
        raise HTTPException(status_code=503, detail="User settings service not available")

    try:
        updated_settings = settings_svc.update_user_settings(user_id, settings_updates)
        return {"settings": updated_settings, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error updating user settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/settings/llm")
async def get_llm_preferences(user_id: str = Depends(get_optional_user)):
    """Get LLM preferences for user."""
    if not user_id or user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required")

    settings_svc = get_user_settings_service()
    if not settings_svc:
        raise HTTPException(status_code=503, detail="User settings service not available")

    try:
        prefs = settings_svc.get_llm_preferences(user_id)
        return {"llm_preferences": prefs, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error getting LLM preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/settings/llm")
async def update_llm_preferences(llm_preferences: Dict[str, Any], user_id: str = Depends(get_optional_user)):
    """Update LLM preferences."""
    if not user_id or user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required")

    settings_svc = get_user_settings_service()
    if not settings_svc:
        raise HTTPException(status_code=503, detail="User settings service not available")

    try:
        updated_settings = settings_svc.update_llm_preferences(user_id, llm_preferences)
        return {"settings": updated_settings, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error updating LLM preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/settings/tools")
async def get_tool_preferences(user_id: str = Depends(get_optional_user)):
    """Get tool preferences for user."""
    if not user_id or user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required")

    settings_svc = get_user_settings_service()
    if not settings_svc:
        raise HTTPException(status_code=503, detail="User settings service not available")

    try:
        prefs = settings_svc.get_tool_preferences(user_id)
        return {"tool_preferences": prefs, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error getting tool preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/settings/tools")
async def update_tool_preferences(tool_preferences: Dict[str, Any], user_id: str = Depends(get_optional_user)):
    """Update tool preferences."""
    if not user_id or user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required")

    settings_svc = get_user_settings_service()
    if not settings_svc:
        raise HTTPException(status_code=503, detail="User settings service not available")

    try:
        updated_settings = settings_svc.update_tool_preferences(user_id, tool_preferences)
        return {"settings": updated_settings, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error updating tool preferences: {e}")
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
        # Handle different LLM types (ChatOpenAI uses model_name, ChatAnthropic uses model)
        llm_model = None
        if hasattr(agent.llm, "model"):
            llm_model = agent.llm.model
        elif hasattr(agent.llm, "model_name"):
            llm_model = agent.llm.model_name
        elif hasattr(agent.llm, "model_id"):
            llm_model = agent.llm.model_id
        else:
            # Fallback: try to get from any attribute that might contain the model name
            llm_model = getattr(agent.llm, "model", getattr(agent.llm, "model_name", "unknown"))

        config = {
            "llm": llm_model or "unknown",
            "source": getattr(agent, "source", "Unknown"),
            "use_tool_retriever": agent.use_tool_retriever,
            "temperature": getattr(agent.llm, "temperature", 0.7),
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
async def upload_data_file(
    file: UploadFile = File(...),
    description: str = Form(...),
    name: str = Form(...),
    session_id: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    user_id: str = Depends(get_optional_user),
):
    """Upload a custom data file to the data lake and store metadata in database."""
    global agent

    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    # Require authenticated user for file uploads
    if not user_id or user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required for file uploads")

    # Parse tags from JSON string
    tags_list = []
    if tags:
        try:
            tags_list = json.loads(tags)
            # Ensure it's a list
            if not isinstance(tags_list, list):
                logger.warning(f"Tags is not a list, converting: {tags_list}")
                tags_list = []
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse tags JSON: {tags}, error: {e}, using empty list")
            tags_list = []

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

        # Save the file to filesystem
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Store metadata in database using FileStorageService
        file_storage_svc = get_file_storage_service()
        file_record = None

        if file_storage_svc:
            try:
                file_record = file_storage_svc.upload_file(
                    user_id=user_id,
                    file_path=file_path,
                    original_filename=name,
                    file_size=file.size or 0,
                    file_type=file.content_type or file_extension,
                    description=description,
                    session_id=session_id,
                    storage_provider="local",
                    tags=tags_list,
                )
                logger.info(f"File metadata stored in database: {file_record.get('id')}")
            except Exception as e:
                logger.error(f"Failed to store file metadata in database: {e}")
                logger.error(f"Error details: {type(e).__name__}: {str(e)}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                # Continue even if DB storage fails to maintain backward compatibility
        else:
            logger.warning("File storage service not available, storing only in filesystem")

        # Maintain backward compatibility: add to agent's data lake dictionary
        if not hasattr(agent, "data_lake_dict"):
            agent.data_lake_dict = {}
        agent.data_lake_dict[safe_filename] = description

        logger.info(f"File uploaded successfully: {safe_filename}")

        response = {
            "message": "File uploaded successfully",
            "filename": safe_filename,
            "path": file_path,
            "size": file.size,
            "description": description,
            "timestamp": datetime.now().isoformat(),
        }

        # Add file_id if metadata was stored
        if file_record:
            response["file_id"] = file_record.get("id")
            response["id"] = file_record.get("id")  # For backward compatibility

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/data/{file_identifier}")
async def delete_data_file(file_identifier: str, user_id: str = Depends(get_optional_user)):
    """Delete a custom data file from the data lake and database.

    Supports both file_id (UUID) and filename for backward compatibility.
    """
    global agent

    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    # Require authenticated user for file deletion
    if not user_id or user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required for file deletion")

    try:
        file_storage_svc = get_file_storage_service()
        file_path = None
        file_id = None

        # Check if identifier is a UUID (file_id) or filename
        is_uuid = len(file_identifier) == 36 and file_identifier.count("-") == 4

        if is_uuid and file_storage_svc:
            # Try to get file metadata by ID
            file_record = file_storage_svc.get_file_metadata(file_identifier, user_id)
            if file_record:
                file_id = file_identifier
                file_path = file_record.get("file_path")
                filename = file_record.get("filename")
            else:
                raise HTTPException(status_code=404, detail="File not found or access denied")
        else:
            # Backward compatibility: treat as filename
            filename = file_identifier
        data_lake_path = os.path.join(agent.path, "data_lake")
        file_path = os.path.join(data_lake_path, filename)

        # Try to find file_id in database if service is available
        if file_storage_svc:
            user_files = file_storage_svc.get_user_files(user_id, limit=1000)
            for f in user_files:
                if f.get("filename") == filename:
                    file_id = f.get("id")
                    break

        # Delete from filesystem
        filesystem_deleted = False
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"File removed from filesystem: {file_path}")
                filesystem_deleted = True
            except Exception as fs_error:
                logger.warning(f"Failed to remove file from filesystem {file_path}: {fs_error}")

        if not filesystem_deleted and filename:
            # If we couldn't determine file path, try default location
            data_lake_path = os.path.join(agent.path, "data_lake")
            default_path = os.path.join(data_lake_path, filename)
            if os.path.exists(default_path):
                try:
                    os.remove(default_path)
                    logger.info(f"File removed from default location: {default_path}")
                    filesystem_deleted = True
                except Exception as fs_error:
                    logger.warning(f"Failed to remove file from default location {default_path}: {fs_error}")

        # Delete from database if we have file_id
        database_deleted = False
        if file_id and file_storage_svc:
            try:
                file_storage_svc.delete_file(file_id, user_id)
                logger.info(f"File metadata deleted from database: {file_id}")
                database_deleted = True
            except Exception as e:
                logger.warning(f"Failed to delete file metadata from database: {e}")
                # If database deletion fails but file was deleted from filesystem, still consider it a partial success
                if filesystem_deleted:
                    logger.info("File deleted from filesystem but not from database - partial success")

        # Remove from agent's data lake dictionary (backward compatibility)
        if hasattr(agent, "data_lake_dict"):
            dict_key = filename if filename else file_identifier
            if dict_key in agent.data_lake_dict:
                del agent.data_lake_dict[dict_key]

        # Determine success status
        if database_deleted or filesystem_deleted:
            if database_deleted and filesystem_deleted:
                logger.info(f"File deleted successfully from both filesystem and database: {file_identifier}")
                message = "File deleted successfully"
            elif database_deleted:
                logger.info(f"File metadata deleted from database (filesystem may not exist): {file_identifier}")
                message = "File metadata deleted successfully"
            else:
                logger.info(f"File deleted from filesystem (database record may not exist): {file_identifier}")
                message = "File deleted from filesystem successfully"
        else:
            # Neither deletion succeeded
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {file_identifier}. Could not delete from filesystem or database.",
            )

        return {
            "message": message,
            "file_id": file_id,
            "filename": filename if filename else file_identifier,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/download/{file_identifier}")
async def download_data_file(file_identifier: str, user_id: str = Depends(get_optional_user)):
    """Download a custom data file."""
    global agent

    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    if not user_id or user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        file_storage_svc = get_file_storage_service()
        file_path = None
        filename = None

        # Check if identifier is a UUID (file_id) or filename
        is_uuid = len(file_identifier) == 36 and file_identifier.count("-") == 4

        if is_uuid and file_storage_svc:
            # Try to get file metadata by ID
            file_record = file_storage_svc.get_file_metadata(file_identifier, user_id)
            if file_record:
                file_path = file_record.get("file_path")
                filename = file_record.get("original_filename") or file_record.get("filename")
            else:
                raise HTTPException(status_code=404, detail="File not found or access denied")
        else:
            # Backward compatibility: treat as filename
            filename = file_identifier
            data_lake_path = os.path.join(agent.path, "data_lake")
            file_path = os.path.join(data_lake_path, filename)

            # Verify user owns the file
            if file_storage_svc:
                user_files = file_storage_svc.get_user_files(user_id, limit=1000)
                file_found = False
                for f in user_files:
                    if f.get("filename") == filename or f.get("original_filename") == filename:
                        file_found = True
                        file_path = f.get("file_path") or file_path
                        filename = f.get("original_filename") or filename
                        break
                if not file_found:
                    raise HTTPException(status_code=404, detail="File not found or access denied")

        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found on filesystem")

        return FileResponse(path=file_path, filename=filename, media_type="application/octet-stream")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/files")
async def get_user_files(
    session_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    user_id: str = Depends(get_optional_user),
):
    """Get list of user's uploaded files, optionally filtered by session."""
    file_storage_svc = get_file_storage_service()

    if not file_storage_svc:
        raise HTTPException(status_code=503, detail="File storage service not available")

    if not user_id or user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        files = file_storage_svc.get_user_files(user_id=user_id, session_id=session_id, limit=limit, offset=offset)
        logger.info(f"Retrieved {len(files)} files for user {user_id} (session_id: {session_id})")

        # Debug: Log files with tags to ensure they're being returned
        for file in files:
            if file.get("tags"):
                logger.info(f"File {file.get('original_filename')} has tags: {file.get('tags')}")
            else:
                logger.debug(f"File {file.get('original_filename')} has no tags (tags field: {file.get('tags')})")

        return {
            "files": files,
            "count": len(files),
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting user files: {e}")
        logger.error(f"Error details: {type(e).__name__}: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== PROMPT LIBRARY ENDPOINTS ====================


@app.post("/api/prompts", response_model=PromptResponse)
async def create_prompt(request: CreatePromptRequest, user_id: str = Depends(get_optional_user)):
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
    user_id: str = Depends(get_optional_user),
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

        # Log user authentication status
        if user_id == "anonymous":
            logger.warning("Listing prompts for anonymous user")
        else:
            logger.info(f"Listing prompts for authenticated user: {user_id}")

        prompts = await service.list_prompts(
            user_id=user_id,
            category=category,
            tags=tag_list,
            search_query=search,
            limit=limit,
            offset=offset,
        )

        logger.info(f"Returning {len(prompts)} prompts to frontend")
        return prompts

    except Exception as e:
        logger.error(f"Error listing prompts: {e}")
        logger.error(traceback.format_exc())
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
async def update_prompt(prompt_id: str, request: UpdatePromptRequest, user_id: str = Depends(get_optional_user)):
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
async def execute_prompt(request: ExecutePromptRequest, user_id: str = Depends(get_optional_user)):
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
        logger.info(f"Executing prompt {request.prompt_id}")
        logger.info(f"Variables received: {request.variables}")
        logger.info(f"Variable count: {len(request.variables) if request.variables else 0}")
        if request.variables:
            for key, value in request.variables.items():
                logger.info(
                    f"  - {key}: '{value}' (type: {type(value).__name__}, length: {len(str(value)) if value else 0})"
                )

        rendered_prompt = await service.render_prompt(request.prompt_id, request.variables)
        logger.info(f"Rendered prompt length: {len(rendered_prompt)} chars")
        logger.info(f"Rendered prompt (first 500 chars): {rendered_prompt[:500]}")

        # Check if any placeholders remain
        import re

        remaining = re.findall(r"\{([^}]+)\}", rendered_prompt)
        if remaining:
            logger.warning(f"⚠️ WARNING: Unsubstituted placeholders found in rendered prompt: {remaining}")

        # Get model config with priority: override > agent's current config > prompt's stored config
        # Note: Use override_llm_config (field name) not override_model_config (alias)
        if request.override_llm_config:
            # Use explicit override from request
            model_config = request.override_llm_config.dict()
        else:
            # Use agent's current configuration (what user configured in Configuration page)
            # This ensures the selected model (e.g., GPT-5) is used instead of prompt's stored config
            model_config = {}

            # Get model from agent's current LLM (handle different LLM types)
            current_model = None
            if hasattr(agent.llm, "model"):
                current_model = agent.llm.model
            elif hasattr(agent.llm, "model_name"):
                current_model = agent.llm.model_name
            elif hasattr(agent.llm, "model_id"):
                current_model = agent.llm.model_id
            elif hasattr(agent.llm, "_model_name"):
                current_model = agent.llm._model_name

            if current_model:
                model_config["model"] = current_model
                # Get source from agent
                model_config["source"] = getattr(agent, "source", None)
                # Get temperature from agent's LLM
                model_config["temperature"] = getattr(agent.llm, "temperature", 0.7)
                logger.info(
                    f"Using agent's current configuration: model={current_model}, source={model_config.get('source')}"
                )
            else:
                # If agent doesn't have a model configured, fall back to prompt's stored config
                logger.warning("Agent's LLM doesn't have a model attribute, falling back to prompt's stored config")
                model_config = prompt.get("model_config", {})
                if not model_config:
                    # Final fallback
                    model_config = {"model": "claude-sonnet-4-20250514", "source": "Anthropic", "temperature": 0.7}
                    logger.warning("Using default fallback configuration")

        # Configure agent if needed
        tool_bindings = prompt.get("tool_bindings", {})
        if tool_bindings.get("use_tool_retriever") is not None:
            agent.use_tool_retriever = tool_bindings["use_tool_retriever"]

        # Determine model and source for execution
        model = model_config.get("model") if model_config else None
        source = model_config.get("source") if model_config else None

        # Auto-detect source if not provided
        if not source and model:
            if model.startswith("gpt-") or model.startswith("o1-") or model.startswith("o3-"):
                source = "OpenAI"
            elif model.startswith("claude-") or "sonnet" in model.lower():
                source = "Anthropic"
            else:
                source = getattr(agent, "source", "Anthropic")

        # Log the model being used for debugging
        logger.info(f"Executing prompt {request.prompt_id} with model: {model}, source: {source}")

        # Log the execution start
        execution_id = await service.execute_prompt(
            prompt_id=request.prompt_id,
            user_id=user_id,
            variables=request.variables,
            session_id=request.session_id,
            override_model_config=model_config,
        )

        # Execute the agent with model override if needed
        start_time = datetime.now()

        # Always create a temporary LLM to ensure correct model configuration
        # This ensures GPT-5 and other models without stop_sequences support work correctly
        from biomni.llm import get_async_llm

        temp_llm = await get_async_llm(
            model=model,
            source=source,
            temperature=model_config.get("temperature", getattr(agent.llm, "temperature", 0.7)),
        )

        # Temporarily swap the LLM
        original_llm = agent.llm
        original_async_llm = agent.async_llm
        original_timeout = agent.timeout_seconds if hasattr(agent, "timeout_seconds") else 600

        agent.llm = temp_llm
        agent.async_llm = temp_llm

        # Ensure agent uses at least 20 minutes timeout for prompt execution
        timeout_seconds = max(original_timeout, 1200)  # Use at least 20 minutes
        agent.timeout_seconds = timeout_seconds

        try:
            logger.info(
                f"Executing prompt with timeout of {timeout_seconds} seconds ({timeout_seconds / 60:.1f} minutes)"
            )

            # Execute with timeout to prevent hanging
            try:
                log, response = await asyncio.wait_for(agent.go_async(rendered_prompt), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.error(f"Prompt execution timed out after {timeout_seconds} seconds")
                raise HTTPException(
                    status_code=504,
                    detail=f"Prompt execution timed out after {timeout_seconds} seconds ({timeout_seconds / 60:.1f} minutes). The prompt may be too complex or the model is taking too long to respond. Try simplifying the prompt or increasing the timeout in configuration.",
                )
        finally:
            # Restore original LLM and timeout
            agent.llm = original_llm
            agent.async_llm = original_async_llm
            agent.timeout_seconds = original_timeout

        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        logger.info(f"Prompt execution completed successfully in {processing_time}ms")

        # Extract tools used from log
        tools_used = []
        for log_entry in log:
            if "Tool:" in log_entry or "Function:" in log_entry:
                # Simple extraction - could be improved
                tools_used.append(log_entry)

        # Update execution with results (only if execution was logged for authenticated users)
        if execution_id:
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
async def add_favorite(prompt_id: str, user_id: str = Depends(get_optional_user)):
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
async def remove_favorite(prompt_id: str, user_id: str = Depends(get_optional_user)):
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
async def get_user_favorites(user_id: str = Depends(get_optional_user)):
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
async def get_user_prompt_stats(user_id: str = Depends(get_optional_user)):
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


# ==================== FEEDBACK ENDPOINTS ====================


@app.get("/api/feedback/schema")
async def get_feedback_schema():
    """Get the feedback form schema for frontend integration."""
    service = get_feedback_service()

    try:
        schema = service.get_feedback_schema()
        return schema

    except Exception as e:
        logger.error(f"Error getting feedback schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackSubmitRequest,
    user_id: str = Depends(get_optional_user),
    session_id: Optional[str] = Query(None, description="Optional session ID for feedback"),
):
    """Submit user feedback for an AI response."""
    service = get_feedback_service()

    # Log authentication status and session info for debugging
    if user_id == "anonymous":
        logger.warning(
            f"Feedback submission attempted by anonymous user. Authorization header may be missing or invalid."
        )
    else:
        logger.info(f"Feedback submission from authenticated user: {user_id}")

    # Log session_id status
    if session_id:
        logger.info(f"Feedback submission with session_id: {session_id}")
    else:
        logger.warning("Feedback submission without session_id - session may not be linked")

    try:
        # Convert request to dictionary
        feedback_data = request.dict()

        # Submit feedback with user and session info
        result = service.submit_feedback(feedback_data, user_id=user_id, session_id=session_id)

        # Check if submission was successful
        if result.get("success") is False:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"Feedback submission failed: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)

        return FeedbackResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/feedback/{feedback_id}")
async def get_feedback_by_id(feedback_id: str, user_id: Optional[str] = None):
    """Get a specific feedback entry by ID."""
    service = get_feedback_service()

    try:
        feedback = service.get_feedback(feedback_id, user_id=user_id)
        if feedback is None:
            raise HTTPException(status_code=404, detail="Feedback not found")

        return feedback

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/feedback")
async def get_all_feedback(limit: Optional[int] = 100, user_id: Optional[str] = None, session_id: Optional[str] = None):
    """Get all feedback entries."""
    service = get_feedback_service()

    try:
        feedback_list = service.get_all_feedback(limit=limit, user_id=user_id, session_id=session_id)
        return {"feedback": feedback_list, "count": len(feedback_list), "timestamp": datetime.now().isoformat()}

    except Exception as e:
        logger.error(f"Error getting all feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== CHAT HISTORY ENDPOINTS ====================


@app.get("/api/sessions", response_model=SessionListResponse)
async def list_user_sessions(
    user_id: str = Depends(get_optional_user), limit: int = 50, offset: int = 0, active_only: bool = False
):
    """List all chat sessions for a user."""
    service = get_chat_service()

    if not service:
        raise HTTPException(status_code=503, detail="Chat service not available")

    try:
        sessions = service.list_user_sessions(user_id=user_id, limit=limit, offset=offset, active_only=active_only)

        return SessionListResponse(sessions=[SessionResponse(**session) for session in sessions], total=len(sessions))

    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}", response_model=SessionWithMessagesResponse)
async def get_session_with_messages(session_id: str, user_id: str = Depends(get_optional_user)):
    """Get a session with all its messages."""
    service = get_chat_service()

    if not service:
        raise HTTPException(status_code=503, detail="Chat service not available")

    try:
        session = service.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if session.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        messages = service.get_session_messages(session_id)

        return SessionWithMessagesResponse(
            session=SessionResponse(**session), messages=[MessageResponse(**msg) for msg in messages]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str, user_id: str = Depends(get_optional_user), limit: Optional[int] = None, offset: int = 0
):
    """Get messages for a specific session."""
    service = get_chat_service()

    if not service:
        raise HTTPException(status_code=503, detail="Chat service not available")

    try:
        session = service.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if session.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        messages = service.get_session_messages(session_id, limit=limit, offset=offset)

        return {
            "session_id": session_id,
            "messages": [MessageResponse(**msg) for msg in messages],
            "total": len(messages),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, user_id: str = Depends(get_optional_user)):
    """Delete a chat session and all its messages."""
    service = get_chat_service()

    if not service:
        raise HTTPException(status_code=503, detail="Chat service not available")

    try:
        success = service.delete_session(session_id, user_id=user_id)

        if not success:
            raise HTTPException(status_code=404, detail="Session not found or access denied")

        return {"message": "Session deleted successfully", "session_id": session_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions/{session_id}/end")
async def end_session(session_id: str, user_id: str = Depends(get_optional_user)):
    """Mark a session as inactive."""
    service = get_chat_service()

    if not service:
        raise HTTPException(status_code=503, detail="Chat service not available")

    try:
        session = service.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if session.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        success = service.end_session(session_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to end session")

        return {"message": "Session ended successfully", "session_id": session_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== QUERY ANALYTICS ENDPOINTS ====================


if __name__ == "__main__":
    # Run the server
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
