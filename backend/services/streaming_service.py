"""
Streaming Service for Biomni

Manages WebSocket connection tracking in Supabase database.
Tracks active streaming sessions for monitoring and analytics.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.utils.user_utils import ensure_session_exists, ensure_user_exists
from supabase import Client, create_client

logger = logging.getLogger(__name__)


class StreamingService:
    """Service for managing WebSocket streaming session tracking."""

    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """
        Initialize the Streaming Service.

        Args:
            supabase_url: Supabase project URL (defaults to SUPABASE_URL env var)
            supabase_key: Supabase service role key (defaults to SUPABASE_SERVICE_ROLE_KEY env var)

        Raises:
            ValueError: If credentials are not provided
        """
        self.url = supabase_url or os.getenv("SUPABASE_URL")
        self.key = supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not self.url or not self.key:
            raise ValueError(
                "Supabase URL and key must be provided either as parameters "
                "or via SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables"
            )

        try:
            self.client: Client = create_client(self.url, self.key)
            logger.info("Streaming service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize streaming service: {e}")
            raise

    def create_streaming_session(
        self,
        connection_id: str,
        user_id: Optional[str] = None,
        chat_session_id: Optional[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Create a new streaming session record.

        Args:
            connection_id: Unique WebSocket connection identifier
            user_id: Optional user identifier
            chat_session_id: Optional chat session ID being streamed
            metadata: Optional additional metadata (IP, user agent, etc.)

        Returns:
            Created streaming session record
        """
        # Validate user if provided
        if user_id and user_id != "anonymous":
            if not ensure_user_exists(self.client, user_id):
                logger.warning(f"User {user_id} not found, creating streaming session without user link")

        # Validate chat session if provided
        if chat_session_id and not ensure_session_exists(self.client, chat_session_id):
            logger.warning(f"Chat session {chat_session_id} not found, creating streaming session without session link")

        try:
            session_data = {
                "connection_id": connection_id,
                "status": "active",
                "metadata": metadata or {},
            }

            if user_id and user_id != "anonymous":
                session_data["user_id"] = user_id
            if chat_session_id:
                session_data["chat_session_id"] = chat_session_id

            result = self.client.table("streaming_sessions").insert(session_data).execute()

            if not result.data or len(result.data) == 0:
                raise Exception("Failed to create streaming session: No data returned")

            logger.info(f"Created streaming session: {result.data[0].get('id')} for connection {connection_id}")
            return result.data[0]

        except Exception as e:
            logger.error(f"Error creating streaming session: {e}")
            raise

    def update_streaming_status(
        self,
        connection_id: str,
        status: str,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update streaming session status.

        Args:
            connection_id: Connection identifier
            status: New status ('active', 'completed', 'disconnected', 'error')
            error_message: Optional error message
            metadata: Optional metadata updates

        Returns:
            True if updated successfully
        """
        try:
            update_data = {"status": status}

            if status in ("completed", "disconnected", "error"):
                update_data["ended_at"] = datetime.now().isoformat()

            if error_message:
                update_data["error_message"] = error_message

            if metadata:
                # Merge with existing metadata
                existing = (
                    self.client.table("streaming_sessions")
                    .select("metadata")
                    .eq("connection_id", connection_id)
                    .execute()
                )
                if existing.data and existing.data[0].get("metadata"):
                    existing_metadata = existing.data[0]["metadata"]
                    existing_metadata.update(metadata)
                    update_data["metadata"] = existing_metadata
                else:
                    update_data["metadata"] = metadata

            result = (
                self.client.table("streaming_sessions").update(update_data).eq("connection_id", connection_id).execute()
            )

            if result.data and len(result.data) > 0:
                logger.info(f"Updated streaming session status: {connection_id} -> {status}")
                return True
            else:
                logger.warning(f"No streaming session found for connection: {connection_id}")
                return False

        except Exception as e:
            logger.error(f"Error updating streaming session status: {e}")
            return False

    def end_streaming_session(self, connection_id: str, error_message: Optional[str] = None) -> bool:
        """
        End a streaming session.

        Args:
            connection_id: Connection identifier
            error_message: Optional error message if session ended with error

        Returns:
            True if ended successfully
        """
        status = "error" if error_message else "completed"
        return self.update_streaming_status(connection_id, status, error_message=error_message)

    def get_active_connections(
        self, user_id: Optional[str] = None, chat_session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get active streaming connections.

        Args:
            user_id: Optional filter by user
            chat_session_id: Optional filter by chat session

        Returns:
            List of active streaming sessions
        """
        try:
            query = self.client.table("streaming_sessions").select("*").eq("status", "active")

            if user_id:
                query = query.eq("user_id", user_id)
            if chat_session_id:
                query = query.eq("chat_session_id", chat_session_id)

            result = query.execute()
            return result.data or []

        except Exception as e:
            logger.error(f"Error getting active connections: {e}")
            return []

    def get_streaming_session(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """
        Get streaming session by connection ID.

        Args:
            connection_id: Connection identifier

        Returns:
            Streaming session record or None
        """
        try:
            result = self.client.table("streaming_sessions").select("*").eq("connection_id", connection_id).execute()

            if result.data and len(result.data) > 0:
                return result.data[0]
            return None

        except Exception as e:
            logger.error(f"Error getting streaming session: {e}")
            return None
