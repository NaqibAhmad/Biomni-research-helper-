"""
Chat Service for Biomni

Manages chat sessions and message persistence to Supabase database.
Provides conversation history and session tracking capabilities.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from supabase import Client, create_client

from backend.utils.user_utils import ensure_user_exists, ensure_session_exists

logger = logging.getLogger(__name__)


class ChatService:
    """Service for managing chat sessions and message history."""

    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """
        Initialize the Chat Service.

        Args:
            supabase_url: Supabase project URL (defaults to SUPABASE_URL env var)
            supabase_key: Supabase service role key (defaults to SUPABASE_SERVICE_ROLE_KEY env var)

        Raises:
            ImportError: If Supabase client is not installed
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
            logger.info("Chat service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize chat service: {e}")
            raise

    def create_session(
        self, session_id: str, user_id: str = "anonymous", metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a new chat session.

        IMPORTANT: Only authenticated users can create sessions. Anonymous users are rejected.

        Args:
            session_id: Unique session identifier
            user_id: User identifier (must be authenticated, not "anonymous")
            metadata: Optional session metadata (model, settings, etc.)

        Returns:
            Created session data

        Raises:
            ValueError: If user is anonymous or not found
            Exception: If session creation fails
        """
        try:
            # Reject anonymous users - only authenticated users can create sessions
            if not user_id or user_id == "anonymous":
                raise ValueError("Anonymous users cannot create chat sessions. Please sign in to continue.")

            # Validate user exists in public.users before creating session
            if not ensure_user_exists(self.client, user_id):
                raise ValueError(
                    f"User {user_id} not found in public.users. "
                    f"Please ensure you are properly authenticated and your account is set up."
                )

            session_data = {
                "id": session_id,
                "user_id": user_id,  # Required - no anonymous sessions
                "metadata": metadata or {},
                "start_time": datetime.now().isoformat(),
                "last_activity_time": datetime.now().isoformat(),
                "message_count": 0,
                "is_active": True,
            }

            logger.debug(f"[CHAT_SERVICE] Inserting session data: {session_data}")
            result = self.client.table("chat_sessions").insert(session_data).execute()

            if result.data and len(result.data) > 0:
                logger.info(
                    f"[CHAT_SERVICE] Created chat session: {session_id} for user: {user_id}, session_record_id: {result.data[0].get('id')}"
                )
                return result.data[0]
            else:
                logger.error(f"[CHAT_SERVICE] Failed to create session: No data returned from insert. Result: {result}")
                raise Exception("Failed to create session: No data returned")

        except Exception as e:
            logger.error(f"[CHAT_SERVICE] Error creating session {session_id}: {e}", exc_info=True)
            raise

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a chat session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session data or None if not found
        """
        try:
            result = self.client.table("chat_sessions").select("*").eq("id", session_id).execute()

            if result.data and len(result.data) > 0:
                return result.data[0]
            return None

        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            return None

    def get_or_create_session(
        self, session_id: str, user_id: str = "anonymous", metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Get existing session or create if it doesn't exist.

        Args:
            session_id: Session identifier
            user_id: User identifier
            metadata: Optional session metadata

        Returns:
            Session data
        """
        logger.debug(f"[CHAT_SERVICE] get_or_create_session called: session_id={session_id}, user_id={user_id}")
        session = self.get_session(session_id)
        if session:
            logger.debug(f"[CHAT_SERVICE] Found existing session: {session_id}")
            return session

        logger.debug(f"[CHAT_SERVICE] Session not found, creating new session: {session_id}")
        return self.create_session(session_id, user_id, metadata)

    def store_message(
        self,
        session_id: str,
        message_type: str,
        content: str,
        model_used: str = None,
        metadata: Dict[str, Any] = None,
        tokens_used: int = None,
        processing_time_ms: int = None,
        user_id: str = None,
    ) -> Dict[str, Any]:
        """
        Store a chat message.

        IMPORTANT: Only authenticated users can store messages. Anonymous users are rejected.

        Args:
            session_id: Session identifier
            message_type: Message type ('user' or 'assistant')
            content: Message content
            model_used: LLM model used for generation (for assistant messages)
            metadata: Optional message metadata (tool calls, etc.)
            tokens_used: Token count
            processing_time_ms: Processing time in milliseconds
            user_id: User ID for the message (required, must be authenticated)

        Returns:
            Created message data

        Raises:
            ValueError: If user is anonymous or session doesn't exist
            Exception: If message storage fails
        """
        try:
            # Reject anonymous users - only authenticated users can store messages
            if not user_id or user_id == "anonymous":
                raise ValueError("Anonymous users cannot store messages. Please sign in to continue.")

            # Verify session exists before storing message
            session = self.get_session(session_id)
            if not session:
                raise ValueError(
                    f"Session {session_id} not found in database. Cannot store message due to foreign key constraint."
                )

            # Verify the session belongs to the user
            if session.get("user_id") != user_id:
                raise ValueError(f"Session {session_id} does not belong to user {user_id}. Cannot store message.")

            # Validate user exists
            if not ensure_user_exists(self.client, user_id):
                raise ValueError(
                    f"User {user_id} not found in public.users. "
                    f"Cannot store message. Please ensure you are properly authenticated."
                )

            message_data = {
                "session_id": session_id,
                "user_id": user_id,  # Required - no anonymous messages
                "message_type": message_type,
                "content": content,
                "model_used": model_used,
                "metadata": metadata or {},
                "tokens_used": tokens_used,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now().isoformat(),
            }

            logger.debug(f"[CHAT_SERVICE] Inserting message data for session {session_id}, type: {message_type}")
            result = self.client.table("chat_messages").insert(message_data).execute()

            if result.data and len(result.data) > 0:
                message_id = result.data[0].get("id")
                logger.info(
                    f"[CHAT_SERVICE] Stored {message_type} message for session {session_id} (user: {user_id}), message_id: {message_id}"
                )
                return result.data[0]
            else:
                logger.error(f"[CHAT_SERVICE] Failed to store message: No data returned from insert. Result: {result}")
                raise Exception("Failed to store message: No data returned")

        except Exception as e:
            logger.error(f"[CHAT_SERVICE] Error storing message for session {session_id}: {e}", exc_info=True)
            raise

    def get_session_messages(self, session_id: str, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieve all messages for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            offset: Number of messages to skip

        Returns:
            List of message data ordered by timestamp
        """
        try:
            query = (
                self.client.table("chat_messages")
                .select("*")
                .eq("session_id", session_id)
                .order("timestamp", desc=False)
            )

            if limit:
                query = query.limit(limit)

            if offset:
                query = query.range(offset, offset + (limit or 100) - 1)

            result = query.execute()
            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Error retrieving messages for session {session_id}: {e}")
            return []

    def update_session_metadata(
        self, session_id: str, message_count: int = None, metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Update session metadata and activity time.

        Args:
            session_id: Session identifier
            message_count: New message count
            metadata: Metadata to merge with existing

        Returns:
            True if update succeeded, False otherwise
        """
        try:
            update_data = {"last_activity_time": datetime.now().isoformat()}

            if message_count is not None:
                update_data["message_count"] = message_count

            if metadata:
                session = self.get_session(session_id)
                if session:
                    existing_metadata = session.get("metadata", {})
                    existing_metadata.update(metadata)
                    update_data["metadata"] = existing_metadata

            result = self.client.table("chat_sessions").update(update_data).eq("id", session_id).execute()

            return len(result.data) > 0 if result.data else False

        except Exception as e:
            logger.error(f"Error updating session {session_id}: {e}")
            return False

    def list_user_sessions(
        self, user_id: str, limit: int = 50, offset: int = 0, active_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List all sessions for a user.

        Args:
            user_id: User identifier
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip
            active_only: Only return active sessions

        Returns:
            List of session data ordered by last activity
        """
        try:
            # Reject anonymous users - only authenticated users can list sessions
            if user_id is None or user_id == "anonymous":
                logger.info(f"Rejecting session list request from anonymous user")
                return []

            query = self.client.table("chat_sessions").select("*")
            query = query.eq("user_id", user_id)

            query = query.order("last_activity_time", desc=True).limit(limit)

            if active_only:
                query = query.eq("is_active", True)

            if offset:
                query = query.range(offset, offset + limit - 1)

            result = query.execute()
            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Error listing sessions for user {user_id}: {e}", exc_info=True)
            return []

    def end_session(self, session_id: str) -> bool:
        """
        Mark a session as inactive.

        Args:
            session_id: Session identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            result = (
                self.client.table("chat_sessions")
                .update({"is_active": False, "end_time": datetime.now().isoformat()})
                .eq("id", session_id)
                .execute()
            )

            return len(result.data) > 0 if result.data else False

        except Exception as e:
            logger.error(f"Error ending session {session_id}: {e}")
            return False

    def delete_session(self, session_id: str, user_id: str = None) -> bool:
        """
        Delete a session and all its messages.

        Args:
            session_id: Session identifier
            user_id: User identifier for authorization check

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete messages first due to foreign key constraint
            self.client.table("chat_messages").delete().eq("session_id", session_id).execute()

            # Delete session
            query = self.client.table("chat_sessions").delete().eq("id", session_id)

            if user_id is not None:
                query = query.eq("user_id", user_id)

            result = query.execute()

            if result.data and len(result.data) > 0:
                logger.info(f"Deleted session {session_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
