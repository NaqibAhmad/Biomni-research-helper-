"""
User Settings Service for Biomni

Manages user preferences and settings in Supabase database.
Handles LLM preferences, tool preferences, UI preferences, and research preferences.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from supabase import Client, create_client

from utils.user_utils import ensure_user_exists

logger = logging.getLogger(__name__)


class UserSettingsService:
    """Service for managing user settings and preferences."""

    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """
        Initialize the User Settings Service.

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
            logger.info("User settings service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize user settings service: {e}")
            raise

    def get_default_settings(self) -> Dict[str, Any]:
        """Get default user settings structure."""
        return {
            "llm_preferences": {
                "model": "claude-sonnet-4-20250514",
                "source": "Anthropic",
                "temperature": 0.7,
                "max_tokens": 8192,
            },
            "tool_preferences": {
                "use_tool_retriever": True,
                "preferred_modules": [],
            },
            "ui_preferences": {
                "theme": "light",
                "font_size": "medium",
            },
            "research_preferences": {
                "domains": [],
                "languages": ["en"],
            },
        }

    def get_user_settings(self, user_id: str, create_if_missing: bool = True) -> Dict[str, Any]:
        """
        Get user settings, creating default settings if they don't exist.

        Args:
            user_id: User identifier
            create_if_missing: Whether to create default settings if not found

        Returns:
            User settings record

        Raises:
            ValueError: If user doesn't exist
        """
        if not user_id or user_id == "anonymous":
            raise ValueError("User settings require authenticated user")

        if not ensure_user_exists(self.client, user_id):
            raise ValueError(
                f"User {user_id} not found in public.users. "
                f"Please ensure you are properly authenticated."
            )

        try:
            # Try to get existing settings
            result = (
                self.client.table("user_settings")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )

            if result.data and len(result.data) > 0:
                settings = result.data[0]
                # Ensure all preference fields exist
                defaults = self.get_default_settings()
                for key in ["llm_preferences", "tool_preferences", "ui_preferences", "research_preferences"]:
                    if not settings.get(key):
                        settings[key] = defaults[key]
                return settings

            # Create default settings if not found and create_if_missing is True
            if create_if_missing:
                return self.create_default_settings(user_id)
            else:
                return None

        except Exception as e:
            logger.error(f"Error getting user settings: {e}")
            raise

    def create_default_settings(self, user_id: str) -> Dict[str, Any]:
        """
        Create default settings for a user.

        Args:
            user_id: User identifier

        Returns:
            Created settings record
        """
        defaults = self.get_default_settings()

        settings_data = {
            "user_id": user_id,
            "llm_preferences": defaults["llm_preferences"],
            "tool_preferences": defaults["tool_preferences"],
            "ui_preferences": defaults["ui_preferences"],
            "research_preferences": defaults["research_preferences"],
        }

        try:
            result = self.client.table("user_settings").insert(settings_data).execute()

            if not result.data or len(result.data) == 0:
                raise Exception("Failed to create user settings: No data returned")

            logger.info(f"Created default settings for user {user_id}")
            return result.data[0]

        except Exception as e:
            logger.error(f"Error creating default settings: {e}")
            raise

    def update_user_settings(
        self, user_id: str, settings_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update user settings (partial update supported).

        Args:
            user_id: User identifier
            settings_updates: Dictionary with settings to update
                Can include: llm_preferences, tool_preferences, ui_preferences, research_preferences

        Returns:
            Updated settings record
        """
        # Ensure settings exist
        self.get_user_settings(user_id, create_if_missing=True)

        # Prepare update data
        update_data = {}
        for key in ["llm_preferences", "tool_preferences", "ui_preferences", "research_preferences"]:
            if key in settings_updates:
                update_data[key] = settings_updates[key]

        if not update_data:
            raise ValueError("No valid settings to update")

        update_data["updated_at"] = datetime.now().isoformat()

        try:
            result = (
                self.client.table("user_settings")
                .update(update_data)
                .eq("user_id", user_id)
                .execute()
            )

            if not result.data or len(result.data) == 0:
                raise Exception("Failed to update user settings: No data returned")

            logger.info(f"Updated settings for user {user_id}")
            return result.data[0]

        except Exception as e:
            logger.error(f"Error updating user settings: {e}")
            raise

    def update_llm_preferences(self, user_id: str, llm_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update LLM preferences.

        Args:
            user_id: User identifier
            llm_preferences: LLM preferences dictionary
                Can include: model, source, temperature, max_tokens

        Returns:
            Updated settings record
        """
        return self.update_user_settings(user_id, {"llm_preferences": llm_preferences})

    def update_tool_preferences(self, user_id: str, tool_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update tool preferences.

        Args:
            user_id: User identifier
            tool_preferences: Tool preferences dictionary
                Can include: use_tool_retriever, preferred_modules

        Returns:
            Updated settings record
        """
        return self.update_user_settings(user_id, {"tool_preferences": tool_preferences})

    def update_ui_preferences(self, user_id: str, ui_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update UI preferences.

        Args:
            user_id: User identifier
            ui_preferences: UI preferences dictionary
                Can include: theme, font_size

        Returns:
            Updated settings record
        """
        return self.update_user_settings(user_id, {"ui_preferences": ui_preferences})

    def update_research_preferences(
        self, user_id: str, research_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update research preferences.

        Args:
            user_id: User identifier
            research_preferences: Research preferences dictionary
                Can include: domains, languages

        Returns:
            Updated settings record
        """
        return self.update_user_settings(user_id, {"research_preferences": research_preferences})

    def get_llm_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get LLM preferences for a user."""
        settings = self.get_user_settings(user_id)
        return settings.get("llm_preferences", self.get_default_settings()["llm_preferences"])

    def get_tool_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get tool preferences for a user."""
        settings = self.get_user_settings(user_id)
        return settings.get("tool_preferences", self.get_default_settings()["tool_preferences"])

