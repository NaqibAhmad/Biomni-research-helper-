"""
Prompt Library Service for Biomni

Manages user-created prompt templates with tool bindings and model configurations.
Simple, focused service for personal prompt management.
"""

import logging
import os
import re
from datetime import datetime
from typing import Any
from uuid import UUID

try:
    from supabase import Client, create_client
except ImportError:
    Client = None
    create_client = None

try:
    from utils.user_utils import ensure_user_exists, ensure_session_exists
except ImportError:
    # Fallback if utils not available
    def ensure_user_exists(client, user_id: str) -> bool:
        return False

    def ensure_session_exists(client, session_id: str) -> bool:
        return False


logger = logging.getLogger(__name__)


class PromptLibraryService:
    """Service for managing the prompt library with Supabase integration."""

    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """
        Initialize the Prompt Library Service.

        Args:
            supabase_url: Supabase project URL (defaults to SUPABASE_URL env var)
            supabase_key: Supabase service role key (defaults to SUPABASE_SERVICE_ROLE_KEY env var)
        """
        if Client is None or create_client is None:
            raise ImportError("Supabase client is not installed. Install with: pip install supabase")

        self.url = supabase_url or os.getenv("SUPABASE_URL")
        self.key = supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not self.url or not self.key:
            raise ValueError(
                "Supabase URL and key must be provided either as parameters "
                "or via SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables"
            )

        self.client: Client = create_client(self.url, self.key)

    # ==================== PROMPT CRUD OPERATIONS ====================

    async def create_prompt(
        self,
        title: str,
        prompt_template: str,
        category: str,
        created_by: str,
        description: str = None,
        tags: list[str] = None,
        system_prompt: str = None,
        variables: list[dict[str, Any]] = None,
        model_config: dict[str, Any] = None,
        tool_bindings: dict[str, Any] = None,
        output_template: dict[str, Any] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Create a new user prompt template.

        Args:
            title: Prompt title
            prompt_template: The prompt template with optional {variable} placeholders
            category: Category (genomics, protein_analysis, drug_discovery, etc.)
            created_by: User ID who created the prompt
            description: Optional description
            tags: Optional list of tags
            system_prompt: Optional system prompt
            variables: List of variable definitions [{"name": "gene", "type": "string", "description": "...", "default": ""}]
            model_config: Model configuration {"model": "...", "temperature": 0.7, ...}
            tool_bindings: Tool bindings {"enabled_modules": [...], "specific_tools": [...]}
            output_template: Output format {"format": "markdown", "schema": {}, ...}

        Returns:
            Created prompt data
        """
        # Don't allow anonymous users to create prompts (database expects UUID)
        if created_by is None or created_by == "anonymous":
            raise ValueError("Anonymous users cannot create prompts. Please authenticate.")

        # Ensure user exists in public.users before creating prompt
        if not ensure_user_exists(self.client, created_by):
            raise ValueError(
                f"User {created_by} not found in public.users. "
                f"Cannot create prompt due to foreign key constraint. "
                f"Please ensure you are properly authenticated."
            )

        prompt_data = {
            "title": title,
            "prompt_template": prompt_template,
            "category": category,
            "created_by": created_by,
            "updated_by": created_by,
            "description": description,
            "tags": tags or [],
            "system_prompt": system_prompt,
            "variables": variables or [],
            "created_at": datetime.now().isoformat(),
            **kwargs,
        }

        # Add optional configurations
        if model_config:
            prompt_data["model_config"] = model_config
        if tool_bindings:
            prompt_data["tool_bindings"] = tool_bindings
        if output_template:
            prompt_data["output_template"] = output_template

        result = self.client.table("prompt_library").insert(prompt_data).execute()
        return result.data[0] if result.data else None

    async def get_prompt_by_id(self, prompt_id: str) -> dict[str, Any] | None:
        """Get a prompt by its ID."""
        result = self.client.table("prompt_library").select("*").eq("id", prompt_id).eq("is_active", True).execute()
        return result.data[0] if result.data else None

    async def update_prompt(
        self,
        prompt_id: str,
        updated_by: str,
        create_version: bool = False,
        **update_fields,
    ) -> dict[str, Any]:
        """
        Update an existing prompt.

        Args:
            prompt_id: ID of the prompt to update
            updated_by: User ID who is updating
            create_version: If True, creates a new version instead of updating
            **update_fields: Fields to update

        Returns:
            Updated prompt data
        """
        if create_version:
            # Create a new version by copying the old prompt
            old_prompt = await self.get_prompt_by_id(prompt_id)
            if not old_prompt:
                raise ValueError(f"Prompt {prompt_id} not found")

            new_prompt_data = {**old_prompt}
            new_prompt_data.update(update_fields)
            new_prompt_data["parent_id"] = prompt_id
            new_prompt_data["version"] = old_prompt.get("version", 1) + 1
            new_prompt_data["updated_by"] = updated_by
            new_prompt_data["created_by"] = updated_by
            # Remove the old ID so a new one is generated
            new_prompt_data.pop("id", None)
            new_prompt_data.pop("created_at", None)
            new_prompt_data.pop("updated_at", None)

            result = self.client.table("prompt_library").insert(new_prompt_data).execute()
            return result.data[0] if result.data else None
        else:
            # Regular update
            update_fields["updated_by"] = updated_by
            result = self.client.table("prompt_library").update(update_fields).eq("id", prompt_id).execute()
            return result.data[0] if result.data else None

    async def delete_prompt(self, prompt_id: str, soft_delete: bool = True) -> bool:
        """
        Delete a prompt.

        Args:
            prompt_id: ID of the prompt to delete
            soft_delete: If True, marks as inactive instead of deleting

        Returns:
            True if successful
        """
        if soft_delete:
            result = self.client.table("prompt_library").update({"is_active": False}).eq("id", prompt_id).execute()
        else:
            result = self.client.table("prompt_library").delete().eq("id", prompt_id).execute()

        return len(result.data) > 0

    # ==================== PROMPT DISCOVERY & SEARCH ====================

    async def list_prompts(
        self,
        user_id: str,
        category: str = None,
        tags: list[str] = None,
        search_query: str = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "created_at",
        descending: bool = True,
    ) -> list[dict[str, Any]]:
        """
        List user's prompts with various filters.

        Args:
            user_id: User ID (required - only returns user's own prompts)
            category: Filter by category
            tags: Filter by tags (prompts containing any of these tags)
            search_query: Search in title and description
            limit: Maximum number of results
            offset: Offset for pagination
            order_by: Field to order by
            descending: Sort order

        Returns:
            List of user's prompt data
        """
        # For authenticated users, we need to query for prompts where:
        # 1. created_by = user_id (user's own prompts)
        # 2. created_by IS NULL (prompts created before user setup or with NULL created_by)
        # Since Supabase Python client doesn't easily support OR with IS NULL, we'll make two queries and combine

        if user_id is None or user_id == "anonymous":
            # Anonymous users only see prompts with NULL created_by
            query = self.client.table("prompt_library").select("*").eq("is_active", True).is_("created_by", "null")

        # Apply filters
        if category:
            query = query.eq("category", category)

        if tags:
            query = query.contains("tags", tags)

        if search_query:
            query = query.or_(f"title.ilike.%{search_query}%,description.ilike.%{search_query}%")

            query = query.order(order_by, desc=descending).limit(limit).range(offset, offset + limit - 1)
            result = query.execute()
            return result.data or []
        else:
            # Authenticated users: fetch prompts where created_by = user_id OR created_by IS NULL
            logger.info(f"Querying prompts for user {user_id} (including NULL created_by prompts)")

            # Build base query for user's own prompts
            user_query = self.client.table("prompt_library").select("*").eq("is_active", True).eq("created_by", user_id)

            # Build query for NULL created_by prompts
            null_query = self.client.table("prompt_library").select("*").eq("is_active", True).is_("created_by", "null")

            # Apply filters to both queries
            if category:
                user_query = user_query.eq("category", category)
                null_query = null_query.eq("category", category)

            if tags:
                user_query = user_query.contains("tags", tags)
                null_query = null_query.contains("tags", tags)

            if search_query:
                user_query = user_query.or_(f"title.ilike.%{search_query}%,description.ilike.%{search_query}%")
                null_query = null_query.or_(f"title.ilike.%{search_query}%,description.ilike.%{search_query}%")

            # Execute both queries
            user_result = user_query.execute()
            null_result = null_query.execute()

            # Combine results
            user_prompts = user_result.data or []
            null_prompts = null_result.data or []

            # Combine and deduplicate by ID
            all_prompts = {}
            for prompt in user_prompts + null_prompts:
                all_prompts[prompt.get("id")] = prompt

            # Convert back to list and sort
            combined_prompts = list(all_prompts.values())

            # Sort by order_by field
            reverse = descending
            if order_by == "created_at":
                combined_prompts.sort(key=lambda x: x.get("created_at", ""), reverse=reverse)
            elif order_by == "updated_at":
                combined_prompts.sort(key=lambda x: x.get("updated_at", ""), reverse=reverse)
            elif order_by == "title":
                combined_prompts.sort(key=lambda x: x.get("title", ""), reverse=reverse)

            # Apply pagination
            paginated_prompts = combined_prompts[offset : offset + limit]

            logger.info(
                f"Found {len(user_prompts)} user prompts and {len(null_prompts)} NULL prompts. Total: {len(paginated_prompts)} after pagination"
            )
            if paginated_prompts:
                logger.debug(f"Prompt IDs found: {[p.get('id') for p in paginated_prompts]}")
                created_by_values = [p.get("created_by") for p in paginated_prompts]
                logger.debug(f"Created_by values in results: {created_by_values}")

            return paginated_prompts

    async def get_user_prompts(self, user_id: str) -> list[dict[str, Any]]:
        """
        Get all prompts for a specific user.

        Args:
            user_id: User ID

        Returns:
            List of user's prompts
        """
        query = self.client.table("prompt_library").select("*")

        # Handle NULL user_id or "anonymous" string (database expects UUID, not string)
        if user_id is None or user_id == "anonymous":
            query = query.is_("created_by", "null")
        else:
            query = query.eq("created_by", user_id)

        result = query.eq("is_active", True).order("created_at", desc=True).execute()
        return result.data or []

    # ==================== PROMPT EXECUTION ====================

    async def render_prompt(self, prompt_id: str, variables: dict[str, Any] = None) -> str:
        """
        Render a prompt template with variables.

        Args:
            prompt_id: ID of the prompt
            variables: Dictionary of variable values

        Returns:
            Rendered prompt text
        """
        prompt = await self.get_prompt_by_id(prompt_id)
        if not prompt:
            raise ValueError(f"Prompt {prompt_id} not found")

        template = prompt.get("prompt_template", "")
        variables = variables or {}

        # Variable substitution using {variable_name} format
        # Replace ALL occurrences of each variable
        rendered = template

        if variables:
            logger.info(f"Rendering prompt {prompt_id} with {len(variables)} variables: {list(variables.keys())}")

        for var_name, var_value in variables.items():
            if var_value is None or var_value == "":
                logger.warning(f"Variable '{var_name}' has empty value, skipping substitution")
                continue

            # Escape special regex characters in variable name for safe replacement
            # But use simple string replacement for exact matching
            placeholder = f"{{{var_name}}}"
            replacement = str(var_value)

            # Count occurrences before replacement
            count_before = rendered.count(placeholder)

            # Replace all occurrences (str.replace replaces all by default)
            rendered = rendered.replace(placeholder, replacement)

            # Count occurrences after replacement
            count_after = rendered.count(placeholder)

            if count_before > 0:
                logger.info(f"Replaced {count_before} occurrence(s) of '{placeholder}' with '{replacement[:50]}...'")
            elif count_before == 0 and var_name in template:
                logger.warning(f"Variable '{var_name}' defined but placeholder '{placeholder}' not found in template")

            # Check for any remaining placeholders
            import re

            remaining_placeholders = re.findall(r"\{([^}]+)\}", rendered)
            if remaining_placeholders:
                logger.warning(f"Template still contains unsubstituted placeholders: {remaining_placeholders}")

        logger.debug(f"Template (first 300 chars): {template[:300]}")
        logger.debug(f"Rendered (first 300 chars): {rendered[:300]}")

        return rendered

    async def execute_prompt(
        self,
        prompt_id: str,
        user_id: str,
        variables: dict[str, Any] = None,
        session_id: str = None,
        override_model_config: dict[str, Any] = None,
    ) -> str | None:
        """
        Execute a prompt template and log the execution.

        Args:
            prompt_id: ID of the prompt to execute
            user_id: User executing the prompt
            variables: Variables to render the prompt with
            session_id: Optional session ID
            override_model_config: Optional model config override

        Returns:
            Execution ID for tracking
        """
        prompt = await self.get_prompt_by_id(prompt_id)
        if not prompt:
            raise ValueError(f"Prompt {prompt_id} not found")

        # Render the prompt
        rendered_prompt = await self.render_prompt(prompt_id, variables)

        # Get model config (use override if provided)
        model_config = override_model_config or prompt.get("model_config", {})

        # Log the execution (skip for anonymous users since database expects UUID)
        execution_id = None
        if user_id and user_id != "anonymous":
            # Validate user exists (required for authenticated users)
            if not ensure_user_exists(self.client, user_id):
                raise ValueError(
                    f"User {user_id} not found in public.users. "
                    f"Cannot log execution. Please ensure you are properly authenticated."
                )

        execution_data = {
            "prompt_id": prompt_id,
            "user_id": user_id,
            "variables_used": variables or {},
            "rendered_prompt": rendered_prompt,
            "model_used": model_config.get("model"),
            "created_at": datetime.now().isoformat(),
        }

        # Validate session exists if provided
        if session_id:
            if not ensure_session_exists(self.client, session_id):
                raise ValueError(
                    f"Session {session_id} not found in database. Cannot log execution with invalid session."
                )
            execution_data["session_id"] = session_id

            try:
                result = self.client.table("prompt_executions").insert(execution_data).execute()
                execution_id = result.data[0]["id"] if result.data else None
            except Exception as e:
                logger.error(f"Error logging prompt execution: {e}")
                raise

        return execution_id

    async def update_execution_result(
        self,
        execution_id: str,
        response: str = None,
        execution_log: list[str] = None,
        tools_used: list[str] = None,
        processing_time_ms: int = None,
        token_usage: dict[str, int] = None,
        cost_estimate: float = None,
    ) -> dict[str, Any]:
        """
        Update an execution record with results.

        Args:
            execution_id: Execution ID
            response: Agent response
            execution_log: Execution log
            tools_used: List of tools used
            processing_time_ms: Processing time in milliseconds
            token_usage: Token usage stats
            cost_estimate: Estimated cost

        Returns:
            Updated execution data
        """
        update_data = {}
        if response is not None:
            update_data["response"] = response
        if execution_log is not None:
            update_data["execution_log"] = execution_log
        if tools_used is not None:
            update_data["tools_used"] = tools_used
        if processing_time_ms is not None:
            update_data["processing_time_ms"] = processing_time_ms
        if token_usage is not None:
            update_data["token_usage"] = token_usage
        if cost_estimate is not None:
            update_data["cost_estimate"] = cost_estimate

        result = self.client.table("prompt_executions").update(update_data).eq("id", execution_id).execute()
        return result.data[0] if result.data else None

    # ==================== FAVORITES ====================

    async def add_favorite(self, user_id: str, prompt_id: str) -> dict[str, Any]:
        """Add a prompt to user's favorites."""
        # Don't allow anonymous users to add favorites (database expects UUID)
        if user_id is None or user_id == "anonymous":
            raise ValueError("Anonymous users cannot add favorites. Please authenticate.")

        # Ensure user exists in public.users before adding favorite
        if not ensure_user_exists(self.client, user_id):
            raise ValueError(
                f"User {user_id} not found in public.users. "
                f"Cannot add favorite due to foreign key constraint. "
                f"Please ensure you are properly authenticated."
            )

        favorite_data = {
            "user_id": user_id,
            "prompt_id": prompt_id,
            "created_at": datetime.now().isoformat(),
        }
        result = self.client.table("prompt_favorites").insert(favorite_data).execute()
        return result.data[0] if result.data else None

    async def remove_favorite(self, user_id: str, prompt_id: str) -> bool:
        """Remove a prompt from user's favorites."""
        # Don't allow anonymous users to remove favorites
        if user_id is None or user_id == "anonymous":
            raise ValueError("Anonymous users cannot remove favorites. Please authenticate.")

        query = self.client.table("prompt_favorites").delete()
        query = query.eq("user_id", user_id)
        result = query.eq("prompt_id", prompt_id).execute()
        return len(result.data) > 0

    async def get_user_favorites(self, user_id: str) -> list[dict[str, Any]]:
        """Get all prompts favorited by a user."""
        # Anonymous users have no favorites (database expects UUID)
        if user_id is None or user_id == "anonymous":
            return []

        # Get favorite IDs
        query = self.client.table("prompt_favorites").select("prompt_id")
        query = query.eq("user_id", user_id)
        favorites = query.execute()

        if not favorites.data:
            return []

        # Get the actual prompts
        prompt_ids = [fav["prompt_id"] for fav in favorites.data]
        result = self.client.table("prompt_library").select("*").in_("id", prompt_ids).eq("is_active", True).execute()
        return result.data or []

    # ==================== VERSIONING ====================

    async def get_prompt_versions(self, prompt_id: str) -> list[dict[str, Any]]:
        """Get all versions of a prompt (including parent and children)."""
        # Get the prompt
        prompt = await self.get_prompt_by_id(prompt_id)
        if not prompt:
            return []

        # Find the root parent
        root_id = prompt_id
        if prompt.get("parent_id"):
            root_id = prompt["parent_id"]

        # Get all versions with this parent or are this parent
        result = (
            self.client.table("prompt_library")
            .select("*")
            .or_(f"id.eq.{root_id},parent_id.eq.{root_id}")
            .order("version", desc=False)
            .execute()
        )
        return result.data or []

    # ==================== ANALYTICS ====================

    async def get_prompt_analytics(self, prompt_id: str) -> dict[str, Any]:
        """Get analytics for a specific prompt."""
        prompt = await self.get_prompt_by_id(prompt_id)
        if not prompt:
            return {}

        # Get execution statistics
        executions = self.client.table("prompt_executions").select("*").eq("prompt_id", prompt_id).execute()

        exec_data = executions.data or []
        total_executions = len(exec_data)

        # Calculate average metrics
        avg_processing_time = 0
        avg_quality_rating = 0
        total_tokens = 0

        if exec_data:
            processing_times = [e.get("processing_time_ms", 0) for e in exec_data if e.get("processing_time_ms")]
            if processing_times:
                avg_processing_time = sum(processing_times) / len(processing_times)

            ratings = [e.get("quality_rating") for e in exec_data if e.get("quality_rating")]
            if ratings:
                avg_quality_rating = sum(ratings) / len(ratings)

            for e in exec_data:
                token_usage = e.get("token_usage", {})
                if isinstance(token_usage, dict):
                    total_tokens += token_usage.get("total", 0)

        # Get favorites count
        favorites = (
            self.client.table("prompt_favorites").select("id", count="exact").eq("prompt_id", prompt_id).execute()
        )
        favorites_count = len(favorites.data) if favorites.data else 0

        return {
            "prompt_id": prompt_id,
            "title": prompt.get("title"),
            "usage_count": prompt.get("usage_count", 0),
            "total_executions": total_executions,
            "favorites_count": favorites_count,
            "avg_processing_time_ms": avg_processing_time,
            "avg_quality_rating": avg_quality_rating,
            "total_tokens_used": total_tokens,
            "last_used_at": prompt.get("last_used_at"),
        }

    async def get_user_prompt_stats(self, user_id: str) -> dict[str, Any]:
        """Get basic prompt statistics for a user."""
        # Anonymous users have no stats (database expects UUID)
        if user_id is None or user_id == "anonymous":
            return {
                "user_id": user_id,
                "total_prompts": 0,
                "total_executions": 0,
                "total_favorites": 0,
            }

        # Get total prompts
        prompts = await self.get_user_prompts(user_id)
        total_prompts = len(prompts)

        # Get total executions
        exec_query = self.client.table("prompt_executions").select("id", count="exact")
        exec_query = exec_query.eq("user_id", user_id)
        executions = exec_query.execute()
        total_executions = len(executions.data) if executions.data else 0

        # Get favorites
        fav_query = self.client.table("prompt_favorites").select("id", count="exact")
        fav_query = fav_query.eq("user_id", user_id)
        favorites = fav_query.execute()
        total_favorites = len(favorites.data) if favorites.data else 0

        return {
            "user_id": user_id,
            "total_prompts": total_prompts,
            "total_executions": total_executions,
            "total_favorites": total_favorites,
        }
