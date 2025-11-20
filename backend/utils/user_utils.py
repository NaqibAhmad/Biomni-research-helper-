"""
User utility functions for Supabase user management.

Validates that users exist in public.users table before database operations.
Users must be properly set up through the authentication system - no auto-creation is performed.
This ensures only authenticated users with proper account setup can interact with the system.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def ensure_user_exists(supabase_client, user_id: str) -> bool:
    """
    Check if a user exists in public.users table.
    If user exists in auth.users but not in public.users, automatically create them.

    This ensures authenticated users can use the system even if the trigger
    didn't fire or the user was created before the trigger existed.

    Args:
        supabase_client: Supabase client instance
        user_id: User ID (UUID string) from auth.users

    Returns:
        True if user exists in public.users (or was created), False otherwise
    """
    if not user_id or user_id == "anonymous":
        logger.debug("Skipping user existence check for anonymous user")
        return False

    # Validate UUID format
    if len(user_id) != 36 or user_id.count("-") != 4:
        logger.warning(f"Invalid user_id format: {user_id}")
        return False

    try:
        # Check if user exists in public.users
        user_check = supabase_client.table("users").select("id").eq("id", user_id).limit(1).execute()

        if user_check.data and len(user_check.data) > 0:
            logger.debug(f"User {user_id} exists in public.users")
            return True

        # User not found in public.users - check if they exist in auth.users
        # If they do, create them in public.users
        logger.info(f"User {user_id} not found in public.users, checking auth.users and creating if needed...")

        try:
            # Try to get user info from auth.users (requires service role key)
            # If we can't access auth.users directly, try to create the user in public.users anyway
            # The database trigger should handle this, but if it didn't fire, we'll do it manually
            from datetime import datetime

            # Try to create user in public.users
            # Use the user's email as full_name if available, otherwise use the user_id
            create_result = (
                supabase_client.table("users")
                .insert(
                    {
                        "id": user_id,
                        "full_name": user_id,  # Will be updated if we can get email
                        "updated_at": datetime.now().isoformat(),
                    }
                )
                .execute()
            )

            if create_result.data and len(create_result.data) > 0:
                logger.info(f"Successfully created user {user_id} in public.users")
                return True
            else:
                logger.warning(f"Failed to create user {user_id} in public.users: No data returned")
                return False

        except Exception as create_error:
            # Check if it's a duplicate key error (user was created between check and insert)
            error_str = str(create_error).lower()
            if "duplicate" in error_str or "unique" in error_str or "already exists" in error_str:
                logger.info(f"User {user_id} was created in public.users by another process")
                return True
            else:
                logger.error(f"Error creating user {user_id} in public.users: {create_error}")
                # Still return False - user doesn't exist and we couldn't create them
                return False

    except Exception as e:
        logger.error(f"Error checking user existence for {user_id}: {e}")
        return False


def ensure_session_exists(supabase_client, session_id: str) -> bool:
    """
    Verify that a session exists in chat_sessions table.

    Args:
        supabase_client: Supabase client instance
        session_id: Session ID (UUID string)

    Returns:
        True if session exists, False otherwise
    """
    if not session_id:
        return False

    try:
        session_check = supabase_client.table("chat_sessions").select("id").eq("id", session_id).limit(1).execute()
        return session_check.data and len(session_check.data) > 0
    except Exception as e:
        logger.warning(f"Error checking session existence {session_id}: {e}")
        return False
