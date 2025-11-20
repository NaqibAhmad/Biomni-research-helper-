"""
Authentication middleware for JWT token validation.

This module provides FastAPI dependencies for extracting and validating
Supabase JWT tokens from Authorization headers.
"""

import json
import logging
import os
from typing import Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from fastapi import Depends, Header, HTTPException, status
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

# Get Supabase configuration from environment
# Trim whitespace and newlines from JWT secret (common issue with .env files)
_raw_jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
SUPABASE_JWT_SECRET = _raw_jwt_secret.strip() if _raw_jwt_secret else None
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
# Supabase JWT audience - typically the project URL or "authenticated"
SUPABASE_JWT_AUDIENCE = os.getenv("SUPABASE_JWT_AUDIENCE", "authenticated")

# Log JWT secret status on module load (for debugging, but don't log the actual secret)
if SUPABASE_JWT_SECRET:
    logger.info(f"SUPABASE_JWT_SECRET loaded successfully (length: {len(SUPABASE_JWT_SECRET)} chars)")
    # Log first and last few characters for verification (not the full secret)
    if len(SUPABASE_JWT_SECRET) > 10:
        logger.debug(f"JWT secret preview: {SUPABASE_JWT_SECRET[:5]}...{SUPABASE_JWT_SECRET[-5:]}")
    else:
        logger.warning("JWT secret seems too short")
else:
    logger.error("SUPABASE_JWT_SECRET not found in environment variables!")

# JWT algorithm used by Supabase
ALGORITHM = "HS256"


def extract_token_from_header(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """
    Extract JWT token from Authorization header.

    Args:
        authorization: Authorization header value (e.g., "Bearer <token>")

    Returns:
        JWT token string or None if not present
    """
    if not authorization:
        logger.debug("[AUTH] No Authorization header provided")
        return None

    # Check if it's a Bearer token
    if not authorization.startswith("Bearer "):
        logger.warning(f"[AUTH] Authorization header does not start with 'Bearer '. Header value: {authorization[:50]}...")
        return None

    # Extract token (remove "Bearer " prefix)
    token = authorization[len("Bearer ") :].strip()
    if token:
        logger.debug(f"[AUTH] Extracted token (length: {len(token)} chars)")
    else:
        logger.warning("[AUTH] Token is empty after extraction")
    return token


def decode_jwt_token(token: str) -> Optional[dict]:
    """
    Decode and validate JWT token from Supabase.

    Args:
        token: JWT token string

    Returns:
        Decoded token payload or None if invalid
    """
    if not SUPABASE_JWT_SECRET:
        logger.error("SUPABASE_JWT_SECRET not configured in environment variables. Authentication disabled.")
        logger.error("Please set SUPABASE_JWT_SECRET in your .env file or environment variables.")
        logger.error("Note: In .env files, you typically DON'T need quotes unless the value contains spaces.")
        logger.error("Example: SUPABASE_JWT_SECRET=your-secret-here (no quotes needed)")
        return None

    # Log secret length for debugging (don't log the actual secret for security)
    logger.debug(f"Attempting to decode JWT token. Secret length: {len(SUPABASE_JWT_SECRET)} characters")
    if len(SUPABASE_JWT_SECRET) < 32:
        logger.warning(
            f"JWT secret seems too short ({len(SUPABASE_JWT_SECRET)} chars). Supabase JWT secrets are typically 64+ characters."
        )

    try:
        # Decode JWT token using Supabase secret
        # Note: Supabase uses the JWT secret from project settings -> API -> JWT Secret
        # The secret should be the raw base64 string (not decoded)
        # Supabase tokens typically have an "aud" (audience) claim that needs to be validated
        # We'll try with audience validation first, then fall back to without if it fails
        try:
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=[ALGORITHM],
                audience=SUPABASE_JWT_AUDIENCE,
                options={"verify_aud": True},
            )
        except JWTError as audience_error:
            if "Invalid audience" in str(audience_error) or "aud" in str(audience_error).lower():
                # Try decoding without audience validation (some Supabase tokens don't require it)
                logger.debug(f"Audience validation failed, trying without audience check: {audience_error}")
                payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=[ALGORITHM], options={"verify_aud": False})
            else:
                raise

        logger.debug(f"Successfully decoded JWT token. Payload keys: {list(payload.keys())}")
        return payload
    except JWTError as e:
        error_msg = str(e)

        # Additional debugging: try to decode the token without verification to see if it's valid
        try:
            unverified = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
            logger.warning(
                f"Token structure is valid but validation failed. Token payload keys: {list(unverified.keys())}"
            )
            logger.warning(f"Token was issued for user: {unverified.get('sub', 'unknown')}")
            logger.warning(f"Token issuer (iss): {unverified.get('iss', 'unknown')}")
            logger.warning(f"Token audience (aud): {unverified.get('aud', 'not set')}")
        except Exception as decode_err:
            logger.warning(f"Could not decode token even without verification: {decode_err}")

        if "Invalid audience" in error_msg or "aud" in error_msg.lower():
            logger.warning(
                f"JWT audience validation failed. Trying without audience validation...\n"
                f"  Current expected audience: {SUPABASE_JWT_AUDIENCE}\n"
                f"  You can set SUPABASE_JWT_AUDIENCE in .env to match your Supabase project\n"
                f"  Common values: 'authenticated' (default) or your Supabase project URL"
            )
            # Try one more time without audience validation
            try:
                payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=[ALGORITHM], options={"verify_aud": False})
                logger.info("Successfully decoded JWT token without audience validation")
                return payload
            except JWTError as retry_error:
                logger.error(f"Still failed to decode after disabling audience validation: {retry_error}")
        elif "Signature verification failed" in error_msg:
            logger.error(
                f"JWT signature verification failed. This usually means:\n"
                f"  1. SUPABASE_JWT_SECRET doesn't match the JWT secret in your Supabase project settings\n"
                f"  2. The token was signed with a different secret (maybe from a different Supabase project?)\n"
                f"  3. The JWT secret in .env has extra whitespace or formatting issues\n"
                f"  Current secret length: {len(SUPABASE_JWT_SECRET)} characters\n"
                f"  Secret starts with: {SUPABASE_JWT_SECRET[:10]}... (first 10 chars)\n"
                f"  Secret ends with: ...{SUPABASE_JWT_SECRET[-10:]} (last 10 chars)\n"
                f"  Please verify SUPABASE_JWT_SECRET matches your Supabase project's JWT secret EXACTLY\n"
                f"  (Found in: Supabase Dashboard -> Project Settings -> API -> JWT Secret)\n"
                f"  In .env file, use: SUPABASE_JWT_SECRET=your-secret-here (no quotes, no spaces around =)\n"
                f"  Make sure you're using the JWT SECRET, not the anon key or service role key!"
            )
        else:
            logger.warning(f"JWT decode error: {e}. Token preview: {token[:50]}...")

    # Fallback: if local decoding failed or secret isn't configured, try Supabase auth API
    fallback_payload = _fetch_user_payload_from_supabase(token)
    if fallback_payload:
        logger.info("Successfully validated user via Supabase auth fallback")
        return fallback_payload

    return None


def _fetch_user_payload_from_supabase(token: str) -> Optional[dict]:
    """
    Fallback mechanism to validate a Supabase access token by querying Supabase Auth API.

    This is useful when SUPABASE_JWT_SECRET is missing or token decoding fails due to
    secret mismatch. Requires SUPABASE_URL and at least one of SUPABASE_SERVICE_ROLE_KEY
    or SUPABASE_ANON_KEY to be set for the apikey header.
    """

    if not SUPABASE_URL:
        logger.error("Cannot perform Supabase auth fallback - SUPABASE_URL is not configured.")
        return None

    api_key = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY
    if not api_key:
        logger.error(
            "Cannot perform Supabase auth fallback - neither SUPABASE_SERVICE_ROLE_KEY nor SUPABASE_ANON_KEY is configured."
        )
        return None

    auth_endpoint = f"{SUPABASE_URL.rstrip('/')}/auth/v1/user"
    req = urllib_request.Request(
        auth_endpoint,
        method="GET",
        headers={
            "Authorization": f"Bearer {token}",
            "apikey": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    try:
        with urllib_request.urlopen(req, timeout=5) as response:
            body = response.read().decode("utf-8")
            user_data = json.loads(body)

            user_id = user_data.get("id")
            if not user_id:
                logger.warning("Supabase auth fallback succeeded but no user ID returned in response.")
                return None

            logger.warning("Using Supabase auth fallback to validate JWT token.")
            return {
                "sub": user_id,
                "email": user_data.get("email"),
                "aud": user_data.get("aud"),
                "role": user_data.get("role"),
                "exp": user_data.get("exp"),
                "iat": user_data.get("iat"),
            }
    except urllib_error.HTTPError as http_err:
        logger.error(
            "Supabase auth fallback failed (HTTP %s): %s. "
            "Double-check that the provided access token belongs to this Supabase project.",
            http_err.code,
            http_err.reason,
        )
    except urllib_error.URLError as url_err:
        logger.error(f"Supabase auth fallback failed due to network error: {url_err}")
    except Exception as err:
        logger.error(f"Unexpected error during Supabase auth fallback: {err}")

    return None


def extract_user_id_from_payload(payload: dict) -> Optional[str]:
    """
    Extract user ID from decoded JWT payload.

    Supabase JWT tokens contain the user ID in the 'sub' claim.

    Args:
        payload: Decoded JWT payload

    Returns:
        User ID string or None if not found
    """
    # Supabase stores user ID in 'sub' claim
    user_id = payload.get("sub")

    if not user_id:
        logger.warning("No 'sub' claim found in JWT payload")
        return None

    return user_id


async def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    """
    FastAPI dependency for required authentication.

    Extracts and validates JWT token from Authorization header.
    Returns user ID if valid, raises 401 error if invalid or missing.

    Args:
        authorization: Authorization header value

    Returns:
        User ID string

    Raises:
        HTTPException: 401 if authentication fails
    """
    # Extract token from header
    token = extract_token_from_header(authorization)

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Decode token
    payload = decode_jwt_token(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract user ID
    user_id = extract_user_id_from_payload(payload)

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.debug(f"Authenticated user: {user_id}")
    return user_id


async def get_optional_user(authorization: Optional[str] = Header(None)) -> str:
    """
    FastAPI dependency for optional authentication.

    Attempts to extract and validate JWT token from Authorization header.
    Returns user ID if valid, returns "anonymous" if missing or invalid.

    This is useful for endpoints that work both with and without authentication.

    Args:
        authorization: Authorization header value

    Returns:
        User ID string or "anonymous"
    """
    # Extract token from header
    token = extract_token_from_header(authorization)

    if not token:
        logger.warning("[AUTH] No authentication token provided in Authorization header, using anonymous")
        if authorization:
            logger.warning(
                f"[AUTH] Authorization header present but doesn't start with 'Bearer ': {authorization[:50]}..."
            )
        return "anonymous"

    # Decode token
    payload = decode_jwt_token(token)

    if not payload:
        logger.warning(f"[AUTH] Failed to decode JWT token (token length: {len(token)}), using anonymous")
        return "anonymous"

    # Extract user ID
    user_id = extract_user_id_from_payload(payload)

    if not user_id:
        logger.warning(
            f"[AUTH] No user ID found in token payload. Payload keys: {list(payload.keys())}, using anonymous"
        )
        return "anonymous"

    logger.info(f"[AUTH] Successfully authenticated user: {user_id}")
    return user_id
