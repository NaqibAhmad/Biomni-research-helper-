"""
File Storage Service for Biomni

Manages user file uploads and metadata storage in Supabase database.
Provides secure file tracking with user ownership validation and session linking.
"""

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from supabase import Client, create_client

from utils.user_utils import ensure_user_exists, ensure_session_exists

logger = logging.getLogger(__name__)


class FileStorageService:
    """Service for managing user file uploads and metadata."""

    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """
        Initialize the File Storage Service.

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
            logger.info("File storage service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize file storage service: {e}")
            raise

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate checksum for {file_path}: {e}")
            return ""

    def upload_file(
        self,
        user_id: str,
        file_path: str,
        original_filename: str,
        file_size: int,
        file_type: str,
        description: str = "",
        session_id: Optional[str] = None,
        storage_provider: str = "local",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Store file metadata in database and optionally link to session.

        Args:
            user_id: User identifier (must be authenticated, not "anonymous")
            file_path: Path where file is stored on filesystem
            original_filename: Original filename from upload
            file_size: File size in bytes
            file_type: MIME type or file extension
            description: User-provided description
            session_id: Optional session ID to link file to
            storage_provider: Storage provider ('local', 'supabase', 's3')
            tags: Optional list of tags for the file

        Returns:
            Created file metadata record

        Raises:
            ValueError: If user doesn't exist or session is invalid
            Exception: If file metadata storage fails
        """
        # Validate user exists
        if not user_id or user_id == "anonymous":
            raise ValueError("File upload requires authenticated user")

        if not ensure_user_exists(self.client, user_id):
            raise ValueError(f"User {user_id} not found in public.users. Please ensure you are properly authenticated.")

        # Validate session if provided
        if session_id:
            if not ensure_session_exists(self.client, session_id):
                logger.warning(
                    f"Session {session_id} not found in database. File will be uploaded but not linked to session."
                )
                # Don't fail the upload if session doesn't exist - just skip linking
                session_id = None

        try:
            # Calculate checksum
            checksum = self._calculate_checksum(file_path)

            # Extract filename from path
            filename = Path(file_path).name

            # Determine MIME type
            mime_type = file_type
            if not mime_type or "/" not in mime_type:
                # Try to infer from extension
                ext = Path(original_filename).suffix.lower()
                mime_type_map = {
                    ".csv": "text/csv",
                    ".json": "application/json",
                    ".txt": "text/plain",
                    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    ".xls": "application/vnd.ms-excel",
                    ".tsv": "text/tab-separated-values",
                    ".fasta": "text/plain",
                    ".fastq": "text/plain",
                    ".bam": "application/octet-stream",
                    ".sam": "text/plain",
                    ".vcf": "text/plain",
                    ".bed": "text/plain",
                    ".gtf": "text/plain",
                    ".gff": "text/plain",
                }
                mime_type = mime_type_map.get(ext, "application/octet-stream")

            # Prepare file data
            # Ensure storage_provider is one of the allowed values
            allowed_providers = ["supabase", "s3", "local"]
            if storage_provider not in allowed_providers:
                logger.warning(f"Invalid storage_provider '{storage_provider}', defaulting to 'local'")
                storage_provider = "local"

            file_data = {
                "user_id": user_id,
                "filename": filename,
                "original_filename": original_filename,
                "file_path": file_path,
                "file_size": file_size,
                "file_type": file_type,
                "mime_type": mime_type,
                "description": description or "",  # Ensure description is not None
                "storage_provider": storage_provider,
                "checksum": checksum or "",  # Ensure checksum is not None
                "upload_status": "completed",
                "processing_status": "pending",
                "tags": tags or [],  # Add tags, default to empty array
            }

            logger.debug(f"Inserting file data: {file_data}")

            # Insert file metadata
            try:
                result = self.client.table("user_data_files").insert(file_data).execute()
            except Exception as insert_error:
                logger.error(f"Supabase insert error: {type(insert_error).__name__}: {str(insert_error)}")
                # Try to get more details from the error
                if hasattr(insert_error, "message"):
                    logger.error(f"Error message: {insert_error.message}")
                if hasattr(insert_error, "details"):
                    logger.error(f"Error details: {insert_error.details}")
                if hasattr(insert_error, "code"):
                    logger.error(f"Error code: {insert_error.code}")
                raise

            if not result.data or len(result.data) == 0:
                logger.error(f"No data returned from insert. Result: {result}")
                raise Exception("Failed to store file metadata: No data returned")

            file_record = result.data[0]
            file_id = file_record["id"]

            # Link to session if provided
            if session_id:
                try:
                    self.link_file_to_session(file_id, session_id)
                except Exception as e:
                    logger.warning(f"Failed to link file to session: {e}")
                    # Don't fail the upload if session linking fails

            logger.info(f"File metadata stored successfully: {file_id} for user {user_id}")
            return file_record

        except Exception as e:
            logger.error(f"Error storing file metadata: {e}")
            raise

    def link_file_to_session(self, file_id: str, session_id: str) -> Dict[str, Any]:
        """
        Link an existing file to a chat session.

        Args:
            file_id: File ID from user_data_files table
            session_id: Session ID from chat_sessions table

        Returns:
            Created session_files record

        Raises:
            ValueError: If file or session doesn't exist
            Exception: If linking fails
        """
        # Validate session exists
        if not ensure_session_exists(self.client, session_id):
            raise ValueError(f"Session {session_id} not found")

        # Check if file exists
        file_check = self.client.table("user_data_files").select("id").eq("id", file_id).execute()
        if not file_check.data or len(file_check.data) == 0:
            raise ValueError(f"File {file_id} not found")

        try:
            # Check if link already exists
            existing = (
                self.client.table("session_files")
                .select("id")
                .eq("session_id", session_id)
                .eq("file_id", file_id)
                .execute()
            )

            if existing.data and len(existing.data) > 0:
                logger.debug(f"File {file_id} already linked to session {session_id}")
                return existing.data[0]

            # Create link
            link_data = {
                "session_id": session_id,
                "file_id": file_id,
            }

            result = self.client.table("session_files").insert(link_data).execute()

            if not result.data or len(result.data) == 0:
                raise Exception("Failed to link file to session: No data returned")

            logger.info(f"File {file_id} linked to session {session_id}")
            return result.data[0]

        except Exception as e:
            logger.error(f"Error linking file to session: {e}")
            raise

    def get_user_files(
        self, user_id: str, session_id: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get files for a user, optionally filtered by session.

        Args:
            user_id: User identifier
            session_id: Optional session ID to filter files
            limit: Maximum number of files to return
            offset: Offset for pagination

        Returns:
            List of file metadata records
        """
        try:
            if session_id:
                # Get files linked to this session
                # Use * to get all fields including tags
                query = (
                    self.client.table("session_files")
                    .select("file_id, user_data_files(*)")
                    .eq("session_id", session_id)
                    .limit(limit)
                    .offset(offset)
                    .execute()
                )

                # Extract file records from join
                files = []
                if query.data:
                    for item in query.data:
                        if "user_data_files" in item and item["user_data_files"]:
                            file_record = item["user_data_files"]
                            # Only return files owned by this user
                            if file_record.get("user_id") == user_id:
                                files.append(file_record)

                # Also get files that belong to user but aren't linked to session yet
                # This handles the case where files were uploaded but session linking failed
                all_user_files_query = (
                    self.client.table("user_data_files")
                    .select("*")
                    .eq("user_id", user_id)
                    .order("created_at", desc=True)
                    .limit(limit * 2)  # Get more to ensure we have enough after filtering
                    .offset(0)
                    .execute()
                )

                # Get file IDs already in the linked files list
                linked_file_ids = {f.get("id") for f in files if f.get("id")}

                # Add files that aren't already in the linked list
                if all_user_files_query.data:
                    for file_record in all_user_files_query.data:
                        if file_record.get("id") not in linked_file_ids:
                            files.append(file_record)

                # Sort by created_at descending (newest first) and limit
                files.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                files = files[:limit]

                logger.debug(f"Retrieved {len(files)} files for user {user_id} (session: {session_id})")
                return files
            else:
                # Get all user files - order by created_at descending
                # Explicitly select all fields including tags
                query = (
                    self.client.table("user_data_files")
                    .select("id, user_id, filename, original_filename, file_path, file_size, file_type, mime_type, description, tags, storage_provider, checksum, upload_status, processing_status, error_message, metadata, created_at, updated_at")
                    .eq("user_id", user_id)
                    .order("created_at", desc=True)
                    .limit(limit)
                    .offset(offset)
                    .execute()
                )

                files = query.data or []
                logger.debug(f"Retrieved {len(files)} files for user {user_id} (no session filter)")
                
                # Debug: Log files with tags
                for file in files:
                    tags_value = file.get("tags")
                    if tags_value:
                        logger.info(f"File {file.get('original_filename')} has tags: {tags_value} (type: {type(tags_value)})")
                    else:
                        logger.warning(f"File {file.get('original_filename')} has no tags (tags value: {tags_value}, type: {type(tags_value)})")
                        # Log all keys to see what's actually in the file record
                        logger.debug(f"File keys: {list(file.keys())}")
                
                return files

        except Exception as e:
            logger.error(f"Error getting user files: {e}")
            return []

    def get_file_metadata(self, file_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get file metadata with ownership validation.

        Args:
            file_id: File ID
            user_id: User identifier for ownership validation

        Returns:
            File metadata record if found and owned by user, None otherwise
        """
        try:
            result = self.client.table("user_data_files").select("*").eq("id", file_id).eq("user_id", user_id).execute()

            if result.data and len(result.data) > 0:
                return result.data[0]
            return None

        except Exception as e:
            logger.error(f"Error getting file metadata: {e}")
            return None

    def delete_file(self, file_id: str, user_id: str) -> bool:
        """
        Delete file metadata from database (file itself must be deleted separately).

        Args:
            file_id: File ID to delete
            user_id: User identifier for ownership validation

        Returns:
            True if deleted successfully, False otherwise

        Raises:
            ValueError: If file doesn't exist or user doesn't own it
        """
        # Verify ownership
        file_record = self.get_file_metadata(file_id, user_id)
        if not file_record:
            raise ValueError(f"File {file_id} not found or access denied")

        try:
            # Delete session links first
            self.client.table("session_files").delete().eq("file_id", file_id).execute()

            # Delete file metadata
            self.client.table("user_data_files").delete().eq("id", file_id).execute()

            logger.info(f"File metadata deleted: {file_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting file metadata: {e}")
            raise

    def update_file_status(
        self, file_id: str, user_id: str, upload_status: Optional[str] = None, processing_status: Optional[str] = None
    ) -> bool:
        """
        Update file upload or processing status.

        Args:
            file_id: File ID
            user_id: User identifier for ownership validation
            upload_status: New upload status
            processing_status: New processing status

        Returns:
            True if updated successfully
        """
        # Verify ownership
        file_record = self.get_file_metadata(file_id, user_id)
        if not file_record:
            raise ValueError(f"File {file_id} not found or access denied")

        try:
            update_data = {}
            if upload_status:
                update_data["upload_status"] = upload_status
            if processing_status:
                update_data["processing_status"] = processing_status

            if update_data:
                update_data["updated_at"] = datetime.now().isoformat()
                self.client.table("user_data_files").update(update_data).eq("id", file_id).execute()
                logger.info(f"File status updated: {file_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error updating file status: {e}")
            raise
