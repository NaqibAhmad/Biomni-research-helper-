"""
Feedback Service for MyBioAi
Handles user feedback submissions for AI-generated responses
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FeedbackService:
    """Service to manage user feedback for AI responses"""

    def __init__(self):
        """Initialize the feedback service with Supabase"""
        self.supabase = None
        self._initialize_supabase()

    def _initialize_supabase(self):
        """Initialize Supabase client"""
        try:
            import os
            from supabase import create_client

            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

            if not supabase_url or not supabase_key:
                raise ValueError(
                    "Supabase credentials not found. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables."
                )

            self.supabase = create_client(supabase_url, supabase_key)
            logger.info("Feedback service initialized with Supabase")
        except ImportError:
            raise ImportError("Supabase library not installed. Run: pip install supabase")
        except Exception as e:
            raise Exception(f"Error initializing Supabase: {e}")

    def submit_feedback(
        self, feedback_data: dict[str, Any], user_id: str | None = None, session_id: str | None = None
    ) -> dict[str, str | list[str] | None]:
        """
        Submit and store user feedback

        Args:
            feedback_data: Dictionary containing feedback form data
            user_id: User ID from authentication
            session_id: Session ID to link feedback to a chat session

        Returns:
            Dictionary with submission confirmation
        """
        try:
            # Validate required fields
            self._validate_feedback(feedback_data)

            # Store in Supabase
            result = self._store_feedback_supabase(feedback_data, user_id, session_id)
            return result

        except Exception as e:
            logger.error(f"Error submitting feedback: {str(e)}")
            return {"success": False, "error": str(e), "message": "Failed to submit feedback"}

    def get_feedback(self, feedback_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve feedback by ID

        Args:
            feedback_id: The feedback ID to retrieve
            user_id: User ID for filtering (for Supabase RLS)

        Returns:
            Dictionary containing feedback data or None
        """
        try:
            response = self.supabase.table("feedback_submissions").select("*").eq("id", feedback_id)

            if user_id:
                response = response.eq("user_id", user_id)

            result = response.execute()

            if result.data and len(result.data) > 0:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Error retrieving feedback {feedback_id}: {str(e)}")
            return None

    def get_all_feedback(
        self, limit: Optional[int] = None, user_id: Optional[str] = None, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all feedback entries

        Args:
            limit: Optional limit on number of entries to return
            user_id: Filter by user ID
            session_id: Filter by session ID

        Returns:
            List of feedback entries
        """
        try:
            query = self.supabase.table("feedback_submissions").select("*").order("created_at", desc=True)

            if user_id:
                query = query.eq("user_id", user_id)

            if session_id:
                query = query.eq("session_id", session_id)

            if limit:
                query = query.limit(limit)

            result = query.execute()
            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Error retrieving all feedback: {str(e)}")
            return []

    def get_feedback_schema(self) -> dict[str, Any]:
        """
        Return the feedback form schema for frontend integration

        Returns:
            Dictionary containing the form schema
        """
        return {
            "title": "MyBioAi Output Feedback Report",
            "sections": [
                {
                    "id": "metadata",
                    "title": "Metadata",
                    "fields": [
                        {"name": "date", "label": "Date", "type": "date", "required": True},
                        {"name": "output_id", "label": "Output ID", "type": "text", "required": True},
                        {"name": "prompt", "label": "Prompt", "type": "textarea", "required": True},
                    ],
                },
                {
                    "id": "task_type",
                    "title": "1. Task Type (check all that apply)",
                    "fields": [
                        {
                            "name": "task_types",
                            "type": "checkbox_group",
                            "options": [
                                "Literature Review",
                                "Data Analysis",
                                "Phenotype/Biomarker Mapping",
                                "Predictive Modeling",
                                "Experimental Design",
                                "Evidence Discovery & Curation",
                                "Tool testing",
                                "Other",
                            ],
                            "allow_other": True,
                            "other_field": "task_type_other",
                        }
                    ],
                },
                {
                    "id": "task_understanding",
                    "title": "2. Task Understanding",
                    "fields": [
                        {
                            "name": "query_interpreted_correctly",
                            "label": "Was the query interpreted correctly?",
                            "type": "radio",
                            "options": ["Yes", "No"],
                            "required": True,
                        },
                        {
                            "name": "followed_instructions",
                            "label": "Did it follow instructions?",
                            "type": "radio",
                            "options": ["Yes", "Partial", "No"],
                            "required": True,
                        },
                        {"name": "task_understanding_notes", "label": "Notes", "type": "textarea", "required": False},
                        {
                            "name": "save_to_library",
                            "label": "Save prompt to Library",
                            "type": "radio",
                            "options": ["Yes", "No, see below"],
                            "required": True,
                        },
                    ],
                },
                {
                    "id": "scientific_quality",
                    "title": "3. Scientific Quality",
                    "fields": [
                        {
                            "name": "accuracy",
                            "label": "Accuracy of content",
                            "type": "radio",
                            "options": ["Good", "Mixed", "Poor"],
                            "required": True,
                        },
                        {
                            "name": "completeness",
                            "label": "Completeness (covered full scope?)",
                            "type": "radio",
                            "options": ["Yes", "Partial", "No"],
                            "required": True,
                        },
                        {
                            "name": "scientific_quality_notes",
                            "label": "Notes / Corrections",
                            "type": "textarea",
                            "required": False,
                        },
                    ],
                },
                {
                    "id": "technical_performance",
                    "title": "4. Technical Performance",
                    "fields": [
                        {
                            "name": "tools_invoked_correctly",
                            "label": "Tools invoked correctly?",
                            "type": "radio",
                            "options": ["Yes", "No", "Not Sure"],
                            "required": True,
                        },
                        {
                            "name": "outputs_usable",
                            "label": "Were outputs usable (tables, figures, citations)?",
                            "type": "radio",
                            "options": ["Yes", "Partial", "No"],
                            "required": True,
                        },
                        {
                            "name": "latency_acceptable",
                            "label": "Latency / responsiveness acceptable?",
                            "type": "radio",
                            "options": ["Yes", "No"],
                            "required": True,
                        },
                        {
                            "name": "technical_performance_notes",
                            "label": "Notes",
                            "type": "textarea",
                            "required": False,
                        },
                    ],
                },
                {
                    "id": "output_clarity",
                    "title": "5. Output Clarity & Usability",
                    "fields": [
                        {
                            "name": "readable_structured",
                            "label": "Readable & well-structured?",
                            "type": "radio",
                            "options": ["Yes", "Partial", "No"],
                            "required": True,
                        },
                        {
                            "name": "formatting_issues",
                            "label": "Formatting issues (tables, lists, sections)?",
                            "type": "radio",
                            "options": ["None", "Minor", "Major"],
                            "required": True,
                        },
                        {"name": "output_clarity_notes", "label": "Notes", "type": "textarea", "required": False},
                    ],
                },
                {
                    "id": "prompt_handling",
                    "title": "6. Prompt Handling & Logic",
                    "fields": [
                        {
                            "name": "prompt_followed_instructions",
                            "label": "Did it follow instructions?",
                            "type": "radio",
                            "options": ["Yes", "Partial", "No"],
                            "required": True,
                        },
                        {"name": "prompt_handling_notes", "label": "Notes", "type": "textarea", "required": False},
                        {
                            "name": "logical_consistency",
                            "label": "Logical consistency",
                            "type": "radio",
                            "options": ["Strong", "Mixed", "Weak"],
                            "required": True,
                        },
                        {"name": "logical_consistency_notes", "label": "Notes", "type": "textarea", "required": False},
                    ],
                },
                {
                    "id": "overall_rating",
                    "title": "7. Overall Rating",
                    "fields": [
                        {
                            "name": "overall_rating",
                            "label": "Overall Rating",
                            "type": "radio",
                            "options": ["Excellent", "Good but needs tweaks", "Needs significant improvement"],
                            "required": True,
                        },
                        {"name": "overall_notes", "label": "Notes", "type": "textarea", "required": False},
                    ],
                },
            ],
        }

    def _validate_feedback(self, feedback_data: dict[str, Any]) -> None:
        """
        Validate feedback data

        Args:
            feedback_data: The feedback data to validate

        Raises:
            ValueError: If required fields are missing
        """
        required_fields = ["prompt", "output_id"]

        for field in required_fields:
            if field not in feedback_data or not feedback_data[field]:
                raise ValueError(f"Required field missing: {field}")

    def _store_feedback_supabase(
        self, feedback_data: dict[str, Any], user_id: str | None, session_id: str | None
    ) -> dict[str, str | list[str] | None]:
        """
        Store feedback to Supabase

        Args:
            feedback_data: The feedback data to store
            user_id: User ID from authentication
            session_id: Session ID to link feedback

        Returns:
            Dictionary with submission confirmation
        """
        # Prepare data for Supabase
        supabase_data = {
            "user_id": user_id,
            "session_id": session_id,
            "output_id": feedback_data.get("output_id"),
            "date": feedback_data.get("date"),
            "prompt": feedback_data.get("prompt"),
            "task_types": feedback_data.get("task_types", []),
            "task_type_other": feedback_data.get("task_type_other"),
            "query_interpreted_correctly": feedback_data.get("query_interpreted_correctly"),
            "followed_instructions": feedback_data.get("followed_instructions"),
            "task_understanding_notes": feedback_data.get("task_understanding_notes"),
            "save_to_library": feedback_data.get("save_to_library"),
            "accuracy": feedback_data.get("accuracy"),
            "completeness": feedback_data.get("completeness"),
            "scientific_quality_notes": feedback_data.get("scientific_quality_notes"),
            "tools_invoked_correctly": feedback_data.get("tools_invoked_correctly"),
            "outputs_usable": feedback_data.get("outputs_usable"),
            "latency_acceptable": feedback_data.get("latency_acceptable"),
            "technical_performance_notes": feedback_data.get("technical_performance_notes"),
            "readable_structured": feedback_data.get("readable_structured"),
            "formatting_issues": feedback_data.get("formatting_issues"),
            "output_clarity_notes": feedback_data.get("output_clarity_notes"),
            "prompt_followed_instructions": feedback_data.get("prompt_followed_instructions"),
            "prompt_handling_notes": feedback_data.get("prompt_handling_notes"),
            "logical_consistency": feedback_data.get("logical_consistency"),
            "logical_consistency_notes": feedback_data.get("logical_consistency_notes"),
            "overall_rating": feedback_data.get("overall_rating"),
            "overall_notes": feedback_data.get("overall_notes"),
        }

        # Insert into Supabase
        result = self.supabase.table("feedback_submissions").insert(supabase_data).execute()

        if result.data and len(result.data) > 0:
            feedback_record = result.data[0]
            logger.info(f"Feedback submitted successfully to Supabase: {feedback_record['id']}")

            return {
                "success": True,
                "feedback_id": str(feedback_record["id"]),
                "message": "Feedback submitted successfully",
                "submitted_at": feedback_record["created_at"],
            }
        else:
            raise Exception("Failed to insert feedback into Supabase")
