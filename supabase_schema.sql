-- =====================================================
-- MYBIOAI SUPABASE DATABASE SCHEMA 
-- =====================================================
-- This schema provides comprehensive user management,
-- authentication, chat history, and query tracking
-- for the MyBioAI biomedical AI research platform.
--
-- USER ID CONSISTENCY APPROACH:
-- =====================================================
-- All user-related tables use the SAME UUID from auth.users(id)
-- as foreign keys. This ensures:
-- 1. Single source of truth: auth.users.id is the canonical user identifier
-- 2. Automatic user creation: Trigger creates public.users record on signup
-- 3. Consistent relationships: All tables reference the same user ID
-- 4. Data integrity: Foreign key constraints ensure referential integrity
--
-- When a user signs up:
-- - Supabase creates auth.users record with UUID (e.g., 2c3444da-dc6f-4b28-87f6-6c4695e43cc5)
-- - Trigger automatically creates public.users record with SAME UUID
-- - All subsequent operations use this UUID for created_by, user_id, etc.
--
-- IMPORTANT: Never generate new UUIDs for user identification!
-- Always use the UUID from auth.users.id (extracted from JWT token 'sub' claim)
-- =====================================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- =====================================================
-- 1. USER MANAGEMENT & AUTHENTICATION
-- =====================================================

-- Users table (extends Supabase auth.users) - SIMPLIFIED
CREATE TABLE public.users (
    id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    full_name TEXT,
    bio TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- AUTO-CREATE USER IN public.users WHEN SIGNING UP
-- =====================================================
-- This trigger automatically creates a user record in public.users
-- when a new user signs up in auth.users, ensuring authenticated
-- users can immediately use the system without manual setup.
--
-- CRITICAL: This ensures the UUID from auth.users.id is propagated
-- to public.users.id, maintaining consistency across all tables.

-- Function to handle new user creation
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    -- Insert user with the SAME UUID from auth.users
    -- This UUID will be used as foreign key in all other tables
    INSERT INTO public.users (id, full_name, updated_at)
    VALUES (
        NEW.id,  -- Use the SAME UUID from auth.users
        COALESCE(NEW.raw_user_meta_data->>'full_name', NEW.email),
        NOW()
    )
    ON CONFLICT (id) DO UPDATE
        SET updated_at = NOW(); -- Update timestamp if user already exists
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Drop trigger if it already exists (for idempotency when re-running schema)
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;

-- Trigger to call the function when a new user signs up
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_new_user();

-- =====================================================
-- MIGRATION HELPER: Backfill existing auth.users to public.users
-- =====================================================
-- Run this function once to create public.users records for existing
-- auth.users that don't have corresponding public.users entries.
-- This ensures all authenticated users have public.users records.

CREATE OR REPLACE FUNCTION public.backfill_public_users()
RETURNS INTEGER AS $$
DECLARE
    inserted_count INTEGER;
BEGIN
    INSERT INTO public.users (id, full_name, updated_at)
    SELECT 
        au.id,
        COALESCE(au.raw_user_meta_data->>'full_name', au.email),
        NOW()
    FROM auth.users au
    WHERE NOT EXISTS (
        SELECT 1 FROM public.users pu WHERE pu.id = au.id
    )
    ON CONFLICT (id) DO NOTHING;
    
    GET DIAGNOSTICS inserted_count = ROW_COUNT;
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =====================================================
-- MIGRATION HELPER: Update NULL created_by to current user
-- =====================================================
-- This function helps migrate prompts/files with NULL created_by
-- to the authenticated user. Use with caution - only for orphaned data.
-- 
-- Example usage (update prompts with NULL created_by for a specific user):
-- UPDATE public.prompt_library 
-- SET created_by = '2c3444da-dc6f-4b28-87f6-6c4695e43cc5'
-- WHERE created_by IS NULL AND id = '9aff474e-d953-40d8-8196-e5339f94ce36';
--
-- Note: This is a manual migration step. The application code now handles
-- NULL created_by values gracefully by including them in queries.

-- =====================================================
-- 2. CHAT SESSIONS & CONVERSATIONS
-- =====================================================

-- Chat sessions table - SIMPLIFIED
-- user_id: UUID from auth.users.id (same as public.users.id)
-- This ensures all sessions are linked to the authenticated user's canonical ID
CREATE TABLE public.chat_sessions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    title TEXT,
    session_type TEXT DEFAULT 'general' CHECK (session_type IN ('general', 'research', 'analysis', 'education')),
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_message_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    -- Additional columns for service compatibility
    start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITH TIME ZONE,
    message_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}'
);

-- Chat messages table - UPDATED with content field
CREATE TABLE public.chat_messages (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES public.chat_sessions(id) ON DELETE CASCADE,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    content TEXT, -- ADDED: The actual message text
    message_type TEXT DEFAULT 'user' CHECK (message_type IN ('user', 'assistant', 'system', 'tool_call', 'tool_result')),
    content_type TEXT DEFAULT 'text' CHECK (content_type IN ('text', 'markdown', 'json', 'code')),
    parent_message_id UUID REFERENCES public.chat_messages(id),
    tool_calls JSONB DEFAULT '[]',
    execution_log JSONB DEFAULT '[]',
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    -- Additional columns for service compatibility
    model_used TEXT,
    tokens_used INTEGER,
    processing_time_ms INTEGER,
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- 3. DATA MANAGEMENT & FILE STORAGE
-- =====================================================

-- User data files table - ENHANCED with upload tracking
-- user_id: UUID from auth.users.id (same as public.users.id)
-- All files are linked to the authenticated user's canonical ID
CREATE TABLE public.user_data_files (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    file_type TEXT NOT NULL,
    description TEXT,
    tags TEXT[] DEFAULT '{}',
    storage_provider TEXT DEFAULT 'supabase' CHECK (storage_provider IN ('supabase', 's3', 'local')),
    checksum TEXT,
    upload_status TEXT DEFAULT 'completed' CHECK (upload_status IN ('pending', 'uploading', 'completed', 'failed', 'deleted')), -- ADDED
    processing_status TEXT DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')), -- ADDED
    error_message TEXT, -- ADDED
    metadata JSONB DEFAULT '{}', -- ADDED
    mime_type TEXT, -- ADDED
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Session files junction table (no changes)
CREATE TABLE public.session_files (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES public.chat_sessions(id) ON DELETE CASCADE,
    file_id UUID REFERENCES public.user_data_files(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(session_id, file_id)
);

-- =====================================================
-- 4. SYSTEM CONFIGURATION & FEATURES
-- =====================================================

-- User preferences and settings
-- user_id: UUID from auth.users.id (same as public.users.id)
-- UNIQUE constraint ensures one settings record per user
CREATE TABLE public.user_settings (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE UNIQUE,
    llm_preferences JSONB DEFAULT '{"model": "claude-3-5-haiku", "temperature": 0.7}',
    tool_preferences JSONB DEFAULT '{"use_tool_retriever": true, "preferred_modules": []}',
    ui_preferences JSONB DEFAULT '{"theme": "light", "font_size": "medium"}',
    research_preferences JSONB DEFAULT '{"domains": [], "languages": ["en"]}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User session tracking (for auth and analytics)
CREATE TABLE public.user_sessions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    session_token TEXT UNIQUE NOT NULL,
    ip_address TEXT,
    user_agent TEXT,
    last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Streaming session tracking (WebSocket connections)
CREATE TABLE public.streaming_sessions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.users(id) ON DELETE SET NULL,
    chat_session_id UUID REFERENCES public.chat_sessions(id) ON DELETE SET NULL,
    connection_id TEXT NOT NULL,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'completed', 'disconnected', 'error')),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'
);

-- =====================================================
-- 7. INDEXES FOR PERFORMANCE
-- =====================================================

-- Users indexes
CREATE INDEX idx_users_updated_at ON public.users(updated_at);

-- Chat sessions indexes
CREATE INDEX idx_chat_sessions_user_id ON public.chat_sessions(user_id);
CREATE INDEX idx_chat_sessions_created_at ON public.chat_sessions(created_at);
CREATE INDEX idx_chat_sessions_type ON public.chat_sessions(session_type);
CREATE INDEX idx_chat_sessions_is_active ON public.chat_sessions(is_active) WHERE is_active = true;
CREATE INDEX idx_chat_sessions_last_activity ON public.chat_sessions(last_activity_time);

-- Chat messages indexes
CREATE INDEX idx_chat_messages_session_id ON public.chat_messages(session_id);
CREATE INDEX idx_chat_messages_user_id ON public.chat_messages(user_id);
CREATE INDEX idx_chat_messages_created_at ON public.chat_messages(created_at);
CREATE INDEX idx_chat_messages_type ON public.chat_messages(message_type);
CREATE INDEX idx_chat_messages_content_search ON public.chat_messages USING gin(to_tsvector('english', content)); -- ADDED
CREATE INDEX idx_chat_messages_timestamp ON public.chat_messages(timestamp);
CREATE INDEX idx_chat_messages_model_used ON public.chat_messages(model_used);

-- Data files indexes
CREATE INDEX idx_user_data_files_user_id ON public.user_data_files(user_id);
CREATE INDEX idx_user_data_files_type ON public.user_data_files(file_type);
CREATE INDEX idx_user_data_files_created_at ON public.user_data_files(created_at);
CREATE INDEX idx_user_data_files_upload_status ON public.user_data_files(upload_status); -- ADDED
CREATE INDEX idx_user_data_files_processing_status ON public.user_data_files(processing_status); -- ADDED

-- Session files indexes
CREATE INDEX idx_session_files_session_id ON public.session_files(session_id);
CREATE INDEX idx_session_files_file_id ON public.session_files(file_id);

-- New tables indexes
CREATE INDEX idx_user_sessions_user_id ON public.user_sessions(user_id);
CREATE INDEX idx_user_sessions_token ON public.user_sessions(session_token);
CREATE INDEX idx_user_sessions_active ON public.user_sessions(is_active) WHERE is_active = true;

CREATE INDEX idx_streaming_sessions_user_id ON public.streaming_sessions(user_id);
CREATE INDEX idx_streaming_sessions_status ON public.streaming_sessions(status);

-- =====================================================
-- 6. ROW LEVEL SECURITY (RLS) POLICIES
-- =====================================================

-- Enable RLS on all tables
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_data_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.session_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_sessions ENABLE ROW LEVEL SECURITY;

-- Users policies
CREATE POLICY "Users can view own profile" ON public.users
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.users
    FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile" ON public.users
    FOR INSERT WITH CHECK (auth.uid() = id);

-- Chat sessions policies
CREATE POLICY "Users can view own sessions" ON public.chat_sessions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create own sessions" ON public.chat_sessions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own sessions" ON public.chat_sessions
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own sessions" ON public.chat_sessions
    FOR DELETE USING (auth.uid() = user_id);

-- Chat messages policies
CREATE POLICY "Users can view messages in own sessions" ON public.chat_messages
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.chat_sessions 
            WHERE id = chat_messages.session_id 
            AND user_id = auth.uid()
        )
    );

CREATE POLICY "Users can create messages in own sessions" ON public.chat_messages
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.chat_sessions 
            WHERE id = chat_messages.session_id 
            AND user_id = auth.uid()
        )
    );

-- Data files policies
CREATE POLICY "Users can view own files" ON public.user_data_files
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create own files" ON public.user_data_files
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own files" ON public.user_data_files
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own files" ON public.user_data_files
    FOR DELETE USING (auth.uid() = user_id);

-- Session files policies
CREATE POLICY "Users can view session files" ON public.session_files
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.chat_sessions 
            WHERE id = session_files.session_id 
            AND user_id = auth.uid()
        )
    );

CREATE POLICY "Users can create session files" ON public.session_files
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.chat_sessions 
            WHERE id = session_files.session_id 
            AND user_id = auth.uid()
        )
    );

CREATE POLICY "Users can delete session files" ON public.session_files
    FOR DELETE USING (
        EXISTS (
            SELECT 1 FROM public.chat_sessions 
            WHERE id = session_files.session_id 
            AND user_id = auth.uid()
        )
    );

-- User sessions policies
CREATE POLICY "Users can view own user sessions" ON public.user_sessions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create own user sessions" ON public.user_sessions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own user sessions" ON public.user_sessions
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own user sessions" ON public.user_sessions
    FOR DELETE USING (auth.uid() = user_id);

-- =====================================================
-- SERVICE ROLE BYPASS POLICIES (FOR TESTING)
-- =====================================================
-- These policies allow service role to bypass RLS for testing
-- Remove or restrict these in production!

-- Chat sessions - allow service role
CREATE POLICY "Service role can manage all sessions" ON public.chat_sessions
    FOR ALL USING (current_setting('request.jwt.claims', true)::json->>'role' = 'service_role')
    WITH CHECK (current_setting('request.jwt.claims', true)::json->>'role' = 'service_role');

-- Chat messages - allow service role  
CREATE POLICY "Service role can manage all messages" ON public.chat_messages
    FOR ALL USING (current_setting('request.jwt.claims', true)::json->>'role' = 'service_role')
    WITH CHECK (current_setting('request.jwt.claims', true)::json->>'role' = 'service_role');

-- =====================================================
-- 7. TRIGGERS FOR AUTOMATIC UPDATES
-- =====================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON public.users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chat_sessions_updated_at BEFORE UPDATE ON public.chat_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_data_files_updated_at BEFORE UPDATE ON public.user_data_files
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_settings_updated_at BEFORE UPDATE ON public.user_settings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Trigger to update message_count in chat_sessions
CREATE OR REPLACE FUNCTION update_session_message_count()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE public.chat_sessions
    SET message_count = (
        SELECT COUNT(*) FROM public.chat_messages 
        WHERE session_id = NEW.session_id
    ),
    last_activity_time = NOW(),
    last_message_at = NOW()
    WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_message_count_trigger AFTER INSERT ON public.chat_messages
    FOR EACH ROW EXECUTE FUNCTION update_session_message_count();

-- =====================================================
-- 8. PROMPT LIBRARY & TEMPLATES  
-- =====================================================

-- Prompt library table for reusable prompt templates
CREATE TABLE public.prompt_library (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    category TEXT NOT NULL, -- Allow any category value (users can create custom categories)
    tags TEXT[] DEFAULT '{}',
    
    -- Prompt content and structure
    prompt_template TEXT NOT NULL,
    system_prompt TEXT,
    variables JSONB DEFAULT '[]', -- Array of {name, type, description, default}
    
    -- Model configuration
    model_config JSONB DEFAULT '{
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.7,
        "max_tokens": 8192,
        "source": "Anthropic"
    }',
    
    -- Tool bindings (preselected tools for this prompt)
    tool_bindings JSONB DEFAULT '{
        "enabled_modules": [],
        "specific_tools": [],
        "use_tool_retriever": true
    }',
    
    -- Output configuration
    output_template JSONB DEFAULT '{
        "format": "markdown",
        "schema": {},
        "field_mapping": {}
    }',
    
    -- Versioning and metadata
    version INTEGER DEFAULT 1,
    parent_id UUID REFERENCES public.prompt_library(id) ON DELETE SET NULL, -- For versioning
    is_active BOOLEAN DEFAULT true,
    -- Removed is_predefined - all prompts are user-created
    
    -- Ownership and permissions
    -- created_by: UUID from auth.users.id (same as public.users.id)
    -- NULL values are allowed for backward compatibility with existing data
    -- New prompts should always have created_by set to the authenticated user's UUID
    created_by UUID REFERENCES public.users(id) ON DELETE SET NULL,
    updated_by UUID REFERENCES public.users(id) ON DELETE SET NULL,
    
    -- Usage tracking
    usage_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Prompt execution history
CREATE TABLE public.prompt_executions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    prompt_id UUID REFERENCES public.prompt_library(id) ON DELETE CASCADE,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES public.chat_sessions(id) ON DELETE SET NULL,
    
    -- Execution details
    variables_used JSONB DEFAULT '{}',
    rendered_prompt TEXT NOT NULL,
    model_used TEXT,
    
    -- Results
    response TEXT,
    execution_log JSONB DEFAULT '[]',
    tools_used TEXT[] DEFAULT '{}',
    
    -- Performance metrics
    processing_time_ms INTEGER,
    token_usage JSONB DEFAULT '{"input": 0, "output": 0, "total": 0}',
    cost_estimate DECIMAL(10,6),
    
    -- User feedback
    quality_rating INTEGER CHECK (quality_rating BETWEEN 1 AND 5),
    user_feedback TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Prompt favorites/bookmarks
CREATE TABLE public.prompt_favorites (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    prompt_id UUID REFERENCES public.prompt_library(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, prompt_id)
);

-- =====================================================
-- PROMPT LIBRARY INDEXES
-- =====================================================

CREATE INDEX idx_prompt_library_category ON public.prompt_library(category);
CREATE INDEX idx_prompt_library_tags ON public.prompt_library USING gin(tags);
CREATE INDEX idx_prompt_library_created_by ON public.prompt_library(created_by);
-- INDEX removed: idx_prompt_library_is_predefined (field removed)
CREATE INDEX idx_prompt_library_is_active ON public.prompt_library(is_active);
CREATE INDEX idx_prompt_library_usage_count ON public.prompt_library(usage_count DESC);
CREATE INDEX idx_prompt_library_created_at ON public.prompt_library(created_at);

CREATE INDEX idx_prompt_executions_prompt_id ON public.prompt_executions(prompt_id);
CREATE INDEX idx_prompt_executions_user_id ON public.prompt_executions(user_id);
CREATE INDEX idx_prompt_executions_created_at ON public.prompt_executions(created_at);

CREATE INDEX idx_prompt_favorites_user_id ON public.prompt_favorites(user_id);
CREATE INDEX idx_prompt_favorites_prompt_id ON public.prompt_favorites(prompt_id);

-- =====================================================
-- PROMPT LIBRARY RLS POLICIES
-- =====================================================

ALTER TABLE public.prompt_library ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.prompt_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.prompt_favorites ENABLE ROW LEVEL SECURITY;

-- Prompt library policies
-- Users can view only their own prompts (or NULL for testing)
CREATE POLICY "Users can view own prompts" ON public.prompt_library
    FOR SELECT USING (created_by = auth.uid() OR created_by IS NULL);

CREATE POLICY "Users can create own prompts" ON public.prompt_library
    FOR INSERT WITH CHECK (auth.uid() = created_by OR created_by IS NULL);

CREATE POLICY "Users can update own prompts" ON public.prompt_library
    FOR UPDATE USING (auth.uid() = created_by OR auth.uid() = updated_by OR created_by IS NULL);

CREATE POLICY "Users can delete own prompts" ON public.prompt_library
    FOR DELETE USING (auth.uid() = created_by OR created_by IS NULL);

-- Service role can manage all prompts (for testing)
CREATE POLICY "Service role can manage all prompts" ON public.prompt_library
    FOR ALL USING (current_setting('request.jwt.claims', true)::json->>'role' = 'service_role')
    WITH CHECK (current_setting('request.jwt.claims', true)::json->>'role' = 'service_role');

-- Prompt executions policies
CREATE POLICY "Users can view own executions" ON public.prompt_executions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create own executions" ON public.prompt_executions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Service role can manage all executions (for testing)
CREATE POLICY "Service role can manage all executions" ON public.prompt_executions
    FOR ALL USING (current_setting('request.jwt.claims', true)::json->>'role' = 'service_role')
    WITH CHECK (current_setting('request.jwt.claims', true)::json->>'role' = 'service_role');

-- Prompt favorites policies
CREATE POLICY "Users can view own favorites" ON public.prompt_favorites
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create own favorites" ON public.prompt_favorites
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own favorites" ON public.prompt_favorites
    FOR DELETE USING (auth.uid() = user_id);

-- Service role can manage all favorites (for testing)
CREATE POLICY "Service role can manage all favorites" ON public.prompt_favorites
    FOR ALL USING (current_setting('request.jwt.claims', true)::json->>'role' = 'service_role')
    WITH CHECK (current_setting('request.jwt.claims', true)::json->>'role' = 'service_role');

-- =====================================================
-- PROMPT LIBRARY TRIGGERS
-- =====================================================

CREATE TRIGGER update_prompt_library_updated_at BEFORE UPDATE ON public.prompt_library
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to increment usage count when prompt is executed
CREATE OR REPLACE FUNCTION increment_prompt_usage()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE public.prompt_library
    SET usage_count = usage_count + 1,
        last_used_at = NOW()
    WHERE id = NEW.prompt_id;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER increment_prompt_usage_trigger AFTER INSERT ON public.prompt_executions
    FOR EACH ROW EXECUTE FUNCTION increment_prompt_usage();

-- =====================================================
-- 9. FEEDBACK SYSTEM
-- =====================================================

-- Feedback submissions table for user feedback on AI responses
CREATE TABLE public.feedback_submissions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    
    -- User and session tracking
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES public.chat_sessions(id) ON DELETE SET NULL,
    output_id TEXT NOT NULL, -- ID of the AI response being rated
    
    -- Metadata
    date DATE,
    prompt TEXT NOT NULL, -- The user's prompt/query that generated the response
    response TEXT NOT NULL, -- The AI response/result being rated 
    
    -- Task Type (multiple selection)
    task_types TEXT[] DEFAULT '{}',
    task_type_other TEXT,
    
    -- Task Understanding
    query_interpreted_correctly TEXT CHECK (query_interpreted_correctly IN ('Yes', 'No')),
    followed_instructions TEXT CHECK (followed_instructions IN ('Yes', 'Partial', 'No')),
    task_understanding_notes TEXT,
    save_to_library TEXT CHECK (save_to_library IN ('Yes', 'No, see below')),
    
    -- Scientific Quality
    accuracy TEXT CHECK (accuracy IN ('Good', 'Mixed', 'Poor')),
    completeness TEXT CHECK (completeness IN ('Yes', 'Partial', 'No')),
    scientific_quality_notes TEXT,
    
    -- Technical Performance
    tools_invoked_correctly TEXT CHECK (tools_invoked_correctly IN ('Yes', 'No', 'Not Sure')),
    outputs_usable TEXT CHECK (outputs_usable IN ('Yes', 'Partial', 'No')),
    latency_acceptable TEXT CHECK (latency_acceptable IN ('Yes', 'No')),
    technical_performance_notes TEXT,
    
    -- Output Clarity & Usability
    readable_structured TEXT CHECK (readable_structured IN ('Yes', 'Partial', 'No')),
    formatting_issues TEXT CHECK (formatting_issues IN ('None', 'Minor', 'Major')),
    output_clarity_notes TEXT,
    
    -- Prompt Handling & Logic
    prompt_followed_instructions TEXT CHECK (prompt_followed_instructions IN ('Yes', 'Partial', 'No')),
    prompt_handling_notes TEXT,
    logical_consistency TEXT CHECK (logical_consistency IN ('Strong', 'Mixed', 'Weak')),
    logical_consistency_notes TEXT,
    
    -- Overall Rating
    overall_rating TEXT CHECK (overall_rating IN ('Excellent', 'Good but needs tweaks', 'Needs significant improvement')),
    overall_notes TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- FEEDBACK INDEXES
-- =====================================================

CREATE INDEX idx_feedback_user_id ON public.feedback_submissions(user_id);
CREATE INDEX idx_feedback_session_id ON public.feedback_submissions(session_id);
CREATE INDEX idx_feedback_output_id ON public.feedback_submissions(output_id);
CREATE INDEX idx_feedback_overall_rating ON public.feedback_submissions(overall_rating);
CREATE INDEX idx_feedback_created_at ON public.feedback_submissions(created_at);
CREATE INDEX idx_feedback_task_types ON public.feedback_submissions USING gin(task_types);

-- =====================================================
-- FEEDBACK RLS POLICIES
-- =====================================================

ALTER TABLE public.feedback_submissions ENABLE ROW LEVEL SECURITY;

-- Users can view own feedback
CREATE POLICY "Users can view own feedback" ON public.feedback_submissions
    FOR SELECT USING (auth.uid() = user_id);

-- Users can create own feedback
CREATE POLICY "Users can create own feedback" ON public.feedback_submissions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Users can update own feedback
CREATE POLICY "Users can update own feedback" ON public.feedback_submissions
    FOR UPDATE USING (auth.uid() = user_id);

-- Users can delete own feedback
CREATE POLICY "Users can delete own feedback" ON public.feedback_submissions
    FOR DELETE USING (auth.uid() = user_id);

-- Service role can manage all feedback (for testing)
CREATE POLICY "Service role can manage all feedback" ON public.feedback_submissions
    FOR ALL USING (current_setting('request.jwt.claims', true)::json->>'role' = 'service_role')
    WITH CHECK (current_setting('request.jwt.claims', true)::json->>'role' = 'service_role');

-- Admins can view all feedback (optional - for analytics)
-- Uncomment if you want admins to see all feedback
-- CREATE POLICY "Admins can view all feedback" ON public.feedback_submissions
--     FOR SELECT USING (
--         EXISTS (
--             SELECT 1 FROM public.users 
--             WHERE id = auth.uid() 
--             AND role = 'admin'
--         )
--     );

-- =====================================================
-- FEEDBACK TRIGGERS
-- =====================================================

CREATE TRIGGER update_feedback_updated_at BEFORE UPDATE ON public.feedback_submissions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- MIGRATION NOTES & BEST PRACTICES
-- =====================================================
--
-- 1. USER ID CONSISTENCY:
--    - Always use auth.users.id (from JWT 'sub' claim) for user identification
--    - Never generate new UUIDs for user-related operations
--    - All foreign keys reference public.users(id) which references auth.users(id)
--
-- 2. MIGRATING EXISTING DATA:
--    a) Run backfill_public_users() to create public.users for existing auth.users
--    b) For prompts/files with NULL created_by:
--       - Option 1: Leave NULL (application handles this gracefully)
--       - Option 2: Update manually to assign to correct user
--       - Option 3: Use migration script to assign based on metadata/email
--
-- 3. NEW DATA:
--    - Always set created_by/user_id when creating records
--    - Use ensure_user_exists() helper in backend to auto-create public.users if missing
--    - Never allow NULL created_by for new authenticated user operations
--
-- 4. FOREIGN KEY CONSTRAINTS:
--    - All user_id/created_by columns reference public.users(id)
--    - public.users(id) references auth.users(id) ON DELETE CASCADE
--    - This ensures data integrity and automatic cleanup on user deletion
--
-- 5. ROW LEVEL SECURITY:
--    - All tables use RLS policies based on auth.uid()
--    - auth.uid() returns the UUID from the JWT token 'sub' claim
--    - This ensures users can only access their own data
--
-- =====================================================
-- SCHEMA COMPLETE 
-- =====================================================
