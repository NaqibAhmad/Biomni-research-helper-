-- =====================================================
-- MYBIOAI SUPABASE DATABASE SCHEMA (UPDATED)
-- =====================================================
-- This schema provides comprehensive user management,
-- authentication, chat history, and query tracking
-- for the MyBioAI biomedical AI research platform.
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

-- User API keys for external integrations
CREATE TABLE public.user_api_keys (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    key_name TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    permissions JSONB DEFAULT '["read", "write"]',
    last_used_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true
);

-- =====================================================
-- 2. CHAT SESSIONS & CONVERSATIONS
-- =====================================================

-- Chat sessions table - SIMPLIFIED
CREATE TABLE public.chat_sessions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    title TEXT,
    session_type TEXT DEFAULT 'general' CHECK (session_type IN ('general', 'research', 'analysis', 'education')),
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_message_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Chat messages table - SIMPLIFIED
CREATE TABLE public.chat_messages (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES public.chat_sessions(id) ON DELETE CASCADE,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    message_type TEXT DEFAULT 'user' CHECK (message_type IN ('user', 'assistant', 'system', 'tool_call', 'tool_result')),
    content_type TEXT DEFAULT 'text' CHECK (content_type IN ('text', 'markdown', 'json', 'code')),
    parent_message_id UUID REFERENCES public.chat_messages(id),
    tool_calls JSONB DEFAULT '[]',
    execution_log JSONB DEFAULT '[]',
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- 3. QUERY TRACKING & ANALYTICS
-- =====================================================

-- User queries table for tracking all user interactions - UPDATED
CREATE TABLE public.user_queries (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES public.chat_sessions(id) ON DELETE SET NULL,
    query_text TEXT NOT NULL,
    query_type TEXT DEFAULT 'general' CHECK (query_type IN ('general', 'data_analysis', 'literature_search', 'protein_analysis', 'gene_analysis', 'drug_discovery', 'pathway_analysis')),
    categories TEXT, -- renamed from domain
    data_sources_accessed TEXT[] DEFAULT '{}',
    response_quality_rating INTEGER CHECK (response_quality_rating BETWEEN 1 AND 5),
    user_feedback TEXT,
    token_usage JSONB DEFAULT '{"input": 0, "output": 0, "total": 0}',
    cost_estimate DECIMAL(10,6),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Note: saved_queries functionality is now handled by prompt_library table
-- which provides more comprehensive features including versioning, tool bindings, etc.

-- =====================================================
-- 4. DATA MANAGEMENT & FILE STORAGE
-- =====================================================

-- User data files table - SIMPLIFIED
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

-- Data analysis results
CREATE TABLE public.analysis_results (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES public.chat_sessions(id) ON DELETE SET NULL,
    query_id UUID REFERENCES public.user_queries(id) ON DELETE SET NULL,
    analysis_type TEXT NOT NULL,
    input_data JSONB,
    output_data JSONB,
    visualization_data JSONB,
    parameters JSONB DEFAULT '{}',
    status TEXT DEFAULT 'completed' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    error_message TEXT,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- 5. TOOL USAGE & PERFORMANCE TRACKING
-- =====================================================

-- Tool usage tracking
CREATE TABLE public.tool_usage_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES public.chat_sessions(id) ON DELETE SET NULL,
    tool_name TEXT NOT NULL,
    tool_module TEXT,
    parameters JSONB DEFAULT '{}',
    result_status TEXT CHECK (result_status IN ('success', 'error', 'timeout')),
    execution_time_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tool performance metrics
CREATE TABLE public.tool_performance_metrics (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    tool_name TEXT NOT NULL,
    tool_module TEXT,
    avg_execution_time_ms DECIMAL(10,2),
    success_rate DECIMAL(5,2),
    usage_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(tool_name, tool_module)
);

-- =====================================================
-- 6. SYSTEM CONFIGURATION & FEATURES
-- =====================================================

-- User preferences and settings
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

-- System announcements and notifications
CREATE TABLE public.system_announcements (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    announcement_type TEXT DEFAULT 'info' CHECK (announcement_type IN ('info', 'warning', 'maintenance', 'feature')),
    target_audience TEXT DEFAULT 'all' CHECK (target_audience IN ('all', 'free', 'pro', 'enterprise')),
    is_active BOOLEAN DEFAULT true,
    starts_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
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

-- Chat messages indexes
CREATE INDEX idx_chat_messages_session_id ON public.chat_messages(session_id);
CREATE INDEX idx_chat_messages_user_id ON public.chat_messages(user_id);
CREATE INDEX idx_chat_messages_created_at ON public.chat_messages(created_at);
CREATE INDEX idx_chat_messages_type ON public.chat_messages(message_type);

-- User queries indexes
CREATE INDEX idx_user_queries_user_id ON public.user_queries(user_id);
CREATE INDEX idx_user_queries_session_id ON public.user_queries(session_id);
CREATE INDEX idx_user_queries_type ON public.user_queries(query_type);
CREATE INDEX idx_user_queries_categories ON public.user_queries(categories);
CREATE INDEX idx_user_queries_created_at ON public.user_queries(created_at);
CREATE INDEX idx_user_queries_text_search ON public.user_queries USING gin(to_tsvector('english', query_text));

-- Data files indexes
CREATE INDEX idx_user_data_files_user_id ON public.user_data_files(user_id);
CREATE INDEX idx_user_data_files_type ON public.user_data_files(file_type);
CREATE INDEX idx_user_data_files_created_at ON public.user_data_files(created_at);

-- Session files indexes
CREATE INDEX idx_session_files_session_id ON public.session_files(session_id);
CREATE INDEX idx_session_files_file_id ON public.session_files(file_id);

-- Tool usage indexes
CREATE INDEX idx_tool_usage_user_id ON public.tool_usage_logs(user_id);
CREATE INDEX idx_tool_usage_tool_name ON public.tool_usage_logs(tool_name);
CREATE INDEX idx_tool_usage_created_at ON public.tool_usage_logs(created_at);

-- =====================================================
-- 8. ROW LEVEL SECURITY (RLS) POLICIES
-- =====================================================

-- Enable RLS on all tables
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_queries ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_data_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.session_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tool_usage_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_settings ENABLE ROW LEVEL SECURITY;

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

-- User queries policies
CREATE POLICY "Users can view own queries" ON public.user_queries
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create own queries" ON public.user_queries
    FOR INSERT WITH CHECK (auth.uid() = user_id);

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

-- =====================================================
-- 9. TRIGGERS FOR AUTOMATIC UPDATES
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

-- =====================================================
-- 10. VIEWS FOR COMMON QUERIES
-- =====================================================

-- User activity summary view
CREATE VIEW public.user_activity_summary AS
SELECT 
    u.id,
    u.full_name,
    COUNT(DISTINCT cs.id) as total_sessions,
    COUNT(DISTINCT cm.id) as total_messages,
    COUNT(DISTINCT uq.id) as total_queries,
    MAX(cm.created_at) as last_message_at,
    MAX(uq.created_at) as last_query_at
FROM public.users u
LEFT JOIN public.chat_sessions cs ON u.id = cs.user_id
LEFT JOIN public.chat_messages cm ON cs.id = cm.session_id
LEFT JOIN public.user_queries uq ON u.id = uq.user_id
GROUP BY u.id, u.full_name;

-- Popular queries view
CREATE VIEW public.popular_queries AS
SELECT 
    query_type,
    categories,
    COUNT(*) as usage_count,
    AVG(response_quality_rating) as avg_rating
FROM public.user_queries
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY query_type, categories
ORDER BY usage_count DESC;

-- Tool usage statistics view
CREATE VIEW public.tool_usage_stats AS
SELECT 
    tool_name,
    tool_module,
    COUNT(*) as total_usage,
    COUNT(*) FILTER (WHERE result_status = 'success') as successful_usage,
    AVG(execution_time_ms) as avg_execution_time,
    COUNT(DISTINCT user_id) as unique_users
FROM public.tool_usage_logs
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY tool_name, tool_module
ORDER BY total_usage DESC;

-- =====================================================
-- 11. PROMPT LIBRARY & TEMPLATES
-- =====================================================

-- Prompt library table for reusable prompt templates
CREATE TABLE public.prompt_library (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    category TEXT NOT NULL CHECK (category IN (
        'genomics', 'protein_analysis', 'drug_discovery', 'literature_review', 
        'pathway_analysis', 'clinical_research', 'data_analysis', 'general', 'custom'
    )),
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
-- Users can view only their own prompts
CREATE POLICY "Users can view own prompts" ON public.prompt_library
    FOR SELECT USING (created_by = auth.uid());

CREATE POLICY "Users can create own prompts" ON public.prompt_library
    FOR INSERT WITH CHECK (auth.uid() = created_by);

CREATE POLICY "Users can update own prompts" ON public.prompt_library
    FOR UPDATE USING (auth.uid() = created_by OR auth.uid() = updated_by);

CREATE POLICY "Users can delete own prompts" ON public.prompt_library
    FOR DELETE USING (auth.uid() = created_by);

-- Prompt executions policies
CREATE POLICY "Users can view own executions" ON public.prompt_executions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create own executions" ON public.prompt_executions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Prompt favorites policies
CREATE POLICY "Users can view own favorites" ON public.prompt_favorites
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create own favorites" ON public.prompt_favorites
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own favorites" ON public.prompt_favorites
    FOR DELETE USING (auth.uid() = user_id);

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
-- 12. INITIAL DATA & CONFIGURATION
-- =====================================================

-- Insert default system announcements
INSERT INTO public.system_announcements (title, content, announcement_type, target_audience) VALUES
('Welcome to MyBioAI!', 'Welcome to the MyBioAI biomedical AI research platform. Start by exploring our tools and uploading your data.', 'info', 'all'),
('New Features Available', 'Check out our latest biomedical analysis tools and improved chat interface.', 'feature', 'all');

-- =====================================================
-- SCHEMA COMPLETE
-- =====================================================