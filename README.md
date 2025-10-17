# MyBioAI: A General-Purpose Biomedical AI Agent

## Overview

MyBioAI is a general-purpose biomedical AI agent designed to autonomously execute a wide range of research tasks across diverse biomedical subfields. By integrating cutting-edge large language model (LLM) reasoning with retrieval-augmented planning and code-based execution, MyBioAI helps scientists dramatically enhance research productivity and generate testable hypotheses.

### Key Features

- **Multi-Agent Architecture** - A1 and A2 agents for complex biomedical reasoning
- **Multiple LLM Support** - Claude Sonnet 4 (default), GPT-5, and more
- **Dynamic Model Selection** - Switch between models at runtime via API
- **Prompt Library** - Save and reuse custom prompts (see [PROMPT_LIBRARY_README.md](./PROMPT_LIBRARY_README.md))
- **Comprehensive Toolset** - Genomics, proteomics, drug discovery, and more
- **RAG-Enhanced** - Tool retrieval for intelligent task execution

## Quick Start

### Installation

Our software environment is massive and we provide a single setup.sh script to setup.
Follow this [file](biomni_env/README.md) to setup the env first.

Then activate the environment E1:

```bash
conda activate biomni_e1
```

then install the biomni official pip package:

```bash
pip install biomni --upgrade
```

For the latest update, install from the github source version, or do:

```bash
pip install git+https://github.com/snap-stanford/Biomni.git@main
```

Lastly, configure your API keys using one of the following methods:

<details>
<summary>Click to expand</summary>

#### Option 1: Using .env file (Recommended)

Create a `.env` file in your project directory:

```bash
# Copy the example file
cp .env.example .env

# Edit the .env file with your actual API keys
```

Your `.env` file should look like:

```env
# Anthropic API Key for Claude models
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# OpenAI API Key (for GPT-5, GPT-4, etc.)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Azure OpenAI API Key (if using Azure OpenAI models)
OPENAI_API_KEY=your_azure_openai_api_key
OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/

# Optional: AI Studio Gemini API Key (if using Gemini models)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: groq API Key (if using groq as model provider)
GROQ_API_KEY=your_groq_api_key_here

# Optional: Set the source of your LLM for example:
#"OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", "Groq", "Custom"
LLM_SOURCE=your_LLM_source_here

# Optional: AWS Bedrock Configuration (if using AWS Bedrock models)
AWS_BEARER_TOKEN_BEDROCK=your_bedrock_api_key_here
AWS_REGION=us-east-1

# Optional: Custom model serving configuration
# CUSTOM_MODEL_BASE_URL=http://localhost:8000/v1
# CUSTOM_MODEL_API_KEY=your_custom_api_key_here

# Optional: Biomni data path (defaults to ./data)
# BIOMNI_DATA_PATH=/path/to/your/data

# Optional: Timeout settings (defaults to 600 seconds)
# BIOMNI_TIMEOUT_SECONDS=600
```

#### Option 2: Using shell environment variables

Alternatively, configure your API keys in bash profile `~/.bashrc`:

```bash
export ANTHROPIC_API_KEY="YOUR_API_KEY"  # For Claude models
export OPENAI_API_KEY="YOUR_API_KEY"  # For GPT-5, GPT-4, GPT-3.5 models
export OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/" # optional unless you are using Azure
export AWS_BEARER_TOKEN_BEDROCK="YOUR_BEDROCK_API_KEY" # optional for AWS Bedrock models
export AWS_REGION="us-east-1" # optional, defaults to us-east-1 for Bedrock
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY" #optional if you want to use a gemini model
export GROQ_API_KEY="YOUR_GROQ_API_KEY" # Optional: set this to use models served by Groq
export LLM_SOURCE="Groq" # Optional: set this to use models served by Groq


```

</details>

#### ⚠️ Known Package Conflicts

Some Python packages are not installed by default in the Biomni environment due to dependency conflicts. If you need these features, you must install the packages manually and may need to uncomment relevant code in the codebase. See the up-to-date list and details in [docs/known_conflicts.md](./docs/known_conflicts.md).

### Basic Usage

Once inside the environment, you can start using MyBioAI:

```python
from biomni.agent import A1

# Initialize the agent with data path, Data lake will be automatically downloaded on first run (~11GB)
# You can use Claude (default)
agent = A1(path='./data', llm='claude-sonnet-4-20250514')

# Or use GPT-5
agent = A1(path='./data', llm='gpt-5', source='OpenAI')

# Execute biomedical tasks using natural language
agent.go("Plan a CRISPR screen to identify genes that regulate T cell exhaustion, generate 32 genes that maximize the perturbation effect.")
agent.go("Perform scRNA-seq annotation at [PATH] and generate meaningful hypothesis")
agent.go("Predict ADMET properties for this compound: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
```

If you plan on using Azure for your model, always prefix the model name with azure- (e.g. llm='azure-gpt-4o').

### Configuration Management

Biomni includes a centralized configuration system that provides flexible ways to manage settings. You can configure Biomni through environment variables, runtime modifications, or direct parameters.

```python
from biomni.config import default_config
from biomni.agent import A1

# RECOMMENDED: Modify global defaults for consistency
# Use GPT-5
default_config.llm = "gpt-5"
default_config.source = "OpenAI"
default_config.timeout_seconds = 1200

# Or use GPT-4
default_config.llm = "gpt-4"
default_config.source = "OpenAI"

# Or use Claude (default)
default_config.llm = "claude-sonnet-4-20250514"
default_config.source = "Anthropic"

# All agents AND database queries use these defaults
agent = A1()  # Everything uses your configured model
```

**Note**: Direct parameters to `A1()` only affect that agent's reasoning, not database queries. For consistent configuration across all operations, use `default_config` or environment variables.

For detailed configuration options, see the **[Configuration Guide](docs/configuration.md)**.

## 📚 Prompt Library

MyBioAI includes a comprehensive **Prompt Library** system for managing, discovering, and executing reusable prompt templates for biomedical research tasks. The library provides:

- **10+ Predefined Templates**: Ready-to-use prompts for common tasks (gene analysis, protein structure, drug discovery, literature review, etc.)
- **Custom Prompts**: Create and save your own prompt templates
- **Variable Substitution**: Parameterized prompts with `{variable}` placeholders
- **Tool Bindings**: Pre-configure which biological tools to use for specific tasks
- **Model Configuration**: Specify optimal LLM settings for different analyses
- **Versioning**: Track changes and maintain multiple versions of prompts
- **Discovery**: Search and filter prompts by category, tags, and popularity
- **Analytics**: Track usage, performance, and effectiveness

### Quick Start

```bash
# Set up Supabase (required for prompt library)
export SUPABASE_URL=your_supabase_url
export SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# Load predefined prompts
cd backend/services
python load_predefined_prompts.py
```

### Using the Prompt Library API

```python
import requests

# List available prompts
prompts = requests.get("http://localhost:8000/api/prompts?category=genomics").json()

# Execute a gene analysis prompt
response = requests.post(
    "http://localhost:8000/api/prompts/execute",
    json={
        "prompt_id": "gene-analysis-prompt-id",
        "variables": {
            "gene_symbol": "BRCA1",
            "years": 5
        }
    }
)

print(response.json()["response"])
```

### Available Prompt Categories

- **Genomics**: Gene function analysis, variant interpretation
- **Protein Analysis**: Structure and function studies
- **Drug Discovery**: Drug-target interactions, immunotherapy targets
- **Literature Review**: Comprehensive research reviews
- **Pathway Analysis**: Enrichment and network analysis
- **Clinical Research**: Variant interpretation, cancer genomics
- **Data Analysis**: RNA-seq, microbiome analysis

### Documentation

- **[Complete Guide](PROMPT_LIBRARY_GUIDE.md)**: Detailed documentation with examples
- **[Quick Start](PROMPT_LIBRARY_QUICKSTART.md)**: 5-minute setup and common workflows
- **[API Reference](http://localhost:8000/docs)**: Interactive API documentation

## 🔄 Dynamic Model Selection

Users can now **dynamically switch between Claude Sonnet 4 and GPT-5** on a per-request basis without restarting the server or editing code.

### Features

- **Default Model**: Claude Sonnet 4 (`claude-sonnet-4-20250514`)
- **Available Models**: Claude Sonnet 4, GPT-5
- **Per-Request Selection**: Choose model for each query
- **No Server Restart**: Switch models on-the-fly
- **Works with Streaming**: WebSocket streaming supports model selection

### Quick Example

```bash
# Use default model (Claude Sonnet 4)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is BRCA1?"}'

# Use GPT-5
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is BRCA1?",
    "model": "gpt-5",
    "source": "OpenAI"
  }'
```

### Get Available Models

```bash
curl http://localhost:8000/api/models
```

For complete documentation and examples, see:

- **[Model Selection Guide](MODEL_SELECTION_GUIDE.md)** - Complete API reference and examples
- Test it: `python test_model_selection.py`

### Example: Creating a Custom Prompt

```python
import requests

# Create a custom pathway analysis prompt
prompt = requests.post(
    "http://localhost:8000/api/prompts",
    json={
        "title": "Custom Pathway Analysis",
        "category": "pathway_analysis",
        "prompt_template": "Analyze pathways for {disease} using genes: {gene_list}",
        "variables": [
            {"name": "disease", "type": "string", "required": True},
            {"name": "gene_list", "type": "string", "required": True}
        ],
        "model_config": {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.3
        },
        "tool_bindings": {
            "enabled_modules": ["systems_biology", "genomics"],
            "use_tool_retriever": True
        }
    }
).json()

# Execute your custom prompt
result = requests.post(
    "http://localhost:8000/api/prompts/execute",
    json={
        "prompt_id": prompt["id"],
        "variables": {
            "disease": "Alzheimer's disease",
            "gene_list": "APP,PSEN1,PSEN2,APOE"
        }
    }
).json()
```

## MCP (Model Context Protocol) Support

Biomni supports MCP servers for external tool integration:

```python
from biomni.agent import A1

agent = A1()
agent.add_mcp(config_path="./mcp_config.yaml")
agent.go("Find FDA active ingredient information for ibuprofen")
```

**Built-in MCP Servers:**
For usage and implementation details, see the [MCP Integration Documentation](docs/mcp_integration.md) and examples in [`tutorials/examples/add_mcp_server/`](tutorials/examples/add_mcp_server/) and [`tutorials/examples/expose_biomni_server/`](tutorials/examples/expose_biomni_server/).

## 🤝 Contributing to Biomni

Biomni is an open-science initiative that thrives on community contributions. We welcome:

- **🔧 New Tools**: Specialized analysis functions and algorithms
- **📊 Datasets**: Curated biomedical data and knowledge bases
- **💻 Software**: Integration of existing biomedical software packages
- **📋 Benchmarks**: Evaluation datasets and performance metrics
- **📚 Misc**: Tutorials, examples, and use cases
- **🔧 Update existing tools**: many current tools are not optimized - fix and replacements are welcome!

Check out this **[Contributing Guide](CONTRIBUTION.md)** on how to contribute to the Biomni ecosystem.

If you have particular tool/database/software in mind that you want to add, you can also submit to [this form](https://forms.gle/nu2n1unzAYodTLVj6) and the biomni team will implement them.

## 🔬 Call for Contributors: Help Build Biomni-E2

Biomni-E1 only scratches the surface of what’s possible in the biomedical action space.

Now, we’re building **Biomni-E2** — a next-generation environment developed **with and for the community**.

We believe that by collaboratively defining and curating a shared library of standard biomedical actions, we can accelerate science for everyone.

**Join us in shaping the future of biomedical AI agent.**

- **Contributors with significant impact** (e.g., 10+ significant & integrated tool contributions or equivalent) will be **invited as co-authors** on our upcoming paper in a top-tier journal or conference.
- **All contributors** will be acknowledged in our publications.
- More contributor perks...

Let’s build it together.

## Tutorials and Examples

**[Biomni 101](./tutorials/biomni_101.ipynb)** - Basic concepts and first steps

More to come!

## 🌐 Web Interface

Experience Biomni through our no-code web interface at **[biomni.stanford.edu](https://biomni.stanford.edu)**.

[![Watch the video](https://img.youtube.com/vi/E0BRvl23hLs/maxresdefault.jpg)](https://youtu.be/E0BRvl23hLs)

## Release schedule

- [ ] 8 Real-world research task benchmark/leaderboard release
- [ ] A tutorial on how to contribute to Biomni
- [ ] A tutorial on baseline agents
- [x] MCP support
- [x] Biomni A1+E1 release

## Important Note

- Security warning: Currently, Biomni executes LLM-generated code with full system privileges. If you want to use it in production, please use in isolated/sandboxed environments. The agent can access files, network, and system commands. Be careful with sensitive data or credentials.
- This release was frozen as of April 15 2025, so it differs from the current web platform.
- Biomni itself is Apache 2.0-licensed, but certain integrated tools, databases, or software may carry more restrictive commercial licenses. Review each component carefully before any commercial use.

## Cite Us

```
@article{huang2025biomni,
  title={Biomni: A General-Purpose Biomedical AI Agent},
  author={Huang, Kexin and Zhang, Serena and Wang, Hanchen and Qu, Yuanhao and Lu, Yingzhou and Roohani, Yusuf and Li, Ryan and Qiu, Lin and Zhang, Junze and Di, Yin and others},
  journal={bioRxiv},
  pages={2025--05},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
