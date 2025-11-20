description = [
    {
        "description": "Executes the provided Python command in the notebook environment and returns the output.",
        "name": "run_python_repl",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Python command to execute in the notebook environment",
                "name": "command",
                "type": "str",
            }
        ],
    },
    {
        "description": "Read the source code of a function from any module path.",
        "name": "read_function_source_code",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Fully qualified function name "
                "(e.g., "
                "'bioagentos.tool.support_tools.write_python_code')",
                "name": "function_name",
                "type": "str",
            }
        ],
    },
    {
        "description": "Search CDC Wonder database for public health data and statistics.",
        "name": "search_cdc_wonder",
        "optional_parameters": [
            {
                "default": 10,
                "description": "Maximum number of results to return",
                "name": "max_results",
                "type": "int",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Search query for CDC Wonder database",
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "Search LOINC (Logical Observation Identifiers Names and Codes) database.",
        "name": "search_loinc",
        "optional_parameters": [
            {
                "default": 10,
                "description": "Maximum number of results to return",
                "name": "max_results",
                "type": "int",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Search query for LOINC codes",
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "Search Biothings HPO (Human Phenotype Ontology) database.",
        "name": "search_biothings_hpo",
        "optional_parameters": [
            {
                "default": 10,
                "description": "Maximum number of results to return",
                "name": "max_results",
                "type": "int",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Search query for HPO phenotypes",
                "name": "query",
                "type": "str",
            }
        ],
    },
]
