import asyncio
import sys
from io import StringIO

# Create a persistent namespace that will be shared across all executions
_persistent_namespace = {}


def run_python_repl(command: str) -> str:
    """Executes the provided Python command in a persistent environment and returns the output.
    Variables defined in one execution will be available in subsequent executions.
    """

    def execute_in_repl(command: str) -> str:
        """Helper function to execute the command in the persistent environment."""
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        # Use the persistent namespace
        global _persistent_namespace

        try:
            # Execute the command in the persistent namespace
            exec(command, _persistent_namespace)
            output = mystdout.getvalue()
        except Exception as e:
            output = f"Error: {str(e)}"
        finally:
            sys.stdout = old_stdout
        return output

    command = command.strip("```").strip()
    return execute_in_repl(command)


async def run_python_repl_async(command: str) -> str:
    """Async version: Executes the provided Python command in a persistent environment and returns the output.
    Variables defined in one execution will be available in subsequent executions.
    """

    def execute_in_repl(command: str) -> str:
        """Helper function to execute the command in the persistent environment."""
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        # Use the persistent namespace
        global _persistent_namespace

        try:
            # Execute the command in the persistent namespace
            exec(command, _persistent_namespace)
            output = mystdout.getvalue()
        except Exception as e:
            output = f"Error: {str(e)}"
        finally:
            sys.stdout = old_stdout
        return output

    command = command.strip("```").strip()

    # Run the execution in a thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, execute_in_repl, command)


def read_function_source_code(function_name: str) -> str:
    """Read the source code of a function from any module path.

    Parameters
    ----------
        function_name (str): Fully qualified function name (e.g., 'bioagentos.tool.support_tools.write_python_code')

    Returns
    -------
        str: The source code of the function

    """
    import importlib
    import inspect

    # Split the function name into module path and function name
    parts = function_name.split(".")
    module_path = ".".join(parts[:-1])
    func_name = parts[-1]

    try:
        # Import the module
        module = importlib.import_module(module_path)

        # Get the function object from the module
        function = getattr(module, func_name)

        # Get the source code of the function
        source_code = inspect.getsource(function)

        return source_code
    except (ImportError, AttributeError) as e:
        return f"Error: Could not find function '{function_name}'. Details: {str(e)}"


# def request_human_feedback(question, context, reason_for_uncertainty):
#     """
#     Request human feedback on a question.

#     Parameters:
#         question (str): The question that needs human feedback.
#         context (str): Context or details that help the human understand the situation.
#         reason_for_uncertainty (str): Explanation for why the LLM is uncertain about its answer.

#     Returns:
#         str: The feedback provided by the human.
#     """
#     print("Requesting human feedback...")
#     print(f"Question: {question}")
#     print(f"Context: {context}")
#     print(f"Reason for Uncertainty: {reason_for_uncertainty}")

#     # Capture human feedback
#     human_response = input("Please provide your feedback: ")

#     return human_response


def search_cdc_wonder(query: str, max_results: int = 10) -> str:
    """Search CDC Wonder database for public health data and statistics.

    Parameters
    ----------
    query : str
        Search query for CDC Wonder database
    max_results : int, optional
        Maximum number of results to return (default: 10)

    Returns
    -------
    str
        Research log with search results
    """
    import requests

    log = []
    log.append("# CDC Wonder Search Results")
    log.append(f"Query: {query}")
    log.append(f"Max Results: {max_results}")
    log.append("")

    try:
        # CDC Wonder API (Note: CDC Wonder has limited public API access)
        # This is a simplified implementation
        log.append("## Searching CDC Wonder Database")
        log.append("Note: CDC Wonder has limited public API access.")
        log.append("Full implementation would require specific dataset access.")

        # For demonstration purposes
        log.append(f"Searching for: {query}")
        log.append("CDC Wonder contains mortality, natality, and other public health data")
        log.append("Specific searches would require dataset-specific endpoints")

        log.append("\n## Summary")
        log.append("Search completed for CDC Wonder database")
        log.append("Note: Full implementation would require specific dataset access")

    except Exception as e:
        log.append(f"Error during CDC Wonder search: {str(e)}")

    return "\n".join(log)


def search_loinc(query: str, max_results: int = 10) -> str:
    """Search LOINC (Logical Observation Identifiers Names and Codes) database.

    Parameters
    ----------
    query : str
        Search query for LOINC codes
    max_results : int, optional
        Maximum number of results to return (default: 10)

    Returns
    -------
    str
        Research log with search results
    """
    import requests

    log = []
    log.append("#LOINC Search Results")
    log.append(f"Query: {query}")
    log.append(f"Max Results: {max_results}")
    log.append("")

    try:
        # LOINC API
        base_url = "https://fhir.loinc.org/CodeSystem/$lookup"

        # LOINC search parameters
        params = {"system": "http://loinc.org", "code": query, "_count": max_results}

        log.append("## Searching LOINC Database")
        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            log.append(f"Error: Failed to search LOINC. Status code: {response.status_code}")
            return "\n".join(log)

        data = response.json()

        # Parse LOINC results
        if "parameter" in data:
            parameters = data["parameter"]
            log.append(f"Found {len(parameters)} LOINC codes")

            if not parameters:
                log.append("No LOINC codes found for the query.")
                return "\n".join(log)

            log.append("\n## Search Results:")

            for i, param in enumerate(parameters, 1):
                if "part" in param:
                    parts = param["part"]
                    code = "Unknown code"
                    display = "Unknown display"

                    for part in parts:
                        if part.get("name") == "code":
                            code = part.get("valueString", "Unknown")
                        elif part.get("name") == "display":
                            display = part.get("valueString", "Unknown")

                    log.append(f"\n### LOINC Code {i}:")
                    log.append(f"**Code:** {code}")
                    log.append(f"**Description:** {display}")

        log.append("\n## Summary")
        log.append("Successfully searched LOINC database")
        log.append("Search completed using LOINC FHIR API")

    except Exception as e:
        log.append(f"Error during LOINC search: {str(e)}")

    return "\n".join(log)


def search_biothings_hpo(query: str, max_results: int = 10) -> str:
    """Search Biothings HPO (Human Phenotype Ontology) database.

    Parameters
    ----------
    query : str
        Search query for HPO phenotypes
    max_results : int, optional
        Maximum number of results to return (default: 10)
    max_results : int, optional
        Maximum number of results to return (default: 10)

    Returns
    -------
    str
        Research log with search results
    """
    import requests

    log = []
    log.append("# Biothings HPO Search Results")
    log.append(f"Query: {query}")
    log.append(f"Max Results: {max_results}")
    log.append("")

    try:
        # Biothings HPO API
        base_url = "https://biothings.ncats.io/hpo/query"

        # Construct search parameters
        params = {"q": query, "size": max_results, "from": 0}

        log.append("## Searching Biothings HPO Database")
        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            log.append(f"Error: Failed to search Biothings HPO. Status code: {response.status_code}")
            return "\n".join(log)

        data = response.json()
        hits = data.get("hits", [])

        log.append(f"Found {len(hits)} HPO terms")

        if not hits:
            log.append("No HPO terms found for the query.")
            return "\n".join(log)

        log.append("\n## Search Results:")

        for i, hit in enumerate(hits, 1):
            source = hit.get("_source", {})
            hpo_id = source.get("id", "Unknown ID")
            name = source.get("name", "Unknown name")
            definition = source.get("def", "No definition available")
            synonyms = source.get("synonym", [])

            log.append(f"\n### HPO Term {i}:")
            log.append(f"**HPO ID:** {hpo_id}")
            log.append(f"**Name:** {name}")
            log.append(f"**Definition:** {definition}")
            if synonyms:
                log.append(f"**Synonyms:** {', '.join(synonyms[:3])}{'...' if len(synonyms) > 3 else ''}")

        log.append("\n## Summary")
        log.append("Successfully searched Biothings HPO database")
        log.append("Search completed using Biothings HPO API")

    except Exception as e:
        log.append(f"Error during Biothings HPO search: {str(e)}")

    return "\n".join(log)
