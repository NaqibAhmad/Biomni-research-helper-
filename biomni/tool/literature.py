import os
import re
import time
from io import BytesIO
from urllib.parse import urljoin

import PyPDF2
import requests
from bs4 import BeautifulSoup
from googlesearch import search


def fetch_supplementary_info_from_doi(doi: str, output_dir: str = "supplementary_info"):
    """Fetches supplementary information for a paper given its DOI and returns a research log.

    Args:
        doi: The paper DOI.
        output_dir: Directory to save supplementary files.

    Returns:
        dict: A dictionary containing a research log and the downloaded file paths.

    """
    research_log = []
    research_log.append(f"Starting process for DOI: {doi}")

    # CrossRef API to resolve DOI to a publisher page
    crossref_url = f"https://doi.org/{doi}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(crossref_url, headers=headers)

    if response.status_code != 200:
        log_message = f"Failed to resolve DOI: {doi}. Status Code: {response.status_code}"
        research_log.append(log_message)
        return {"log": research_log, "files": []}

    publisher_url = response.url
    research_log.append(f"Resolved DOI to publisher page: {publisher_url}")

    # Fetch publisher page
    response = requests.get(publisher_url, headers=headers)
    if response.status_code != 200:
        log_message = f"Failed to access publisher page for DOI {doi}."
        research_log.append(log_message)
        return {"log": research_log, "files": []}

    # Parse page content
    soup = BeautifulSoup(response.content, "html.parser")
    supplementary_links = []

    # Look for supplementary materials by keywords or links
    for link in soup.find_all("a", href=True):
        href = link.get("href")
        text = link.get_text().lower()
        if "supplementary" in text or "supplemental" in text or "appendix" in text:
            full_url = urljoin(publisher_url, href)
            supplementary_links.append(full_url)
            research_log.append(f"Found supplementary material link: {full_url}")

    if not supplementary_links:
        log_message = f"No supplementary materials found for DOI {doi}."
        research_log.append(log_message)
        return research_log

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    research_log.append(f"Created output directory: {output_dir}")

    # Download supplementary materials
    downloaded_files = []
    for link in supplementary_links:
        file_name = os.path.join(output_dir, link.split("/")[-1])
        file_response = requests.get(link, headers=headers)
        if file_response.status_code == 200:
            with open(file_name, "wb") as f:
                f.write(file_response.content)
            downloaded_files.append(file_name)
            research_log.append(f"Downloaded file: {file_name}")
        else:
            research_log.append(f"Failed to download file from {link}")

    if downloaded_files:
        research_log.append(f"Successfully downloaded {len(downloaded_files)} file(s).")
    else:
        research_log.append(f"No files could be downloaded for DOI {doi}.")

    return "\n".join(research_log)


def search_pubmed_ncbi(query: str, max_results: int = 10, sort: str = "relevance") -> str:
    """Search PubMed using NCBI E-utilities API for biomedical literature.

    Parameters
    ----------
    query : str
        Search query string for PubMed
    max_results : int, optional
        Maximum number of results to return (default: 10)
    sort : str, optional
        Sort order for results (default: "relevance")

    Returns
    -------
    str
        Research log with search results
    """
    import xml.etree.ElementTree as ET

    import requests

    log = []
    log.append("# PubMed NCBI Search Results")
    log.append(f"Query: {query}")
    log.append(f"Max Results: {max_results}")
    log.append(f"Sort Order: {sort}")
    log.append("")

    try:
        # Step 1: Search for PMIDs using esearch
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        esearch_params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "xml", "sort": sort}

        log.append("## Step 1: Searching for PMIDs")
        response = requests.get(esearch_url, params=esearch_params)

        if response.status_code != 200:
            log.append(f"Error: Failed to search PubMed. Status code: {response.status_code}")
            return "\n".join(log)

        # Parse PMIDs from response
        root = ET.fromstring(response.content)
        pmids = []
        for id_elem in root.findall(".//Id"):
            pmids.append(id_elem.text)

        log.append(f"Found {len(pmids)} PMIDs")

        if not pmids:
            log.append("No results found for the query.")
            return "\n".join(log)

        # Step 2: Get detailed information using efetch
        log.append("\n## Step 2: Retrieving detailed information")
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        efetch_params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"}

        response = requests.get(efetch_url, params=efetch_params)

        if response.status_code != 200:
            log.append(f"Error: Failed to fetch details. Status code: {response.status_code}")
            return "\n".join(log)

        # Parse detailed results
        root = ET.fromstring(response.content)
        articles = root.findall(".//PubmedArticle")

        log.append(f"Retrieved {len(articles)} detailed articles")
        log.append("\n## Search Results:")

        for i, article in enumerate(articles, 1):
            # Extract title
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "No title"

            # Extract authors
            authors = []
            for author in article.findall(".//Author"):
                last_name = author.find("LastName")
                first_name = author.find("ForeName")
                if last_name is not None:
                    name = last_name.text
                    if first_name is not None:
                        name += f", {first_name.text}"
                    authors.append(name)

            # Extract journal
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else "Unknown journal"

            # Extract publication date
            pub_date = article.find(".//PubDate")
            date_str = "Unknown date"
            if pub_date is not None:
                year = pub_date.find("Year")
                month = pub_date.find("Month")
                if year is not None:
                    date_str = year.text
                    if month is not None:
                        date_str += f"-{month.text}"

            # Extract PMID
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else "Unknown PMID"

            # Extract abstract
            abstract_parts = article.findall(".//AbstractText")
            if abstract_parts:
                # Combine all abstract sections (some papers have structured abstracts)
                abstract = " ".join([part.text for part in abstract_parts if part.text])
            else:
                abstract = "No abstract available"

            log.append(f"\n### Result {i}:")
            log.append(f"**Title:** {title}")
            log.append(f"**Authors:** {', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}")
            log.append(f"**Journal:** {journal}")
            log.append(f"**Date:** {date_str}")
            log.append(f"**PMID:** {pmid}")
            log.append(f"**Abstract:** {abstract}")

        log.append("\n## Summary")
        log.append(f"Successfully retrieved {len(articles)} articles from PubMed")
        log.append("Search completed using NCBI E-utilities API")

    except Exception as e:
        log.append(f"Error during PubMed search: {str(e)}")

    return "\n".join(log)


def search_europe_pmc(query: str, max_results: int = 10, sort: str = "relevance") -> str:
    """Search Europe PMC for European biomedical literature.

    Parameters
    ----------
    query : str
        Search query string for Europe PMC
    max_results : int, optional
        Maximum number of results to return (default: 10)
    sort : str, optional
        Sort order for results (default: "relevance")

    Returns
    -------
    str
        Research log with search results
    """
    import requests

    log = []
    log.append("# Europe PMC Search Results")
    log.append(f"Query: {query}")
    log.append(f"Max Results: {max_results}")
    log.append(f"Sort Order: {sort}")
    log.append("")

    try:
        # Europe PMC REST API
        base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {"query": query, "resultType": "core", "pageSize": max_results, "sortBy": sort, "format": "json"}

        log.append("## Searching Europe PMC")
        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            log.append(f"Error: Failed to search Europe PMC. Status code: {response.status_code}")
            return "\n".join(log)

        data = response.json()
        articles = data.get("resultList", {}).get("result", [])

        log.append(f"Found {len(articles)} articles")

        if not articles:
            log.append("No results found for the query.")
            return "\n".join(log)

        log.append("\n## Search Results:")

        for i, article in enumerate(articles, 1):
            title = article.get("title", "No title")
            authors = article.get("authorString", "Unknown authors")
            journal = article.get("journalTitle", "Unknown journal")
            pub_year = article.get("pubYear", "Unknown year")
            pmid = article.get("pmid", "No PMID")
            doi = article.get("doi", "No DOI")

            log.append(f"\n### Result {i}:")
            log.append(f"**Title:** {title}")
            log.append(f"**Authors:** {authors}")
            log.append(f"**Journal:** {journal}")
            log.append(f"**Year:** {pub_year}")
            log.append(f"**PMID:** {pmid}")
            log.append(f"**DOI:** {doi}")

        log.append("\n## Summary")
        log.append(f"Successfully retrieved {len(articles)} articles from Europe PMC")
        log.append("Search completed using Europe PMC REST API")

    except Exception as e:
        log.append(f"Error during Europe PMC search: {str(e)}")

    return "\n".join(log)


def search_nih_reporter(query: str, max_results: int = 10) -> str:
    """Search NIH Reporter for funded research projects.

    Parameters
    ----------
    query : str
        Search query for NIH Reporter
    max_results : int, optional
        Maximum number of results to return (default: 10)

    Returns
    -------
    str
        Research log with search results
    """
    import requests

    log = []
    log.append("# NIH Reporter Search Results")
    log.append(f"Query: {query}")
    log.append(f"Max Results: {max_results}")
    log.append("")

    try:
        # NIH Reporter API
        base_url = "https://api.reporter.nih.gov/v2/projects/search"

        # Construct search payload
        search_payload = {
            "criteria": {"search_text": query, "search_field": "all"},
            "offset": 0,
            "limit": max_results,
            "sort_field": "project_start_date",
            "sort_order": "desc",
        }

        log.append("## Searching NIH Reporter")
        response = requests.post(base_url, json=search_payload)

        if response.status_code != 200:
            log.append(f"Error: Failed to search NIH Reporter. Status code: {response.status_code}")
            return "\n".join(log)

        data = response.json()
        projects = data.get("results", [])

        log.append(f"Found {len(projects)} projects")

        if not projects:
            log.append("No results found for the query.")
            return "\n".join(log)

        log.append("\n## Search Results:")

        for i, project in enumerate(projects, 1):
            title = project.get("project_title", "No title")
            pi_name = project.get("contact_pi_name", "Unknown PI")
            org_name = project.get("org_name", "Unknown organization")
            project_start = project.get("project_start_date", "Unknown start date")
            project_end = project.get("project_end_date", "Unknown end date")
            total_cost = project.get("total_cost", "Unknown cost")
            project_num = project.get("project_num", "Unknown project number")

            log.append(f"\n### Project {i}:")
            log.append(f"**Title:** {title}")
            log.append(f"**Principal Investigator:** {pi_name}")
            log.append(f"**Organization:** {org_name}")
            log.append(f"**Project Number:** {project_num}")
            log.append(f"**Start Date:** {project_start}")
            log.append(f"**End Date:** {project_end}")
            log.append(f"**Total Cost:** ${total_cost}")

        log.append("\n## Summary")
        log.append(f"Successfully retrieved {len(projects)} projects from NIH Reporter")
        log.append("Search completed using NIH Reporter API v2")

    except Exception as e:
        log.append(f"Error during NIH Reporter search: {str(e)}")

    return "\n".join(log)


def query_arxiv(query: str, max_papers: int = 10) -> str:
    """Query arXiv for papers based on the provided search query.

    Parameters
    ----------
    - query (str): The search query string.
    - max_papers (int): The maximum number of papers to retrieve (default: 10).

    Returns
    -------
    - str: The formatted search results or an error message.

    """
    import arxiv

    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_papers, sort_by=arxiv.SortCriterion.Relevance)
        results = "\n\n".join([f"Title: {paper.title}\nSummary: {paper.summary}" for paper in client.results(search)])
        return results if results else "No papers found on arXiv."
    except Exception as e:
        return f"Error querying arXiv: {e}"


def query_scholar(query: str) -> str:
    """Query Google Scholar for papers based on the provided search query.

    Parameters
    ----------
    - query (str): The search query string.

    Returns
    -------
    - str: The first search result formatted or an error message.

    """
    from scholarly import scholarly

    try:
        search_query = scholarly.search_pubs(query)
        result = next(search_query, None)
        if result:
            return f"Title: {result['bib']['title']}\nYear: {result['bib']['pub_year']}\nVenue: {result['bib']['venue']}\nAbstract: {result['bib']['abstract']}"
        else:
            return "No results found on Google Scholar."
    except Exception as e:
        return f"Error querying Google Scholar: {e}"


# !COMMENTED OUT: This function uses PyMed library which returns incorrect/irrelevant results
# Use search_pubmed_ncbi() instead, which uses the official NCBI E-utilities API
#
# def query_pubmed(query: str, max_papers: int = 10, max_retries: int = 3) -> str:
#     """Query PubMed for papers based on the provided search query.
#
#     Parameters
#     ----------
#     - query (str): The search query string.
#     - max_papers (int): The maximum number of papers to retrieve (default: 10).
#     - max_retries (int): Maximum number of retry attempts with modified queries (default: 3).
#
#     Returns
#     -------
#     - str: The formatted search results or an error message.
#
#     """
#     from pymed import PubMed
#
#     try:
#         pubmed = PubMed(tool="MyTool", email="mmahmood.dev@gmail.com")  # Update with a valid email address
#
#         # Initial attempt
#         papers = list(pubmed.query(query, max_results=max_papers))
#
#         # Retry with modified queries if no results
#         retries = 0
#         while not papers and retries < max_retries:
#             retries += 1
#             # Simplify query with each retry by removing the last word
#             simplified_query = " ".join(query.split()[:-retries]) if len(query.split()) > retries else query
#             time.sleep(1)  # Add delay between requests
#             papers = list(pubmed.query(simplified_query, max_results=max_papers))
#
#         if papers:
#             results = "\n\n".join(
#                 [f"Title: {paper.title}\nAbstract: {paper.abstract}\nJournal: {paper.journal}" for paper in papers]
#             )
#             return results
#         else:
#             return "No papers found on PubMed after multiple query attempts."
#     except Exception as e:
#         return f"Error querying PubMed: {e}"


def search_google(query: str, num_results: int = 3, language: str = "en") -> list[dict]:
    """Search using Google search.

    Args:
        query (str): The search query (e.g., "protocol text or seach question")
        num_results (int): Number of results to return (default: 10)
        language (str): Language code for search results (default: 'en')
        pause (float): Pause between searches to avoid rate limiting (default: 2.0 seconds)

    Returns:
        List[dict]: List of dictionaries containing search results with title and URL

    """
    try:
        results_string = ""
        search_query = f"{query}"

        for res in search(search_query, num_results=num_results, lang=language, advanced=True):
            title = res.title
            url = res.url
            description = res.description

            results_string += f"Title: {title}\nURL: {url}\nDescription: {description}\n\n"

    except Exception as e:
        print(f"Error performing search: {str(e)}")
    return results_string


def extract_url_content(url: str) -> str:
    """Extract the text content of a webpage using requests and BeautifulSoup.

    Args:
        url: Webpage URL to extract content from

    Returns:
        Text content of the webpage

    """
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

    # Check if the response is in text format
    if "text/plain" in response.headers.get("Content-Type", "") or "application/json" in response.headers.get(
        "Content-Type", ""
    ):
        return response.text.strip()  # Return plain text or JSON response directly

    # If it's HTML, use BeautifulSoup to parse
    soup = BeautifulSoup(response.text, "html.parser")

    # Try to find main content first, fallback to body
    content = soup.find("main") or soup.find("article") or soup.body

    # Remove unwanted elements
    for element in content(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
        element.decompose()

    # Extract text with better formatting
    paragraphs = content.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    cleaned_text = []

    for p in paragraphs:
        text = p.get_text().strip()
        if text:  # Only add non-empty paragraphs
            cleaned_text.append(text)

    return "\n\n".join(cleaned_text)


def extract_pdf_content(url: str) -> str:
    """Extract the text content of a PDF file given its URL.

    Args:
        url: URL of the PDF file to extract text from

    Returns:
        The extracted text content from the PDF

    """
    try:
        # Check if the URL ends with .pdf
        if not url.lower().endswith(".pdf"):
            # If not, try to find a PDF link on the page
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # Look for PDF links in the HTML content
                pdf_links = re.findall(r'href=[\'"]([^\'"]+\.pdf)[\'"]', response.text)
                if pdf_links:
                    # Use the first PDF link found
                    if not pdf_links[0].startswith("http"):
                        # Handle relative URLs
                        base_url = "/".join(url.split("/")[:3])
                        url = base_url + pdf_links[0] if pdf_links[0].startswith("/") else base_url + "/" + pdf_links[0]
                    else:
                        url = pdf_links[0]
                else:
                    return f"No PDF file found at {url}. Please provide a direct link to a PDF file."

        # Download the PDF
        response = requests.get(url, timeout=30)

        # Check if we actually got a PDF file (by checking content type or magic bytes)
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" not in content_type and not response.content.startswith(b"%PDF"):
            return f"The URL did not return a valid PDF file. Content type: {content_type}"

        pdf_file = BytesIO(response.content)

        # Try with PyPDF2 first
        try:
            text = ""
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")

        # Clean up the text
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return "The PDF file did not contain any extractable text. It may be an image-based PDF requiring OCR."

        return text

    except requests.exceptions.RequestException as e:
        return f"Error downloading PDF: {str(e)}"
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"
