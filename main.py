import os
import requests
from bs4 import BeautifulSoup
import json
import re

def summarize_with_ollama(paragraphs, model="llama3.1"):
    """
    Summarizes a list of paragraphs using a local ollama model.
    
    Args:
        paragraphs (list): List of paragraph objects with 'chunk_text' field
        model (str): Name of the ollama model to use (default: "llama2")
        
    Returns:
        str: Summarized text, or None if summarization fails
    """
    if not paragraphs:
        return None
        
    # Combine all paragraph texts with newlines
    full_text = "\n\n".join(p['chunk_text'] for p in paragraphs)
    unique_ref = list(set([p['document_url'] for p in paragraphs]))

    #print(f"!!! ref: {unique_ref}")
    
    try:
        # Call ollama's HTTP API (default: localhost:11434)
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                #"prompt": f"Please summarize this text concisely:\n\n{full_text}, and provide references to the source URLs: {', '.join(unique_ref)}",
                "prompt": f"""
                    Please provide a concise answer to the question based on the reference material: {full_text}, and then list the references to the source URLsL {', '.join(unique_ref)}.
                    e.g.
                    Answer: <your concise answer here>
                    References:
                    - <source URL 1>
                    - <source URL 2>
                """,
                "stream": False  # Get complete response
            },
            timeout=30  # Longer timeout for summarization
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "").strip()
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling ollama API: {e}")
        print("Make sure ollama is running locally with: ollama serve")
        return None
    except Exception as e:
        print(f"Error during summarization: {e}")
        return None

def extract_content_to_json(urls):
    """
    Fetches one or more HTML pages, extracts titles, sections and paragraphs,
    and structures each page into a JSON-compatible Python dictionary.

    Args:
        urls (str | list[str]): A single URL string or a list of URL strings to scrape.

    Returns:
        dict | list[dict]: If a single URL is provided, returns a dict with
            the document structure. If a list of URLs is provided, returns a
            list of document dicts (one per URL). Returns None for a URL
            that could not be fetched; such entries are omitted from the
            returned list.
    """

    single_input = isinstance(urls, str)
    if single_input:
        urls = [urls]

    results = []

    for url in urls:
        print(f"Fetching content from: {url}")

        # 1. Fetch the HTML content
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL ({url}): {e}")
            # Skip this URL but continue processing others
            continue

        # 2. Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Initialize the final structured data for this document
        data = {
            "url": url,
            "document_title": soup.title.string if soup.title else "No Title Found",
            "sections": []
        }

        body = soup.find('body')
        if not body:
            results.append(data)
            continue

        content_elements = body.find_all(['h1', 'h2', 'h3', 'p'])

        current_section = None

        # 3. Iterate through elements and structure the content
        for element in content_elements:
            text = element.get_text(strip=True)

            # Skip empty or very short elements that might be navigation/footer debris
            if not text or len(text) < 5:
                continue

            tag_name = element.name

            if tag_name in ['h1', 'h2', 'h3']:
                current_section = {
                    "title": text,
                    "level": int(tag_name[1]),
                    "content_paragraphs": []
                }
                data['sections'].append(current_section)

            elif tag_name == 'p':
                if current_section:
                    current_section['content_paragraphs'].append(text)
                else:
                    if data['sections'] and data['sections'][0]['title'] == 'Introduction':
                        data['sections'][0]['content_paragraphs'].append(text)
                    else:
                        intro_section = {
                            "title": "Introduction",
                            "level": 0,
                            "content_paragraphs": [text]
                        }
                        data['sections'].insert(0, intro_section)
                        current_section = data['sections'][0]

        results.append(data)

    return results[0] if single_input and results else results


def get_all_paragraphs(extracted_data):
    """
    Collects and returns all paragraphs from the structured `extracted_data` as 
    a list of objects with unique IDs and metadata.

    Args:
        extracted_data (dict): The dict returned by `extract_content_to_json` for
            a single document. The function expects a dict with keys
            `document_title`, `url`, and `sections`.

    Returns:
        list[dict]: List of objects with structure:
            {
                "_id": str,  # Unique identifier based on content hash
                "chunk_text": str,  # The paragraph text
                "section_title": str  # Title of the containing section
            }
        Returns empty list if extracted_data is None or malformed.
    """
    if not extracted_data or 'sections' not in extracted_data:
        return []

    import hashlib  # Import here to avoid global namespace pollution

    paragraphs = []
    doc_title = extracted_data.get('document_title')
    doc_url = extracted_data.get('url')

    for section in extracted_data.get('sections', []):
        section_title = section.get('title', 'Untitled Section')

        # Defensive: section may be missing content_paragraphs
        for p in section.get('content_paragraphs', []) or []:
            if p and isinstance(p, str):
                # Create deterministic ID by hashing content + section
                content_hash = hashlib.sha256(
                    f"{doc_title}:{section_title}:{p}".encode('utf-8')
                ).hexdigest()[:24]  # First 24 chars are sufficient

                paragraphs.append({
                    "_id": content_hash,
                    "chunk_text": p,
                    "section_title": section_title,
                    "document_title": doc_title,
                    "document_url": doc_url
                })

    return paragraphs

def get_pinecone_client_for_index(index_name, api_key=None, cloud="aws", region="us-east-1", embed_model="llama-text-embed-v2"):
    """
    Create (if necessary) and return a Pinecone client configured for `index_name`.

    - Tries to read `api_key` from the environment variable `PINECONE_API_KEY` when not provided.
    - Ensures the index exists by calling `create_index_for_model` if `pc.has_index(index_name)` is False.

    Returns the Pinecone client instance (pc).

    Note: this function intentionally avoids printing or logging the API key.
    """
    try:
        # Import here so the module can be used even if Pinecone isn't installed in some workflows
        from pinecone import Pinecone
    except Exception as e:
        raise ImportError("pinecone client library not available. Install with `pip install pinecone` or the appropriate client package.") from e

    if api_key is None:
        api_key = os.environ.get("PINECONE_API_KEY")

    if not api_key:
        raise ValueError("Pinecone API key not provided and PINECONE_API_KEY environment variable is not set.")

    pc = Pinecone(api_key=api_key)

    # Create the index for the desired model if it does not exist yet
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud=cloud,
            region=region,
            embed={
                "model": embed_model,
                "field_map": {"text": "chunk_text"}
            }
        )

    return pc



# --- Example Usage ---

if __name__ == "__main__":

    urls = [
        "https://www.cnbc.com/2025/11/19/nvidia-nvda-earnings-report-q3-2026.html",
        "https://finance.yahoo.com/news/nvidia-stock-soars-after-q3-earnings-forecasts-top-estimates-with-sales-for-ai-chips-off-the-charts-153409984.html",
        "https://www.globenewswire.com/news-release/2025/11/19/3191444/0/en/NVIDIA-Announces-Financial-Results-for-Third-Quarter-Fiscal-2026.html"
        #"https://www.investing.com/news/transcripts/earnings-call-transcript-nvidia-q3-2025-sees-revenue-surge-stock-rises-93CH-4369112"
    ]

    # extracted_data = extract_content_to_json(urls)
    
    # if not extracted_data:
    #     print("No content extracted from the provided URLs.")
    # else:
    #     # If multiple documents were returned, aggregate paragraphs across them
    #     if isinstance(extracted_data, list):
    #         paragraphs = []
    #         for doc in extracted_data:
    #             paragraphs.extend(get_all_paragraphs(doc))
    #     else:
    #         paragraphs = get_all_paragraphs(extracted_data)

    #     # Print first paragraph as a sample
    #     if paragraphs:
    #         print(f"\nExtracted {len(paragraphs)} paragraphs")
    #         print("\nSample paragraph structure:")
    #         print(json.dumps(paragraphs[0], indent=4, ensure_ascii=False))

    #     # Optional: Save the output to a file
    #     with open("page_content.json", "w", encoding="utf-8") as f:
    #         f.write(json.dumps(paragraphs, indent=4, ensure_ascii=False))
    #     print("\nContent saved to page_content.json")

    # Initialize Pinecone client safely. Prefer using the PINECONE_API_KEY
    # environment variable; if initialization fails handle it gracefully
    # so the rest of the script can continue or exit cleanly.
    try:
        pc = get_pinecone_client_for_index("rag-business-index", api_key="<your key>")
    except ImportError:
        print("Pinecone client library not installed. Install with: pip install pinecone")
        pc = None
    except ValueError as e:
        # Raised when API key is missing
        print(f"Pinecone config error: {e}")
        pc = None
    except Exception as e:
        # Catch-all for other runtime issues (network, API, etc.)
        print(f"Failed to initialize Pinecone client: {e}")
        pc = None

    # TODO: add stuff to check is the namepace exists        
    # if pc: 
    #     index = pc.Index("rag-business-index")
    #     # Upsert paragraphs into Pinecone index
    #     index.upsert_records("Nvidia Q3 result", paragraphs)

    # Try to ask a question
    if pc:
        # Prompt the user to type the query from the command line (safe for non-interactive envs)
        try:
            query = input("\n Enter search query (leave blank for default): ").strip()
        except Exception:
            # input() may fail in non-interactive environments; fall back to default
            query = ""

        if not query:
            query = "what is this article about?"

        print(f"\nSearching for: {query}\n")
        
        # Search the dense index
        index = pc.Index("rag-business-index")
        results = index.search(
            namespace="Nvidia Q3 result",
            query={
                "top_k": 10,
                "inputs": {
                    'text': query
                }
            },
            rerank={
                "model": "bge-reranker-v2-m3",
                "top_n": 10,
                "rank_fields": ["chunk_text"]
            }
        )

        # Print the results
        # for hit in results['result']['hits']:
        #         print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | title: {hit["fields"]["document_title"]} | text: {hit['fields']['chunk_text']:<50}")

        context = [hit['fields'] for hit in results['result']['hits']]
        #print(context)

        # get the summary from a local ollama model
        print("\nGenerating response with ollama...")
        try:
            summary = summarize_with_ollama(context)
            if summary:
                print(f"\nResponse: \n\n {summary} \n")
            else:
                print("Failed to generate answer - no response from ollama")
        except Exception as e:
            print(f"Error while generating summary: {e}")
            summary = None

        
