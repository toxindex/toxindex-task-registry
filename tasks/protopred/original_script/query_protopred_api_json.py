"""
Example implementation using Embedded JSON in request body.
This is the recommended format for LLM agent usage.
"""
import requests
import json

def query_protopred_api_json(smiles: str, models: str = "model_phys:water_solubility"):
    """
    Query ProtoPRED API using embedded JSON in request body.
    
    This is the BEST format for LLM agent usage because:
    1. LLMs excel at generating JSON structures
    2. Standard REST API pattern
    3. Easy to construct, validate, and debug
    4. No file I/O overhead
    5. Flexible for extensions
    
    Args:
        smiles: SMILES string of the chemical compound
        models: Comma-separated list of models to query
        
    Returns:
        dict: API response as JSON
    """
    url = "https://protopred.protoqsar.com/API/v2/"
    
    # Construct JSON payload - LLMs can easily generate this structure
    payload = {
        "account_token": "1JX3LP",
        "account_secret_key": "A8X9641JM",
        "account_user": "OOntox",
        "module": "ProtoPHYSCHEM",
        "input_type": "SMILES_TEXT",
        "input_data": smiles,
        "models_list": models
    }
    
    # Use json= parameter to automatically:
    # 1. Serialize to JSON
    # 2. Set Content-Type: application/json header
    # 3. Send as request body
    response = requests.post(url, json=payload)
    
    # Raise exception for HTTP errors
    response.raise_for_status()
    
    return response.json()

# Example usage
if __name__ == "__main__":
    # Single SMILES query
    result = query_protopred_api_json("CCCCC")
    
    # Save to file
    with open("output.json", "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    # Print result
    print(json.dumps(result, indent=2, ensure_ascii=False))

