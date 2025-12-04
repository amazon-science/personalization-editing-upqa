import re
import csv
import json
import time
import boto3
from typing import List, Dict

bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

def load_unique_personas(csv_files: List[str]) -> List[str]:
    """Load unique personas from user 1 and user 2 columns in multiple CSV files.
    Returns a deterministic list of unique personas."""
    seen_personas = set()
    unique_personas = []  # Maintain insertion order
    
    for csv_file in csv_files:
        print(f"Loading personas from {csv_file}...")
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            # Skip header
            next(reader)
            
            for row in reader:
                if len(row) >= 2:
                    # Extract user 1 personas (column 0)
                    user1_personas_text = row[0]
                    user1_personas = [p.strip() for p in user1_personas_text.split('\n') if p.strip()]
                    
                    for persona in user1_personas:
                        # Filter out empty or very short personas and ensure uniqueness
                        if len(persona.strip()) > 10 and persona not in seen_personas:
                            seen_personas.add(persona)
                            unique_personas.append(persona)
                    
                    # Extract user 2 personas (column 1)
                    user2_personas_text = row[1]
                    user2_personas = [p.strip() for p in user2_personas_text.split('\n') if p.strip()]
                    
                    for persona in user2_personas:
                        if len(persona.strip()) > 10 and persona not in seen_personas:
                            seen_personas.add(persona)
                            unique_personas.append(persona)
    
    return unique_personas


# system_msg = """
# Generate natural-language questions and answers that test whether a language model 
# understands a user profile fact. Questions should vary in directness and include 
# a product recommendation prompt. All output must be returned in structured JSON format.
# """

def create_personalization_prompt(persona: str) -> str:
    prompt = f"""You are tasked with analyzing a persona attribute and generating structured personalization data. 

Given this persona attribute: "{persona}"

Generate a JSON response with the following fields:
1. "input_attribute": the original persona attribute (exactly as provided).
2. "attribute_type": a high-level category of the attribute (e.g., "hobby", "profession", "pet", "location", "job", "family", "food preference", "career goal"). Use the broadest appropriate category, not the specific value. Use lowercase and separate words with spaces.
3. "question": explicitly ask about the attribute type, use the exact word of the attribute_type in the question (e.g., "What's my hobby?" for a hiking-related persona).
4. "question_paraphrased": a natural rewording of the direct question.
5. "implicit_question": a conversational question that avoids directly naming the attribute type but still guides toward answers closely aligned with the target in an everyday, non-diagnostic way (e.g., "What should I do this weekend?" for a hiking hobby).
6. "product_recommendation_question": asks for a product suggestion relevant to the attribute_type without mentioning the specific attribute value (e.g., "Any gear I should buy for my hobby?").
7. "target": a concise description of what the persona reveals about the person (a single word or short phrase grounded in the input attribute).

Example input: I enjoy hiking in the mountains.
Example Output:
{{
  "input_attribute": "I enjoy hiking in the mountains.",
  "attribute_type": "hobby",
  "question": "What's my hobby?",
  "question_paraphrased": "What do I like to do for fun?",
  "implicit_question": "Got any suggestions for a relaxing weekend activity?",
  "product_recommendation_question": "Any gear I should buy for my hobby?",
  "target": "Hiking in the mountains",
}}

Respond only with valid JSON, no additional text or explanation."""

    return prompt


def invoke_claude(prompt: str, max_retries: int = 3) -> Dict:
    """Invoke Claude model through Bedrock with retry logic."""
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3
    }
    
    for attempt in range(max_retries):
        try:
            response = bedrock.invoke_model(
                body=json.dumps(body),
                modelId="us.anthropic.claude-sonnet-4-20250514-v1:0"
            )
            
            response_body = json.loads(response.get('body').read())
            content = response_body['content'][0]['text']
            
            # Parse the JSON response
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # Try to extract JSON from the response if it's wrapped in other text
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result
                else:
                    raise ValueError(f"Could not parse JSON from response: {content}")
                    
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
    
    raise Exception(f"Failed to get response after {max_retries} attempts")


def generate_personalization_data(personas: List[str], data_size: int) -> List[Dict]:
    """Generate personalization data for a list of personas."""
    results = []
    
    # Try to load existing data first
    existing_data_file = "/home/personalization-editing/data/UPQA/personalization_edit_970.json"
    existing_data_size = 0
    
    try:
        with open(existing_data_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            existing_data_size = len(existing_data)
            results.extend(existing_data)
            print(f"Loaded {existing_data_size} existing entries from {existing_data_file}")
    except FileNotFoundError:
        print(f"Warning: {existing_data_file} not found. Will generate all {data_size} entries from scratch.")
    except Exception as e:
        print(f"Error loading existing data: {str(e)}. Will generate all {data_size} entries from scratch.")
    
    # If data_size is less than or equal to existing data size, no need to generate more
    if data_size <= existing_data_size:
        print(f"Data size {data_size} is less than or equal to existing data size {existing_data_size}. No processing needed.")
        return results[:data_size]  # Return only the requested amount
    
    # Generate additional data for personas from existing data size onwards
    start_index = existing_data_size
    selected_personas = list(personas)[start_index:data_size]
    
    if selected_personas:
        print(f"Generating {len(selected_personas)} additional entries (personas {start_index+1}-{data_size})...")
        
        for i, persona in enumerate(selected_personas, 1):
            print(f"Processing persona {start_index + i}/{data_size}: {persona[:50]}...")
            
            try:
                prompt = create_personalization_prompt(persona)
                result = invoke_claude(prompt)
                results.append(result)
                # Add small delay to avoid rate limiting
                # time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing persona '{persona}': {str(e)}")
                continue
    
    return results


if __name__ == "__main__":
    data_size = 1000
    # https://github.com/google-research-datasets/Synthetic-Persona-Chat/tree/main/data
    input_files = [
        "/home/personalization-editing/data/UPQA/Synthetic-Persona-Chat_test.csv",
        "/home/personalization-editing/data/UPQA/Synthetic-Persona-Chat_valid.csv",
        "/home/personalization-editing/data/UPQA/Synthetic-Persona-Chat_train.csv"
    ]
    output_file = f"/home/personalization-editing/data/UPQA/personalization_edit_{data_size}.json"
    
    print("Loading unique personas from CSV files...")
    personas_list = load_unique_personas(input_files)
    print(f"Found {len(personas_list)} unique personas total")
    
    print(f"Generating personalization data for {min(data_size, len(personas_list))} personas...")
    personalization_data = generate_personalization_data(personas_list, data_size)
    print(f"Generated {len(personalization_data)} personalization entries")
    
    # Only save if we have data to save
    if personalization_data:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(personalization_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_file}")
    else:
        print("No data to save.")