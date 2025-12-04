import json
import time
import boto3
from typing import List

bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")


def call_llm_for_synonyms(text: str, field_type: str, cluster_size: int = 5) -> List[str]:
    """
    Call LLM API to generate (cluster_size-1) synonyms for a given text, then include the original.
    
    Args:
        text: The original text to generate synonyms for
        field_type: Either 'attribute_type' or 'target' to provide context
        cluster_size: Total size of the cluster (synonyms + original)
    
    Returns:
        List of cluster_size items: [synonym1, synonym2, ..., synonymN, original]
        (lowercase for attribute_type)
    """
    num_synonyms = cluster_size - 1  # Reserve one slot for the original
    
    if field_type == "attribute_type":
        prompt = f"""Generate exactly {num_synonyms} concise synonyms for the attribute type: "{text}"

Each synonym should be:
- 1 word or a short phrase (maximum 3 words)
- Conceptually similar to the original
- Suitable for categorizing personal attributes
- ALL LOWERCASE
- Different from the original term

Original: {text}

Provide only the {num_synonyms} synonyms, one per line, without numbering or bullet points. Ensure all synonyms are in lowercase."""
    
    else:  # target
        prompt = f"""Generate exactly {num_synonyms} concise synonyms or alternative phrasings for: "{text}"

Each synonym should be:
- 1 word or a short phrase (maximum 4 words)  
- Capture the same meaning or concept
- Be concise and clear
- Different from the original phrase

Original: {text}

Provide only the {num_synonyms} synonyms, one per line, without numbering or bullet points."""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 64,
        "temperature": 0.3,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    })

    try:
        response = bedrock.invoke_model(body=body, modelId="us.anthropic.claude-sonnet-4-20250514-v1:0")
        # response = bedrock.invoke_model(body=body, modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
        response_body = json.loads(response.get("body").read())
        
        # Extract the text content and split into lines
        content = response_body.get('content', [{}])[0].get('text', '')
        generated_synonyms = [line.strip() for line in content.split('\n') if line.strip()]
        
        # For attribute_type, make synonyms lowercase; for target, keep original case
        if field_type == "attribute_type":
            generated_synonyms = [syn.lower() for syn in generated_synonyms]
            original_item = text.lower()
        else:
            original_item = text
        
        # Ensure we have exactly (cluster_size-1) generated synonyms, then add original at the end
        if len(generated_synonyms) >= num_synonyms:
            final_synonyms = generated_synonyms[:num_synonyms] + [original_item]
        else:
            # If we don't have enough, pad with variations of the original
            while len(generated_synonyms) < num_synonyms:
                generated_synonyms.append(original_item)
            final_synonyms = generated_synonyms[:num_synonyms] + [original_item]
        
        return final_synonyms
            
    except Exception as e:
        print(f"Error calling LLM API for '{text}': {e}")
        # Return original text repeated cluster_size times as fallback (order doesn't matter when all are the same)
        if field_type == "attribute_type":
            return [text.lower()] * cluster_size
        else:
            return [text] * cluster_size


def call_llm_for_question(attribute_type: str, max_retries: int = 3) -> str:
    """
    Call API to generate a question for a given attribute_type.
    Ensures the attribute_type exists in the generated question.
    
    Args:
        attribute_type: The attribute type to generate a question for
        max_retries: Maximum number of retries if attribute_type not found in question
    
    Returns:
        Generated question containing the attribute_type
    """
    for attempt in range(max_retries):
        prompt = f"""Generate a natural, conversational question that asks about someone's "{attribute_type}".

Requirements:
- The question must contain the exact phrase "{attribute_type}" 
- The question should be natural and conversational
- The question should be suitable for asking someone about their personal {attribute_type}
- Keep it concise (under 15 words)

Examples:
- For "hobby": "What's your hobby?"
- For "career goal": "What's your career goal?"
- For "fitness activity": "What's your fitness activity?"

Generate a question for: {attribute_type}

Provide only the question, nothing else."""

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 128,
            "temperature": 0.3,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        })

        try:
            # response = bedrock.invoke_model(body=body, modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
            response = bedrock.invoke_model(body=body, modelId="us.anthropic.claude-sonnet-4-20250514-v1:0")
            response_body = json.loads(response.get("body").read())
            
            # Extract the text content
            question = response_body.get('content', [{}])[0].get('text', '').strip()
            
            # Check if attribute_type exists in the question (case-insensitive)
            if attribute_type.lower() in question.lower():
                return question
            else:
                print(f"Attempt {attempt + 1}: Generated question '{question}' doesn't contain '{attribute_type}'. Retrying...")
                time.sleep(0.5)  # Small delay before retry
                
        except Exception as e:
            print(f"Error calling LLM API for question generation (attempt {attempt + 1}): {e}")
            time.sleep(0.5)
    
    # If all retries failed, return a fallback question
    fallback_question = f"What's your {attribute_type}?"
    print(f"All retries failed. Using fallback question: '{fallback_question}'")
    return fallback_question


def validate_attribute_in_questions(attribute_types: List[str], questions: List[str]) -> List[bool]:
    """
    Validate that each attribute_type exists in its corresponding question.
    
    Args:
        attribute_types: List of attribute types
        questions: List of questions corresponding to attribute types
    
    Returns:
        List of booleans indicating whether each attribute_type exists in its question
    """
    validations = []
    for i, (attr_type, question) in enumerate(zip(attribute_types, questions)):
        # Check if attribute_type exists in the question (case-insensitive)
        is_valid = attr_type.lower() in question.lower()
        validations.append(is_valid)
        if not is_valid:
            print(f"Validation failed: '{attr_type}' not found in question '{question}'")
    return validations


def process_json_data(input_file: str, output_file: str, data_size: int = None, cluster_size: int = 5):
    """
    Process the JSON data file to replace attribute_type and target with synonym sets.
    Also generates corresponding questions for each new attribute_type.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        data_size: Number of entries to process (None for all entries)
        cluster_size: Size of synonym clusters (synonyms + original)
    
    Changes made:
    - attribute_type becomes a list of cluster_size items: [synonym1, synonym2, ..., synonymN, original_lowercase] (all lowercase)
    - target becomes a list of cluster_size items: [synonym1, synonym2, ..., synonymN, original]
    - New 'questions' field added with cluster_size questions: [question1, question2, ..., questionN, original_question]
    """
    # Load the original data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Limit data size if specified
    if data_size is not None:
        data = data[:data_size]
        print(f"Limited to first {data_size} entries")
    
    print(f"Processing {len(data)} entries...")
    
    # Process each entry
    for i, entry in enumerate(data):
        print(f"Processing entry {i+1}/{len(data)}")
        
        # Generate synonyms for attribute_type
        if "attribute_type" in entry:
            original_attr = entry["attribute_type"]
            original_question = entry.get("question", f"What's your {original_attr}?")
            
            # Get synonyms list (now includes original at last position)
            attr_synonyms = call_llm_for_synonyms(original_attr, "attribute_type", cluster_size)
            entry["attribute_type"] = attr_synonyms
            print(f"  attribute_type: {original_attr} -> {attr_synonyms}")
            
            # Generate questions: questions for new synonyms first, then original question at the end
            questions = []
            for attr_type in attr_synonyms[:-1]:  # Skip last element (original), generate for synonyms only
                question = call_llm_for_question(attr_type)
                questions.append(question)
                print(f"    Generated question for '{attr_type}': {question}")
            questions.append(original_question)  # Add original question at the end
            
            # Validate that each attribute_type exists in its corresponding question
            max_validation_retries = 3
            for validation_attempt in range(max_validation_retries):
                validations = validate_attribute_in_questions(attr_synonyms, questions)
                
                if all(validations):
                    print(f"    All attribute types validated in questions successfully")
                    break
                else:
                    print(f"    Validation attempt {validation_attempt + 1}: Some attribute types not found in questions")
                    
                    # Regenerate questions for failed validations
                    for i, (is_valid, attr_type) in enumerate(zip(validations, attr_synonyms)):
                        if not is_valid:
                            print(f"    Regenerating question for '{attr_type}'...")
                            new_question = call_llm_for_question(attr_type)
                            questions[i] = new_question
                            print(f"    New question for '{attr_type}': {new_question}")
                    
                    # If this is the last attempt, break regardless
                    if validation_attempt == max_validation_retries - 1:
                        print(f"    Max validation retries reached. Some questions may not contain their attribute types.")
                        final_validations = validate_attribute_in_questions(attr_synonyms, questions)
                        failed_count = sum(1 for v in final_validations if not v)
                        if failed_count > 0:
                            print(f"    WARNING: {failed_count} questions still don't contain their attribute types")
            
            # Store the questions ((cluster_size-1) new ones + original = cluster_size total)
            entry["question"] = questions
            print(f"    Final Questions: {questions}")
        
        # Generate synonyms for target
        if "target" in entry:
            original_target = entry["target"]
            # Get synonyms list (now includes original at last position)
            target_synonyms = call_llm_for_synonyms(original_target, "target", cluster_size)
            entry["target"] = target_synonyms
            print(f"  target: {original_target} -> {target_synonyms}")
        
        # time.sleep(1.0)  # Increased delay due to more API calls
    
    # Save the revised data
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Revised data saved to {output_file}")


if __name__ == "__main__":
    # Set data_size to control how many entries to process (None for all)
    data_size = 100
    cluster_size = 9
    # input_file = "../data/personalization_edit_100.json"
    input_file = "../data/UPQA/balanced_subset.json"
    output_file = f"../data/UPQA/balanced_subset_cluster{cluster_size}.json"

    process_json_data(input_file, output_file, data_size, cluster_size)

