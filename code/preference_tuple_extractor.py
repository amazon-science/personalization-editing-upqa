"""
Preference Tuple Extractor using Claude via AWS Bedrock

get_system_prompt_old generate new questions: travel_restaurant_old_50.json
get_system_prompt use the original questions and directly extract the subject from them (travel_restaurant_50.json)
"""

import os
import re
import glob
import json
import boto3
import argparse
from typing import List, Dict, Optional


class PreferenceTupleExtractor:
    def __init__(self, region_name: str = "us-east-1"):
        """Initialize the extractor with Bedrock client."""
        self.bedrock = boto3.client(service_name="bedrock-runtime", region_name=region_name)
        # self.model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        self.model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
        
    def get_system_prompt_old(self) -> str:
        return """You are a preference analysis expert. Your task is to extract structured preference tuples from natural language preference statements across diverse domains including food, entertainment, shopping, travel, lifestyle, technology, and more.

For each preference statement, identify the main SUBJECT (preference type) and TARGET (specific constraint/value).

SUBJECT Guidelines:
- Use clear, general preference types that describe the nature of the preference
- Examples include: allergy, diet, gaming preference, music preference, brand preference, activity preference, etc.
- Choose the most specific and descriptive type that captures the essence of the preference
- Use lowercase without underscores (e.g., "gaming preference" not "gaming_preference")
- The model should infer appropriate preference types based on context rather than being limited to predefined categories

TARGET Guidelines:
1. Use specific, actionable terms without underscores
2. For negative preferences, use "avoid" or "no" prefix (e.g., "avoid heights", "no subscription")
3. For positive preferences, use the specific item/concept
4. Keep consistent with lowercase, use spaces for multi-word targets

Examples across domains:
- "I have a severe peanut allergy" → (allergy, peanut)
- "I follow a strict vegan diet" → (diet, vegan)
- "I don't enjoy games with pixel art graphics" → (gaming preference, avoid pixel art)
- "I prefer subscription-free software" → (software preference, avoid subscription)
- "I have a fear of heights" → (phobia, heights)
- "I only shop from independent brands" → (brand preference, independent)
- "I avoid activities involving animal exploitation" → (activity preference, avoid animal exploitation)

Return the result as JSON in this exact format:
{
  "subject": "preference_type",
  "target": "specific_value",
  "question": "What's my [subject]?"
}"""

    def get_system_prompt(self) -> str:
        return """You are a recommendation specialist. Your task is to extract structured preference information from natural language preference statements and their associated recommendation questions to enable personalized recommendations across diverse domains including food, entertainment, shopping, travel, lifestyle, technology, and more.

You will be given both a PREFERENCE statement and a QUESTION. Your task is to:
1. SUBJECT: Extract the key recommendable category from the input QUESTION (must be exact word from the question)
2. TARGET: Extract the user's specific preference constraint from the PREFERENCE that directly reflects their needs

SUBJECT Guidelines:
- Extract the key recommendable word directly from the input question
- Must be the exact same word as it appears in the question (same case, same form)
- Look for concrete, recommendable categories (restaurants, hotels, activities, places, etc.)
- The subject should be the main thing being asked for recommendation about
- For compound phrases like "ways to explore nature", extract the most specific and meaningful part that captures what's being recommended (e.g., "ways to explore nature" rather than just "ways")

TARGET Guidelines:
- Directly reflect what the user wants or needs based on their PREFERENCE (positive framing preferred)
- For dietary restrictions/allergies: use the safe alternative (e.g., "gluten-free" not "avoid gluten")
- For preferences: use the desired characteristic (e.g., "independent brands" not "avoid corporations")
- Use specific, actionable terms that guide recommendations
- Keep lowercase, use hyphens for compound terms when standard (e.g., "gluten-free", "budget-friendly")

Examples:
Input:
- PREFERENCE: "I strictly avoid restaurants that serve foods containing gluten due to a severe gluten intolerance."
- QUESTION: "I'll be visiting Rome soon. What are some must-try local restaurants you'd recommend for me?"
Output: {"subject": "restaurants", "target": "gluten-free"}

Input:
- PREFERENCE: "I only stay at budget-friendly accommodations when traveling"
- QUESTION: "What hotels would you suggest for my trip to Paris?"
Output: {"subject": "hotels", "target": "budget-friendly"}

Return the result as JSON in this exact format:
{
  "subject": "exact_word_from_question",
  "target": "user_preference_constraint",
}"""

    def clean_json_string(self, json_str: str) -> str:
        """Clean JSON string to fix common formatting issues."""
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        # Remove any extra whitespace
        json_str = json_str.strip()
        return json_str

    def extract_tuple_from_preference(self, preference: str, question: str = None, max_retries: int = 3) -> Optional[Dict[str, str]]:
        """Extract preference information from a preference statement and optional question."""
        
        # Choose system prompt and construct user message based on whether question is provided
        if question is not None:
            system_prompt = self.get_system_prompt()
            user_message = f"PREFERENCE: \"{preference}\"\n\nQUESTION: \"{question}\"\n\nExtract the subject from the QUESTION (exact word), target from the PREFERENCE (positive framing), and use the exact QUESTION as provided. Return as valid JSON with no trailing commas:\n\nJSON:"
        else:
            system_prompt = self.get_system_prompt_old()
            user_message = f"Extract the core preference relationship from this statement and return as valid JSON with no trailing commas:\n\n\"{preference}\"\n\nJSON:"
        
        # Prepare the request body
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 128,
            "temperature": 0.3,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_message
                        }
                    ]
                }
            ]
        })
        
        for attempt in range(max_retries):
            try:
                response = self.bedrock.invoke_model(body=body, modelId=self.model_id)
                response_body = json.loads(response.get("body").read())
                
                if 'content' in response_body and len(response_body['content']) > 0:
                    response_text = response_body['content'][0]['text'].strip()
                    
                    try:
                        # Extract JSON from response (might have extra text)
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            # Clean the JSON string to fix common issues
                            json_str = self.clean_json_string(json_str)
                            parsed_json = json.loads(json_str)
                            
                            # Validate required fields
                            if 'subject' in parsed_json and 'target' in parsed_json:
                                return {
                                    'subject': parsed_json['subject'].strip(),
                                    'target': parsed_json['target'].strip(),
                                    'question': question
                                }
                            else:
                                print(f"Warning: Missing required fields in JSON response: {parsed_json}")
                                if attempt < max_retries - 1:
                                    print(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                                    continue
                                return None
                        else:
                            print(f"Warning: No JSON found in response: {response_text}")
                            if attempt < max_retries - 1:
                                print(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                                continue
                            return None
                            
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse JSON from response: {response_text}, Error: {e}")
                        if attempt < max_retries - 1:
                            print(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                            continue
                        return None
                else:
                    print("Warning: No content in response")
                    if attempt < max_retries - 1:
                        print(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                        continue
                    return None
                    
            except Exception as e:
                print(f"Error processing preference '{preference[:50]}...': {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                    continue
                return None
        
        print(f"Failed to extract after {max_retries} attempts")
        return None

    def process_dataset(self, dataset_path: str, size: Optional[int] = None) -> List[Dict]:
        """Process dataset(s) and extract tuples for all preferences."""
        
        # Check if path is directory or file
        if os.path.isdir(dataset_path):
            # Process all JSON files in directory
            json_files = glob.glob(os.path.join(dataset_path, "*.json"))
            if not json_files:
                print(f"No JSON files found in directory: {dataset_path}")
                return []
            
            all_results = []
            for json_file in sorted(json_files):
                print(f"\n=== Processing file: {os.path.basename(json_file)} ===")
                file_results = self._process_single_file(json_file, size)
                all_results.extend(file_results)
            
            return all_results
        else:
            # Process single file
            return self._process_single_file(dataset_path, size)
    
    def _process_single_file(self, file_path: str, size: Optional[int] = None) -> List[Dict]:
        """Process a single JSON file and extract tuples."""
        
        # Load the dataset
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading dataset {file_path}: {str(e)}")
            return []
        
        results = []
        file_name = os.path.basename(file_path)
        
        # Apply size limit if specified
        if size is not None and size > 0:
            data = data[:size]
            print(f"Processing first {len(data)} preference statements from {file_name} (limited by size parameter)...")
        else:
            print(f"Processing {len(data)} preference statements from {file_name}...")
        
        for i, item in enumerate(data):
            if 'preference' not in item:
                print(f"Warning: Item {i} missing 'preference' field")
                continue
                
            preference = item['preference']
            question = item.get('question', '')
            print(f"\nProcessing {i+1}/{len(data)}: {preference[:80]}...")
            
            # Extract preference information
            extraction_result = self.extract_tuple_from_preference(preference, question)
            
            # # Extract topic from filename (remove .json extension and any path)
            # topic = os.path.splitext(file_name)[0]
            
            # Create result entry in the new format
            result_entry = {
                'topic': item['topic'],
                'preference': preference,
                'question': item.get('question', ''),
                'explanation': item.get('explanation', '')
            }
            
            if extraction_result:
                result_entry['subject'] = extraction_result['subject']
                result_entry['target'] = extraction_result['target']
                # result_entry['direct_question'] = extraction_result['question']
                print(f"  → Subject: {extraction_result['subject']}, Target: {extraction_result['target']}")
                print(f"  → Question: {extraction_result['question']}")
            else:
                result_entry['subject'] = None
                result_entry['target'] = None
                # result_entry['direct_question'] = None
                print(f"  → Failed to extract preference information")
            
            results.append(result_entry)
        
        return results

    def save_results(self, results: List[Dict], output_path: str):
        """Save the extraction results to a JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_path}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")


def main():
    # default_input_path = '/home/personalization-editing/code/PrefEval/benchmark_dataset/explicit_preference/travel_restaurant.json'
    # default_input_path = '/home/personalization-editing/code/PrefEval/benchmark_dataset/explicit_preference/entertain_sports.json'
    default_input_path = '../data/prefeval_pro/prefeval_pro_balanced_raw.json'
    parser = argparse.ArgumentParser(description='Extract preference tuples using Claude')
    parser.add_argument('--input', '-i', 
                        default=default_input_path,
                        help='Path to input JSON dataset or directory')
    parser.add_argument('--output', '-o', 
                        default=f'../data/prefeval_pro',
                        help='Path to output JSON file')
    parser.add_argument('--region', '-r', 
                        default='us-east-1',
                        help='AWS region for Bedrock')
    parser.add_argument('--test', '-t', 
                        action='store_true',
                        help='Run with test examples only')
    parser.add_argument('--size', '-s', 
                        type=int,
                        default=None,
                        help='Limit processing to first N items (default: process all)')
    
    args = parser.parse_args()
    
    extractor = PreferenceTupleExtractor(region_name=args.region)
    
    if args.test:
        # Test with provided examples and diverse domain examples
        test_data = [
            {
                "preference": "I have a severe peanut allergy, so I must avoid any foods containing peanuts or peanut products.",
                "question": "What restaurants would you recommend for me?"
            },
            {
                "preference": "I follow a strict vegan diet and refuse to consume any animal-derived products, including honey.",
                "question": "What food would you suggest for dinner?"
            },
            {
                "preference": "I don't enjoy games with pixel art graphics.",
                "question": "What games would you recommend for me?"
            },
            {
                "preference": "I have a strong aversion to subscription-based models and prefer to pay a one-time fee for software and services whenever possible.",
                "question": "What software would you recommend for me?"
            },
            {
                "preference": "I have an acute fear of heights, so I strictly avoid any activities or locations involving significant elevation.",
                "question": "What activities would you recommend for me?"
            },
            {
                "preference": "I prefer to shop for technology products from smaller, independent brands rather than large corporations.",
                "question": "What brands would you recommend for me?"
            }
        ]
        
        print("Testing with sample preferences and questions:")
        for test_item in test_data:
            print(f"\nPreference: {test_item['preference']}")
            print(f"Question: {test_item['question']}")
            result = extractor.extract_tuple_from_preference(test_item['preference'], test_item['question'])
            print(f"Extracted: {result}")
    
    else:
        results = extractor.process_dataset(args.input, args.size)
        extractor.save_results(results, f'{args.output}/{default_input_path.split("/")[-1].split(".")[0]}_{args.size}.json')
        successful_extractions = sum(1 for r in results if r['subject'] is not None)
        print(f"\n=== SUMMARY ===")
        if args.size is not None:
            print(f"Size limit applied: {args.size} items per file")
        print(f"Total preferences processed: {len(results)}")
        print(f"Successful extractions: {successful_extractions}")
        print(f"Success rate: {successful_extractions/len(results)*100:.1f}%")


if __name__ == "__main__":
    main()

# python preference_tuple_extractor.py -s=50
