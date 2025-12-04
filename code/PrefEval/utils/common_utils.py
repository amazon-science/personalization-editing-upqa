import os
import json
import time
import yaml
import torch
import tiktoken
from openai import OpenAI
# from google import genai
# from google.genai import types
from transformers import AutoTokenizer, AutoModelForCausalLM


def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Using a default model
    return len(encoding.encode(text))


# Global cache for loaded models and tokenizers
_model_cache = {}


def generate_message_local(
    model_id,
    device="cuda:0",
    system_prompt=None,
    messages=None,
    max_tokens=300,
    temperature=0,
    max_retries=3,
):
    """Generate message using local HuggingFace models with tokenizer.apply_chat_template()"""
    
    # Load model and tokenizer (with caching)
    cache_key = f"{model_id}_{device}"
    if cache_key not in _model_cache:
        print(f"Loading model {model_id} on device {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        model = model.to(device)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        _model_cache[cache_key] = (model, tokenizer)
        print(f"Model {model_id} loaded successfully.")
    else:
        model, tokenizer = _model_cache[cache_key]
    
    # Prepare messages
    chat_messages = []
    if system_prompt:
        chat_messages.append({"role": "system", "content": system_prompt})
    
    if isinstance(messages, str):
        chat_messages.append({"role": "user", "content": messages})
    elif isinstance(messages, list):
        chat_messages.extend(messages if all(isinstance(m, dict) and "role" in m for m in messages) 
                           else [{"role": "user", "content": str(m)} for m in messages])
    else:
        chat_messages.append({"role": "user", "content": str(messages)})
    
    # Generate response
    for attempt in range(max_retries):
        try:
            # Apply chat template or fallback
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                try:
                    msg_tokenized = tokenizer.apply_chat_template(
                        chat_messages, add_generation_prompt=True, return_tensors='pt'
                    ).to(model.device)
                except Exception as e:
                    # Handle models that don't support system role (like Gemma)
                    if "System role not supported" in str(e) or "system" in str(e).lower():
                        # For models that don't support system messages, combine system prompt with first user message
                        modified_messages = []
                        system_content = None
                        
                        for msg in chat_messages:
                            if msg["role"] == "system":
                                system_content = msg["content"]
                            elif msg["role"] == "user" and system_content:
                                # Combine system prompt with first user message
                                combined_content = f"{system_content}\n\n{msg['content']}"
                                modified_messages.append({"role": "user", "content": combined_content})
                                system_content = None  # Only combine with first user message
                            else:
                                modified_messages.append(msg)
                        
                        msg_tokenized = tokenizer.apply_chat_template(
                            modified_messages, add_generation_prompt=True, return_tensors='pt'
                        ).to(model.device)
                    else:
                        raise e
            else:
                # Fallback for models without chat template
                input_text = ""
                for msg in chat_messages:
                    if msg["role"] == "system":
                        input_text += f"System: {msg['content']}\n"
                    elif msg["role"] == "user":
                        input_text += f"User: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        input_text += f"Assistant: {msg['content']}\n"
                input_text += "Assistant: "
                msg_tokenized = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            terminators = [tokenizer.eos_token_id]
            if hasattr(tokenizer, 'convert_tokens_to_ids'):
                eot_token = tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if eot_token is not None:
                    terminators.append(eot_token)
            
            output_ids = model.generate(
                msg_tokenized if hasattr(msg_tokenized, 'shape') else msg_tokenized['input_ids'],
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            input_length = msg_tokenized.shape[-1] if hasattr(msg_tokenized, 'shape') else msg_tokenized['input_ids'].shape[-1]
            return tokenizer.decode(
                output_ids[0][input_length:], skip_special_tokens=True
            ).strip()
            
        except Exception as e:
            print(f"Local model generation error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(2)


def check_file_exists(save_file, total_len):
    if os.path.exists(
        save_file,
    ):
        with open(
            save_file,
            "r",
        ) as infile:
            already_saved_data = json.load(infile)
        if len(already_saved_data) == total_len:
            print(f"Already saved enough data of {total_len}, Skipping evaluation.")
            return True
        else:
            print("only have ", len(already_saved_data))
            return False
    return False


def print_conversation(messages):
    for message in messages:
        role = message["role"]
        content = message["content"]
        print(f"{role.capitalize()}: {content}\n")
        print()


def generate_message(
    bedrock_runtime,
    model_name,
    model_type,
    system_prompt=None,
    messages=None,
    max_tokens=None,
    temperature=0,
    max_retries=10,
    device="cuda:0",
):
    model_id, _ = get_model_info(model_name)
    # Check if this is a local model
    if "local" in model_name:
        return generate_message_local(
            model_id=model_id,
            device=device,
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
        )
    
    retries = 0
    while retries < max_retries:
        try:
            if model_type == "claude":
                body = json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": max_tokens,
                        "system": system_prompt,
                        "messages": messages,
                        "temperature": temperature,
                    }
                )
                response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
                response_body = json.loads(response.get("body").read())
                return response_body["content"][0]["text"]

            elif model_type == "mistral":
                native_request = {
                    "prompt": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                request = json.dumps(native_request)
                response = bedrock_runtime.invoke_model(modelId=model_id, body=request)
                model_response = json.loads(response["body"].read())
                outputs = model_response.get("outputs")
                response_text = outputs[0]["text"]
                return response_text

            elif model_type == "llama":
                native_request = {
                    "prompt": messages,
                    "max_gen_len": max_tokens,
                    "temperature": temperature,
                }
                request = json.dumps(native_request)
                response = bedrock_runtime.invoke_model(modelId=model_id, body=request)
                model_response = json.loads(response["body"].read())
                response_text = model_response["generation"]
                return response_text

            elif model_type in ["gemma", "qwen", "olmo"]:
                native_request = {
                    "prompt": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                request = json.dumps(native_request)
                response = bedrock_runtime.invoke_model(modelId=model_id, body=request)
                model_response = json.loads(response["body"].read())
                
                # Handle different response formats for these model types
                if "outputs" in model_response:
                    response_text = model_response["outputs"][0]["text"]
                elif "generation" in model_response:
                    response_text = model_response["generation"]
                elif "text" in model_response:
                    response_text = model_response["text"]
                else:
                    # Fallback - try to find text in the response
                    response_text = str(model_response)
                
                return response_text
            elif model_type == "gpt":
                client = OpenAI()  # Use the client passed or create a new one
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    # temperature=temperature,
                    # max_tokens=max_tokens,
                )
                return completion.choices[0].message.content
            elif model_type == "gemini":
                client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))
                config = types.GenerateContentConfig(
                    system_instruction=messages["system_instruction"],
                    # max_output_tokens=max_tokens,
                    # temperature=temperature,
                )
                response = client.models.generate_content(
                    model=model_id,
                    contents=messages["messages"],
                    config=config,
                )
                return response.text
            else:
                raise ValueError(f"Invalid model_type: {model_type}")

        except Exception as e:
            print(e, "retrying time:", retries, model_type)
            if "reduce" in str(e):
                raise Exception(f"max context length is exceeded")
            if retries == max_retries - 1:
                time.sleep(20)
                print("sleeping 20 seconds")
                retries = 0
            retries += 1
            time.sleep(5)  # Wait for 10 seconds before retrying


def extract_multi_turn_conversation(multi_turn_message, turn_number=3, model_type="llama", model_id=None):
    """
    Extract multi-turn conversation for different model types.
    Uses tokenizer.apply_chat_template() when available.
    """
    from .chat_template_utils import get_tokenizer_for_model_type, apply_chat_template_if_available
    
    message = []
    standard_messages = []
    
    for turn in multi_turn_message:
        role = turn["role"]
        content = turn["content"]
        standard_messages.append({"role": role, "content": content})
        
        # For new models, try to use tokenizer chat template first
        if model_type in ["gemma", "qwen", "olmo"]:
            # We'll handle this after collecting all messages
            pass
        elif model_type == "llama":
            message.append(f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>")
        elif model_type == "claude":
            message.append({"role": role, "content": content})
        elif model_type == "mistral":
            if role == "user":
                message.append(f"[INST] {content} [/INST]")
            else:
                message.append(f"{content}</s>")
        elif model_type == "gpt":
            message.append({"role": role, "content": content})
        elif model_type == "gemini":
            gemini_role = {"user": "user", "assistant": "model"}.get(role, "user")
            message.append({"role": gemini_role, "parts": [{"text": str(content)}]})
        else:
            raise ValueError(f"Invalid model_type: {model_type}")
        
        if len(standard_messages) == turn_number * 2:
            if role != "assistant":
                raise ValueError("The last turn must be from assistant")
            break
    
    assert len(standard_messages) == turn_number * 2, "The number of turns is less than the specified number"
    
    # For new models, try to use tokenizer chat template
    if model_type in ["gemma", "qwen", "olmo"]:
        tokenizer = get_tokenizer_for_model_type(model_type, model_id)
        template_result = apply_chat_template_if_available(
            tokenizer, standard_messages, add_generation_prompt=False, tokenize=False
        )
        if template_result is not None:
            return template_result
        
        # Fallback to manual formatting
        from .chat_template_utils import format_conversation_for_model
        return format_conversation_for_model(standard_messages, model_type, tokenizer)
    
    if model_type in ["llama", "mistral"]:
        message = "".join(message)
    return message


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Use the dir_path from the config file instead of overriding it
    # If dir_path is not in config, fall back to the old behavior
    if "dir_path" not in config:
        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)  # This removes the last directory
        config["dir_path"] = parent_dir
    
    return config


def extract_multi_turn_message(turns_data, args, model_type):
    if args.inter_turns > 0:
        multi_turn_message = []
        for turn_data in turns_data:
            multi_turn_message.extend(turn_data["conversation"])
        return (
            extract_multi_turn_conversation(multi_turn_message, args.inter_turns, model_type=model_type),
            multi_turn_message,
        )
    else:
        multi_turn_message = None
    return "", multi_turn_message


def get_model_info(model_name):
    if model_name == "claude3s":
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    elif model_name == "claude3.5s":
        model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    elif model_name == "llama38b":
        model_id = "meta.llama3-8b-instruct-v1:0"
    elif model_name == "claude2":
        model_id = "anthropic.claude-v2:1"
    elif model_name == "claude3hk":
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    elif model_name == "llama3-8b":
        model_id = "meta.llama3-8b-instruct-v1:0"
    elif model_name == "llama370b":
        model_id = "meta.llama3-70b-instruct-v1:0"
    elif model_name == "llama318b":
        model_id = "meta.llama3-1-8b-instruct-v1:0"
    elif model_name == "llama3170b":
        model_id = "meta.llama3-1-70b-instruct-v1:0"
    elif model_name == "llama31405b":
        model_id = "meta.llama3-1-405b-instruct-v1:0"
    elif model_name == "mistral-7b":
        model_id = "mistral.mistral-7b-instruct-v0:2"
    elif model_name == "mistral8x7b":
        model_id = "mistral.mixtral-8x7b-instruct-v0:1"
    elif model_name == "mistrallarge":
        model_id = "mistral.mistral-large-2402-v1:0"
    elif model_name == "mistrallarge2":
        model_id = "mistral.mistral-large-2407-v1:0"

    elif model_name == "gpt4o-mini":
        model_id = "gpt-4o-mini"
    elif model_name == "gpt-o1":
        model_id = "o1-preview-2024-09-12"
    elif model_name == "gpt-o3":
        model_id = "o3-2025-04-16"
    elif model_name == "gpt-o4-mini":
        model_id = "o4-mini-2025-04-16"
    elif model_name == "gemini2.5-flash":
        model_id = "gemini-2.5-flash-preview-04-17"
    elif model_name == "gemini2.5-pro":
        model_id = "gemini-2.5-pro-exp-03-25"
        
    # although in model catelog, still not directly available
    elif model_name == "gemma-7b":
        model_id = "huggingface-llm-gemma-7b-instruct"
    elif model_name == "gemma2-9b":
        model_id = "huggingface-llm-gemma-2-9b-instruct"
    elif model_name == "gpt-j-6b":
        model_id = "huggingface-textgeneration1-gpt-j-6b"

    elif model_name == "llama3-8b-local":
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_name == "mistral-7b-local":
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    elif model_name == "qwen3-8b-local":
        model_id = "Qwen/Qwen3-8B"
    elif model_name == "deepseek-7b-local":
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    elif model_name == "gemma-7b-local":
        model_id = "google/gemma-1.1-7b-it"
    elif model_name == "gemma2-9b-local":
        model_id = "google/gemma-2-9b-it"
    elif model_name == "olmo2-7b-local":
        model_id = "allenai/OLMo-2-1124-7B-Instruct"
    elif model_name == "gpt-j-6b-local":
        model_id = "EleutherAI/gpt-j-6b"
    

    # model_type detection to include gemini and new models
    model_type = (
        "mistral"
        if "mistral" in model_name
        else (
            "claude"
            if "claude" in model_name
            else (
                "llama"
                if "llama" in model_name
                else (
                    "gpt" 
                    if "gpt" in model_name 
                    else (
                        "gemini" 
                        if "gemini" in model_name 
                        else (
                            "gemma"
                            if "gemma" in model_name
                            else (
                                "qwen"
                                if "qwen" in model_name or "deepseek" in model_name
                                else (
                                    "olmo"
                                    if "olmo" in model_name
                                    else None
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    return model_id, model_type


ALL_TOPICS = [
    "travel_transportation",
    "shop_motors",
    "lifestyle_beauty",
    "travel_restaurant",
    "shop_fashion",
    "entertain_shows",
    "pet_ownership",
    "lifestyle_fit",
    "entertain_games",
    "shop_home",
    "lifestyle_health",
    "travel_activities",
    "education_learning_styles",
    "entertain_music_book",
    "professional_work_location_style",
    "education_resources",
    "lifestyle_dietary",
    "shop_technology",
    "travel_hotel",
    "entertain_sports",
]


COT_PROMPT = """
When answering a user's question, a good assistant should carefully consider the user's stated preferences and tailor the response accordingly.

Example 1:
User1's preference: "I have an intense phobia of bugs and insects. What outdoor activities would you recommend for the summer?"
Good assistant respnose: "Considering your phobia of bugs and insects, I'll focus on recommending activities that minimize encounters with insects. Let's explore outdoor options with fewer bugs, like beaches, pools, urban areas, well-maintained park trails, and activities during cooler times of day. I'll avoid suggesting things like camping, off-trail hiking, picnicking, or activities in dense forests. Here are some summer outdoor activities you might enjoy: spending a day at the beach swimming and playing sports, exploring outdoor malls or pedestrian areas in the city, hiking on maintained trails in parks (avoiding dense woods), attending evening outdoor concerts or movies, and enjoying water activities like kayaking or paddle boating on lakes or rivers. These options should help you enjoy the outdoors while minimizing encounters with insects."
Example 2:
User2's preference: "I'm a vegan looking for new dinner recipes. Any ideas?"
Good assistant response: "As a vegan, you'll need plant-based recipes without animal products. I'll focus on nutrient-rich, diverse ingredients to ensure balanced meals. Consider: quinoa and black bean burrito bowls, lentil and vegetable curry, roasted vegetable and hummus wrap, or zucchini noodles with avocado pesto. These options offer protein, fiber, and essential nutrients while adhering to vegan principles."
Example 3:
User3's preference: "I'm a beginner guitarist with small hands. What's a good first guitar?"
Good assistant response: "For a beginner with small hands, we'll prioritize comfort and playability. Look for guitars with narrow necks and shorter scale lengths. Consider: Fender Mustang, Ibanez Mikro, or Squier Mini Strat. These models are smaller overall, making them easier to handle. Avoid full-sized dreadnoughts or jumbo acoustics. Try these at a local music store to ensure they feel comfortable before purchasing."
Example 4:
User4's preference: "I'm on a tight budget but want to start a home gym. What equipment should I get?"
Good assistant response: "Working with a tight budget, we'll focus on versatile, cost-effective equipment for a home gym. Prioritize items that offer multiple exercise options. Consider: resistance bands, a set of adjustable dumbbells, a stability ball, and a yoga mat. These allow for a wide range of strength, flexibility, and cardio exercises without breaking the bank. Avoid expensive, single-purpose machines. As your budget allows, you can add more specialized equipment later."
Example 5:
User5's preference: "I'm sensitive to caffeine but need an energy boost in the mornings. Any suggestions?"
Good assistant response: "Given your caffeine sensitivity, we'll explore natural, caffeine-free energy boosters. Focus on lifestyle changes and alternatives to caffeinated drinks. Try: starting your day with a brisk walk or light exercise, drinking a glass of water with lemon, eating a protein-rich breakfast, or trying herbal teas like peppermint or ginger. Avoid coffee, energy drinks, and black tea. These alternatives can help increase alertness without the jitters or crashes associated with caffeine."

Now, please answer the following question while considering my preferences (not the user preferences in the examples above), which I have stated either explicitly or implicitly in our previous conversation:

            """
REMINDER = "\nIn your response, please ensure that you take into account our earlier discussion, and provide an answer that is consistent with my preference."
