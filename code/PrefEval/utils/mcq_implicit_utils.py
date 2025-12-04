import os
import json
import time
from .utils_mcq import format_options
from .chat_template_utils import (
    format_conversation_for_model,
    create_messages_with_system,
    create_messages_without_system,
    get_tokenizer_for_model_type,
    apply_chat_template_if_available
)

COT = """
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


def get_mcq_question_format(options):
    """
    Generate the question format with provided options.

    :param options: List of options to be included in the question format
    :return: A formatted MCQ question string
    """
    formatted_options = format_options(options)  # Define formatted_options here

    mcq_question_format = f"""
    I'm trying to decide on this and here are 4 options for my query: \n{formatted_options}\nNow, I'd like you to pick one of them as your top recommendation for me.
    Important instructions for your response:
    1. Choose only one option (A, B, C, or D) that best matches my preferences.
    2. Your answer must be one of these options.
    3. Don't say things like "I can't choose" or suggest alternatives not listed.
    4. Answer example: <choice>B</choice>. Give me your answer in this exact format, without any additional explanation:
       <choice>[A/B/C/D]</choice>
    """

    return mcq_question_format


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


def load_files_implicit(args):
    # system prompt:
    with open(
        f"/data/siyanz-sandbox/preference_dataset/implicit_preference_benchmark/type_{args.mcq}/{args.topic}.json",
        "r",
    ) as infile:
        existing_data = json.load(infile)
    # check if this directory does not exist, create one:
    dir_path = f"/data/siyanz-sandbox/benchmark_results/implicit/{args.task}_{args.mcq}/{args.topic}/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    save_file = f"/data/siyanz-sandbox/benchmark_results/implicit/{args.task}_{args.mcq}/{args.topic}/{args.model}_{args.topic}_{args.inter_turns+2}turn.json"
    return existing_data, save_file


def load_files_implicit_mcq(args):
    # system prompt:
    if args.pref_type == "imp_mcq":
        with open(
            f"/data/siyanz-sandbox/preference_dataset/implicit_preference_benchmark/type_{args.mcq}/{args.topic}.json",
            "r",
        ) as infile:
            topic_data = json.load(infile)
    elif args.pref_type == "imp_persona":
        with open(
            f"/data/siyanz-sandbox/preference_dataset/implicit_preference_benchmark/persona_conversation/{args.topic}.json",
            "r",
        ) as infile:
            topic_data = json.load(infile)
    # check if this directory does not exist, create one:
    dir_path = (
        f"/data/siyanz-sandbox/benchmark_results/mcq_results/{args.pref_type}_{args.mcq}_{args.task}/{args.topic}/"
    )
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    save_file = f"/data/siyanz-sandbox/benchmark_results/mcq_results/{args.pref_type}_{args.mcq}_{args.task}/{args.topic}/{args.model}_{args.topic}_{args.inter_turns+2}turn.json"
    return topic_data, save_file


def load_files_implicit_rag(args):
    if args.pref == "mcq":
        with open(
            f"/data/siyanz-sandbox/preference_dataset/implicit_preference_benchmark/type_a/{args.topic}.json",
            "r",
        ) as infile:
            topic_data = json.load(infile)
    elif args.pref == "persona":
        with open(
            f"/data/siyanz-sandbox/preference_dataset/implicit_preference_benchmark/persona_conversation/{args.topic}.json",
            "r",
        ) as infile:
            topic_data = json.load(infile)
    dir_path = f"/data/siyanz-sandbox/benchmark_results/implicit/{args.task}_{args.pref}/{args.topic}/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    save_file = f"/data/siyanz-sandbox/benchmark_results/implicit/{args.task}_{args.pref}/{args.topic}/{args.model}_{args.topic}_{args.inter_turns+2}turn.json"

    with open(
        "/data/siyanz-sandbox/lmsys-chat-1m/filtered_inter_turns.json",
        "r",
    ) as infile:
        turns_data = json.load(infile)

    with open(
        f"/data/siyanz-sandbox/preference_dataset/simcse_top15_retrived_history_implicit_{args.pref}/{args.topic}_overall300_topk_history_{args.pref}.json",
        "r",
    ) as infile:
        rag_data = json.load(infile)

    with open(
        f"/data/siyanz-sandbox/preference_dataset/simcse_question_inter_similarities/{args.topic}_300_inter_similarities.json",
        "r",
    ) as infile:
        msg_rag_data = json.load(infile)
    return topic_data, save_file, turns_data, rag_data, msg_rag_data


def load_files_implicit_persona(args):
    # system prompt:
    with open(
        f"/data/siyanz-sandbox/preference_dataset/implicit_preference_benchmark/persona_conversation/{args.topic}.json",
        "r",
    ) as infile:
        existing_data = json.load(infile)
    # check if this directory does not exist, create one:
    dir_path = f"/data/siyanz-sandbox/benchmark_results/implicit/{args.task}/{args.topic}/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    save_file = f"/data/siyanz-sandbox/benchmark_results/implicit/{args.task}/{args.topic}/{args.model}_{args.topic}_{args.inter_turns+2}turn.json"
    return existing_data, save_file


def load_files_implicit_selfcritic(args):
    # system prompt:
    if args.pref_type == "mcq":
        with open(
            f"/data/siyanz-sandbox/benchmark_results/mcq_results/imp_mcq_{args.mcq}_zero-shot/{args.topic}/{args.model}_{args.topic}_{args.inter_turns+2}turn.json",
            "r",
        ) as infile:
            existing_data = json.load(infile)
    elif args.pref_type == "persona":
        with open(
            f"/data/siyanz-sandbox/benchmark_results/mcq_results/imp_persona_{args.mcq}_zero-shot/{args.topic}/{args.model}_{args.topic}_{args.inter_turns+2}turn.json",
            "r",
        ) as infile:
            existing_data = json.load(infile)
    # check if this directory does not exist, create one:
    dir_path = f"/data/siyanz-sandbox/benchmark_results/implicit/{args.task}_{args.mcq}/{args.topic}/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    save_file = f"/data/siyanz-sandbox/benchmark_results/implicit/{args.task}_{args.mcq}/{args.topic}/{args.model}_{args.topic}_{args.inter_turns+2}turn.json"
    return existing_data, save_file


def load_files_implicit_selfcritic_mcq(args):
    # system prompt:
    if args.pref_type == "mcq":
        with open(
            f"/data/siyanz-sandbox/benchmark_results/mcq_results/imp_mcq_{args.mcq}_zero-shot/{args.topic}/{args.model}_{args.topic}_{args.inter_turns+2}turn.json",
            "r",
        ) as infile:
            existing_data = json.load(infile)
    elif args.pref_type == "persona":
        with open(
            f"/data/siyanz-sandbox/benchmark_results/mcq_results/imp_persona_{args.mcq}_zero-shot/{args.topic}/{args.model}_{args.topic}_{args.inter_turns+2}turn.json",
            "r",
        ) as infile:
            existing_data = json.load(infile)
    # check if this directory does not exist, create one:
    dir_path = f"/data/siyanz-sandbox/benchmark_results/mcq_results/imp_{args.pref_type}_{args.task}/{args.topic}/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    save_file = f"/data/siyanz-sandbox/benchmark_results/mcq_results/imp_{args.pref_type}_{args.task}/{args.topic}/{args.model}_{args.topic}_{args.inter_turns+2}turn.json"
    return existing_data, save_file


def generate_message(
    bedrock_runtime,
    model_id,
    model_type,
    system_prompt=None,
    messages=None,
    max_tokens=None,
    temperature=0,
    max_retries=10,
):
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

            else:
                raise ValueError(f"Invalid model_type: {model_type}")

        except Exception as e:
            print(e, "retrying time:", retries, model_id)
            if "reduce" in str(e):
                raise Exception(f"max context length is exceeded")
            if retries == max_retries - 1:
                time.sleep(30)
                print("sleeping 60 seconds")
                retries = 0
            retries += 1
            time.sleep(10)  # Wait for 10 seconds before retrying


def extract_multi_turn_conversation(multi_turn_message, turn_number=3, model_type="llama"):
    message = []
    for turn in multi_turn_message:
        role = turn["role"]
        content = turn["content"]
        if model_type == "llama":
            message.append(f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>")
        elif model_type == "claude":
            message.append({"role": role, "content": content})
        elif model_type == "mistral":
            if role == "user":
                message.append(f"[INST] {content} [/INST]")
            else:
                message.append(f"{content}</s>")
        if len(message) == turn_number * 2:
            if role != "assistant":
                raise ValueError("The last turn must be from assistant")
            break
    assert len(message) == turn_number * 2, "The number of turns is less than the specified number"
    if "llama" in model_type or "mistral" in model_type:
        message = "".join(message)
    return message


def extract_conversation_to_messages(conversation, model_type):
    # this is for implicit preference
    messages = []
    role_list = []

    for key, content in conversation.items():
        if model_type == "mistral":
            if key == "query" or key == "user_selection":
                messages.append(f"[INST] {content} [/INST]")
            else:
                messages.append(f"{content}</s>")
        elif model_type == "llama":
            if key == "query" or key == "user_selection":
                messages.append(f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>")
            else:
                messages.append(f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>")
        elif model_type == "claude":
            if key == "query" or key == "user_selection":
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "assistant", "content": content})
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

    return messages, role_list


def extract_conversation_to_msg_persona(conversation, model_type, model_id=None):
    """
    Extract conversation to messages for persona-driven conversations.
    Uses tokenizer.apply_chat_template() when available.
    """
    messages = []
    
    # For new models, try to use tokenizer chat template first
    if model_type in ["gemma", "qwen", "olmo"]:
        tokenizer = get_tokenizer_for_model_type(model_type, model_id)
        
        # Build standard conversation messages
        standard_messages = []
        for turn_idx, turn in conversation.items():
            for role, content in turn.items():
                standard_messages.append({"role": role, "content": content})
        
        # Try tokenizer chat template
        template_result = apply_chat_template_if_available(
            tokenizer, standard_messages, add_generation_prompt=False, tokenize=False
        )
        if template_result is not None:
            return [template_result]
        
        # Fallback to manual formatting
        return [format_conversation_for_model(standard_messages, model_type, tokenizer)]
    
    # Original logic for existing models
    for turn_idx, turn in conversation.items():
        for role, content in turn.items():
            if model_type == "mistral":
                if role == "user":
                    messages.append(f"[INST] {content} [/INST]")
                elif role == "assistant":
                    messages.append(f"{content}</s>")
            elif model_type == "claude":
                if role == "user":
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({"role": "assistant", "content": content})
            elif model_type == "llama":
                if role == "user":
                    messages.append(f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>")
                else:
                    messages.append(f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>")
            else:
                raise ValueError(f"Invalid model_type: {model_type}")
    assert len(messages) == (int(turn_idx) + 1) * 2
    return messages


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
    elif model_name == "llama38b":
        model_id = "meta.llama3-8b-instruct-v1:0"
    elif model_name == "llama370b":
        model_id = "meta.llama3-70b-instruct-v1:0"
    elif model_name == "llama318b":
        model_id = "meta.llama3-1-8b-instruct-v1:0"
    elif model_name == "llama3170b":
        model_id = "meta.llama3-1-70b-instruct-v1:0"
    elif model_name == "llama31405b":
        model_id = "meta.llama3-1-405b-instruct-v1:0"
    elif model_name == "mistral7b":
        model_id = "mistral.mistral-7b-instruct-v0:2"
    elif model_name == "mistral8x7b":
        model_id = "mistral.mixtral-8x7b-instruct-v0:1"
    elif model_name == "mistrallarge":
        model_id = "mistral.mistral-large-2402-v1:0"
    elif model_name == "mistrallarge2":
        model_id = "mistral.mistral-large-2407-v1:0"

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
                    "gemma"
                    if "gemma" in model_name
                    else (
                        "qwen"
                        if "qwen" in model_name
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
    return model_id, model_type


def create_user_pref_message(preference, model_type, model_id=None):
    """
    Create user preference message for different model types.
    Uses tokenizer.apply_chat_template() when available.
    """
    system_prompt = "You are a helpful AI assistant."
    
    # For new models, try to use tokenizer chat template first
    if model_type in ["gemma", "qwen", "olmo"]:
        tokenizer = get_tokenizer_for_model_type(model_type, model_id)
        messages = create_messages_with_system(system_prompt, preference)
        
        template_result = apply_chat_template_if_available(
            tokenizer, messages, add_generation_prompt=True, tokenize=False
        )
        if template_result is not None:
            return template_result
        
        # Fallback to manual formatting
        return format_conversation_for_model(
            [{"role": "user", "content": preference}], 
            model_type, 
            tokenizer, 
            system_prompt
        )
    
    # Original logic for existing models
    if model_type == "claude":
        user_message = [{"role": "user", "content": preference}]
    elif model_type == "llama":
        user_message = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant. <|eot_id|><|start_header_id|>user<|end_header_id|>

{preference}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    elif model_type == "mistral":
        user_message = f"<s>[INST] {preference} [/INST]"
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return user_message


def get_question_prompt(preference, pref_generation, question, multi_inter_message, model_type, model_id=None):
    """
    Get question prompt for different model types.
    Uses tokenizer.apply_chat_template() when available.
    """
    system_prompt = "You are a helpful AI assistant."
    
    # For new models, try to use tokenizer chat template first
    if model_type in ["gemma", "qwen", "olmo"]:
        tokenizer = get_tokenizer_for_model_type(model_type, model_id)
        
        # Build conversation messages
        messages = [
            {"role": "user", "content": preference},
            {"role": "assistant", "content": pref_generation}
        ]
        
        if multi_inter_message:
            if isinstance(multi_inter_message, list):
                messages.extend(multi_inter_message)
            else:
                messages.append({"role": "user", "content": str(multi_inter_message)})
        
        messages.append({"role": "user", "content": question})
        
        # Try tokenizer chat template
        template_result = apply_chat_template_if_available(
            tokenizer,
            [{"role": "system", "content": system_prompt}] + messages,
            add_generation_prompt=True,
            tokenize=False
        )
        if template_result is not None:
            return template_result
        
        # Fallback to manual formatting
        return format_conversation_for_model(messages, model_type, tokenizer, system_prompt)
    
    # Original logic for existing models
    if model_type == "claude":
        user_message = {"role": "user", "content": preference}
        messages = [user_message, {"role": "assistant", "content": pref_generation}]
        if multi_inter_message:
            messages.extend(multi_inter_message)
        messages.append({"role": "user", "content": question})
    elif model_type == "llama":
        messages = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{preference}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{pref_generation}
<|eot_id|>
{multi_inter_message}
<|start_header_id|>user<|end_header_id|>
{question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    elif model_type == "mistral":
        messages = f"""<s>[INST]
You are a helpful AI assistant.
{preference}
[/INST]
{pref_generation}</s>
{multi_inter_message}
[INST]
{question}
[/INST]"""
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return messages


def get_implicit_question_prompt(
    conversation_messages,
    question,
    multi_inter_message,
    model_type,
    turn_number,
    remind,
    cot,
):
    if remind:
        reminder = "\nIn your response, please ensure that you take into account our earlier discussion, and provide an answer that is consistent with my preference."
    else:
        reminder = ""
    if cot:
        cot_prompt = COT
    else:
        cot_prompt = ""
    if model_type == "claude":
        messages = conversation_messages
        if turn_number > 0:
            messages.extend(multi_inter_message)
        messages.append({"role": "user", "content": cot_prompt + question + reminder})
    elif model_type == "llama":
        if turn_number == 0:
            multi_inter_message = ""
        conversation_messages = "".join(conversation_messages)
        messages = f"""<|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are a helpful AI assistant.
        <|eot_id|>
        {conversation_messages}
        {multi_inter_message}
        <|start_header_id|>user<|end_header_id|>
        {cot_prompt}{question}{reminder}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>"""
    elif model_type == "mistral":
        conversation_messages = "".join(conversation_messages)
        if turn_number == 0:
            multi_inter_message = ""
        messages = (
            messages
        ) = f"""<s>[INST]
You are a helpful AI assistant.
{conversation_messages[6:]}
{multi_inter_message}
[INST]
{cot_prompt}{question}{reminder}
[/INST]"""
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return messages


def get_implicit_cot_question_prompt(
    conversation_messages,
    question,
    multi_inter_message,
    model_type,
    turn_number,
):
    cot = """
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
    if model_type == "claude":
        messages = conversation_messages
        if turn_number > 0:
            messages.extend(multi_inter_message)
        messages.append({"role": "user", "content": cot + question})
    elif model_type == "llama":
        if turn_number == 0:
            multi_inter_message = ""
        conversation_messages = "".join(conversation_messages)
        messages = f"""<|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are a helpful AI assistant.
        <|eot_id|>
        {conversation_messages}
        {multi_inter_message}
        <|start_header_id|>user<|end_header_id|>
        {cot}{question}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>"""
    elif model_type == "mistral":
        conversation_messages = "".join(conversation_messages)
        if turn_number == 0:
            multi_inter_message = ""
        messages = (
            messages
        ) = f"""<s>[INST]
You are a helpful AI assistant.
{conversation_messages[6:]}
{multi_inter_message}
[INST]
{cot}{question}
[/INST]"""
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return messages


def get_self_critic_prompt_critic_mcq_implicit(
    args,
    conversation_messages,
    critic_request,
    response_to_q,
    multi_inter_message,
    question,
    options,
):
    mcq_question_format = get_mcq_question_format(options)
    question = question + mcq_question_format
    if "claude" in args.model:
        critic_messages = conversation_messages
        if args.inter_turns > 0:
            critic_messages.extend(multi_inter_message)
        critic_messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response_to_q},
            ]
        )
        critic_messages.append({"role": "user", "content": critic_request})
    elif "llama" in args.model:
        if args.inter_turns == 0:
            multi_inter_message = ""
        conversation_messages = "".join(conversation_messages)
        critic_messages = f"""<|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are a helpful AI assistant.
        <|eot_id|>
        {conversation_messages}
        <|eot_id|>
        {multi_inter_message}
        <|start_header_id|>user<|end_header_id|>
        {question}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        {response_to_q}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {critic_request}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """
    elif "mistral" in args.model:

        conversation_messages = "".join(conversation_messages)
        if args.inter_turns == 0:
            multi_inter_message = ""
        critic_messages = f"""<s>[INST]
You are a helpful AI assistant.
{conversation_messages[6:]}
[/INST]
{multi_inter_message}
[INST]
{question}
[/INST]
{response_to_q}</s>
[INST]
{critic_request}
[/INST]
"""
    return critic_messages


def get_self_critic_prompt_response(
    args,
    conversation_messages,
    critic_request,
    response_to_q,
    multi_inter_message,
    question,
    critic,
    revision_request,
):
    if "claude" in args.model:

        critic_messages = conversation_messages
        if args.inter_turns > 0:
            critic_messages.extend(multi_inter_message)
        critic_messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response_to_q},
            ]
        )
        critic_messages.append({"role": "user", "content": critic_request})
        critic_messages.append({"role": "assistant", "content": critic})
        critic_messages.append({"role": "user", "content": revision_request})
    elif "llama" in args.model:
        if args.inter_turns == 0:
            multi_inter_message = ""
        conversation_messages = "".join(conversation_messages)
        critic_messages = f"""<|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are a helpful AI assistant.
        <|eot_id|>
        {conversation_messages}
        <|eot_id|>
        {multi_inter_message}
        <|start_header_id|>user<|end_header_id|>
        {question}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        {response_to_q}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {critic_request}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        {critic}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {revision_request}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """
    elif "mistral" in args.model:
        conversation_messages = "".join(conversation_messages)
        if args.inter_turns == 0:
            multi_inter_message = ""
        critic_messages = f"""<s>[INST]
You are a helpful AI assistant.
{conversation_messages[6:]}

{multi_inter_message}
[INST]
{question}
[/INST]
{response_to_q}</s>
[INST]
{critic_request}
[/INST]
{critic}</s>
[INST]
{revision_request}
[/INST]
"""
    return critic_messages


def get_self_critic_prompt_response_mcq_implicit(
    args,
    options,
    model_type,
    conversation_messages,
    critic_request,
    response_to_q,
    multi_inter_message,
    question,
    critic,
    revision_request,
):

    mcq_question_format = get_mcq_question_format(options)
    question = question + mcq_question_format
    revision_request += mcq_question_format
    if "claude" in args.model:

        critic_messages = conversation_messages
        if args.inter_turns > 0:
            critic_messages.extend(multi_inter_message)
        critic_messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response_to_q},
            ]
        )
        critic_messages.append({"role": "user", "content": critic_request})
        critic_messages.append({"role": "assistant", "content": critic})
        critic_messages.append({"role": "user", "content": revision_request})
        critic_messages.append({"role": "assistant", "content": "<choice>"})
    elif "llama" in args.model:
        if args.inter_turns == 0:
            multi_inter_message = ""
        conversation_messages = "".join(conversation_messages)
        critic_messages = f"""<|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are a helpful AI assistant.
        <|eot_id|>
        {conversation_messages}
        <|eot_id|>
        {multi_inter_message}
        <|start_header_id|>user<|end_header_id|>
        {question}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        {response_to_q}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {critic_request}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        {critic}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {revision_request}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """
    elif "mistral" in args.model:
        conversation_messages = "".join(conversation_messages)
        if args.inter_turns == 0:
            multi_inter_message = ""
        critic_messages = f"""<s>[INST]
You are a helpful AI assistant.
{conversation_messages[6:]}

{multi_inter_message}
[INST]
{question}
[/INST]
{response_to_q}</s>
[INST]
{critic_request}
[/INST]
{critic}</s>
[INST]
{revision_request}
[/INST]
<choice>
"""
    return critic_messages


def convert_top_k_sentences_to_msg(top_k_sentences):
    msg = (
        "Before answering my question, please consider the following context from our previous conversations. "
        "These are the 5 most relevant exchanges that we had previously, which may contain information about "
        "my preferences or prior discussions related to my query:\n\n"
        "#Start of Context#\n"
    )
    for idx, sentence in enumerate(top_k_sentences, 1):
        msg += f"exchange {idx}. {sentence}\n"

    msg += (
        "#End of Context#\n\n"
        "Please use this context to inform your answer and adhere to any preferences I've expressed "
        "that are relevant to the current query. Note that not all contexts are useful for answering "
        "my question and there may be no context that is useful. Now, please address my question:\n\n"
    )
    return msg


def get_implicit_question_prompt_rag(
    conversation_messages,
    question,
    multi_inter_message,
    model_type,
    turn_number,
    top_k_sentences,
    system_prompt="You are a helpful asisstant.",
):
    top_k_msg = convert_top_k_sentences_to_msg(top_k_sentences)
    question = top_k_msg + question
    if model_type == "claude":
        messages = conversation_messages
        if multi_inter_message:
            assert turn_number > 0
            messages.extend(multi_inter_message)
        messages.append({"role": "user", "content": question})
    elif model_type == "llama":
        conversation_messages = "".join(conversation_messages)
        if multi_inter_message is None:
            assert turn_number <= 0
            multi_inter_message = ""
        messages = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|>
{conversation_messages}
{multi_inter_message}
<|start_header_id|>user<|end_header_id|>
{question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    elif model_type == "mistral":
        conversation_messages = "".join(conversation_messages)
        if multi_inter_message is None:
            assert turn_number <= 0
            multi_inter_message = ""
        messages = f"""<s>[INST]
{system_prompt}
{conversation_messages[6:]}
{multi_inter_message}
[INST]
{question}
[/INST]"""
    elif model_type == "gpt":
        system_prompt = {"role": "system", "content": system_prompt}
        messages = [system_prompt, conversation_messages]
        if multi_inter_message:
            assert turn_number > 0
            messages.extend(multi_inter_message)
        messages.append({"role": "user", "content": question})
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return messages


def get_implicit_question_prompt_rag_mcq(
    conversation_messages,
    question,
    multi_inter_message,
    model_type,
    turn_number,
    top_k_sentences,
    system_prompt="You are a helpful asisstant.",
    options="",
):

    mcq_question_format = get_mcq_question_format(options)
    top_k_msg = convert_top_k_sentences_to_msg(top_k_sentences)
    question = top_k_msg + question + mcq_question_format
    if model_type == "claude":
        messages = conversation_messages
        if multi_inter_message:
            assert turn_number > 0
            messages.extend(multi_inter_message)
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": "<choice>"})
    elif model_type == "llama":
        conversation_messages = "".join(conversation_messages)
        if multi_inter_message is None:
            assert turn_number <= 0
            multi_inter_message = ""
        messages = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|>
{conversation_messages}
{multi_inter_message}
<|start_header_id|>user<|end_header_id|>
{question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    elif model_type == "mistral":
        conversation_messages = "".join(conversation_messages)
        if multi_inter_message is None:
            assert turn_number <= 0
            multi_inter_message = ""
        messages = f"""<s>[INST]
{system_prompt}
{conversation_messages[6:]}
{multi_inter_message}
[INST]
{question}
[/INST]
<choice>
"""
    elif model_type == "gpt":
        system_prompt = {"role": "system", "content": system_prompt}
        messages = [system_prompt, conversation_messages]
        if multi_inter_message:
            assert turn_number > 0
            messages.extend(multi_inter_message)
        messages.append({"role": "user", "content": question})
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return messages
