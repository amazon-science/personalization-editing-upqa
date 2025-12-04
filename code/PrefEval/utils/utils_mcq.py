import os
import json
import time
import random
from bs4 import BeautifulSoup
import re
from .common_utils import COT_PROMPT, REMINDER
from .chat_template_utils import (
    format_conversation_for_model,
    create_messages_with_system,
    create_messages_without_system,
    get_tokenizer_for_model_type,
    apply_chat_template_if_available
)
from .common_utils import get_model_info, generate_message_local


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


def extract_choice(response):
    """
    Extract the choice (A, B, C, or D) from the LLM's response.

    :param response: String containing the LLM's response
    :return: The extracted choice or None if no valid choice is found
    """
    # Create a BeautifulSoup object
    soup = BeautifulSoup(response, "html.parser")
    # Find the choice tag
    choice_tag = soup.find("choice")

    try:
        if choice_tag:
            # Extract the content of the choice tag
            choice_content = choice_tag.string
            # Use regex to find the letter
            match = re.search(r"[ABCD]", choice_content)

            if match:
                return match.group(0)
    except:
        return None


def extract_choice_mistral(response):
    """
    Extract the choice (A, B, C, or D) from the LLM's JSON response.

    :param response: String containing the LLM's response
    :return: The extracted choice or None if no valid choice is found
    """
    try:
        # Try to parse the entire response as JSON
        data = json.loads(response.strip())
        if isinstance(data, dict) and "choice" in data:
            choice = data["choice"].strip().upper()
            if choice in ["A", "B", "C", "D"]:
                return choice
    except json.JSONDecodeError:
        # If parsing fails, try to find a JSON object within the response
        import re

        json_match = re.search(r"\{.*\}", response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if isinstance(data, dict) and "choice" in data:
                    choice = data["choice"].strip().upper()
                    if choice in ["A", "B", "C", "D"]:
                        return choice
            except json.JSONDecodeError:
                pass

    # If JSON parsing fails, fall back to searching for A, B, C, or D
    choice_match = re.search(r"\b[ABCD]\b", response, re.IGNORECASE)
    if choice_match:
        return choice_match.group().upper()

    return None


def shuffle_options(options):
    """
    Note: In the MCQ datasets, the first choice in the JSON file is the correct answer.
    This function shuffles the options randomly while keeping track of the correct answer.

    :param options: List of options where the first option is the correct answer
    :return: Tuple containing the shuffled options and the index of the correct answer
    """
    correct_answer = options[0]
    shuffled_options = random.sample(options, len(options))
    correct_index = shuffled_options.index(correct_answer)

    return shuffled_options, correct_index


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


def load_files_explicit(args):
    # system prompt:
    with open(
        f"{args.dir}/preference_dataset/finished_topics/{args.topic}.json",
        "r",
    ) as infile:
        existing_data = json.load(infile)
    # check if this directory does not exist, create one:
    dir_path = f"{args.dir}/benchmark_results/explicit/{args.task}/{args.topic}/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    save_file = f"{args.dir}/benchmark_results/explicit/{args.task}/{args.topic}/{args.model}_{args.topic}_{args.inter_turns+2}turn.json"
    return existing_data, save_file


def load_files_mcq(args):
    # Load topic data
    with open(f"{args.dir}/benchmark_dataset/mcq_options/{args.topic}.json", "r") as infile:
        topic_data = json.load(infile)

    # Define directory path based on preference form
    dir_path = f"{args.dir}/benchmark_results/{args.pref_form}/"
    if args.pref_form != "implicit":
        dir_path += f"mcq_results/{args.task}/{args.topic}/"
    else:
        dir_path += f"{args.pref_type}/mcq_results/{args.task}/{args.topic}/"

    os.makedirs(dir_path, exist_ok=True)

    save_file = f"{dir_path}{args.model}_{args.topic}_{args.inter_turns}interturn.json"

    return topic_data, save_file


def generate_message(
    bedrock_runtime,
    model_name,
    model_type,
    system_prompt=None,
    messages=None,
    max_tokens=None,
    temperature=0,
    max_retries=10,
    device=None,
):
    model_id, _ = get_model_info(model_name)
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

            else:
                raise ValueError(f"Invalid model_type: {model_type}")

        except Exception as e:
            print(e, "retrying time:", retries, model_type)
            if "reduce" in str(e):
                raise Exception(f"max context length is exceeded")
            if retries == max_retries - 1:
                time.sleep(60)
                print("sleeping 60 seconds")
                retries = 0
            retries += 1
            time.sleep(5)  # Wait for 10 seconds before retrying


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
        elif model_type == "gpt":
            message.append({"role": role, "content": content})
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

    for line in conversation:
        role = line["role"]
        content = line["content"]
        role_list.append(role)

        if model_type == "mistral":
            if role == "user":
                messages.append(f"[INST] {content} [/INST]")
            else:
                messages.append(f"{content}</s>")
        elif model_type == "llama":
            messages.append(f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>")
        elif model_type == "claude":
            messages.append({"role": role, "content": content})
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

    return messages, role_list


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


def format_options(options):
    """
    Format a list of options into a lettered string.

    :param options: List of option strings
    :return: Formatted string with lettered options
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    formatted_options = []

    for i, option in enumerate(options):
        formatted_options.append(f"{letters[i]}. {option}")
    return "\n".join(formatted_options)


def get_question_prompt_mcq(
    preference,
    options,
    pref_generation,
    question,
    multi_inter_message,
    model_type,
    turn_number,
    remind,
    cot,
    system_prompt="You are a helpful asisstant.",
    model_id=None,
):
    """
    Get MCQ question prompt for different model types.
    Uses tokenizer.apply_chat_template() when available.
    """
    mcq_question_format = get_mcq_question_format(options)
    if remind:
        question = question + REMINDER
    if cot:
        question = COT_PROMPT + question
    
    full_question = question + mcq_question_format
    
    # For new models, try to use tokenizer chat template first
    if model_type in ["gemma", "qwen", "olmo"]:
        tokenizer = get_tokenizer_for_model_type(model_type, model_id)
        
        # Build conversation messages
        messages = [
            {"role": "user", "content": preference},
            {"role": "assistant", "content": pref_generation}
        ]
        
        if multi_inter_message:
            assert turn_number > 0
            if isinstance(multi_inter_message, list):
                messages.extend(multi_inter_message)
            else:
                messages.append({"role": "user", "content": str(multi_inter_message)})
        
        messages.extend([
            {"role": "user", "content": full_question},
            {"role": "assistant", "content": "<choice>"}
        ])
        
        # Try tokenizer chat template
        template_result = apply_chat_template_if_available(
            tokenizer,
            [{"role": "system", "content": system_prompt}] + messages,
            add_generation_prompt=False,
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
            assert turn_number > 0
            messages.extend(multi_inter_message)
        messages.append({"role": "user", "content": full_question})
        messages.append({"role": "assistant", "content": "<choice>"})
    elif model_type == "llama":
        if multi_inter_message is None:
            assert turn_number <= 0
            multi_inter_message = ""
        messages = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{preference}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{pref_generation}
<|eot_id|>
{multi_inter_message}
<|start_header_id|>user<|end_header_id|>
{full_question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    elif model_type == "mistral":
        if multi_inter_message is None:
            assert turn_number <= 0
            multi_inter_message = ""
        messages = f"""<s>[INST]
{system_prompt}
{preference}
[/INST]
{pref_generation}</s>
{multi_inter_message}
[INST]
{full_question}
[/INST]
<choice>
"""
    elif model_type == "gpt":
        user_message = {"role": "user", "content": preference}
        system_prompt = {"role": "system", "content": system_prompt}
        messages = [
            system_prompt,
            user_message,
            {"role": "assistant", "content": pref_generation},
        ]
        if multi_inter_message:
            assert turn_number > 0
            messages.extend(multi_inter_message)
        messages.append({"role": "user", "content": full_question})
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return messages


def get_question_prompt_mcq_rag(
    preference,
    options,
    pref_generation,
    question,
    multi_inter_message,
    model_type,
    top_k_sentences,
    turn_number,
    system_prompt="You are a helpful asisstant.",
):
    top_k_msg = convert_top_k_sentences_to_msg(top_k_sentences)
    question = top_k_msg + question
    mcq_question_format = get_mcq_question_format(options)
    if model_type == "claude":
        user_message = {"role": "user", "content": preference}
        messages = [user_message, {"role": "assistant", "content": pref_generation}]
        if multi_inter_message:
            assert turn_number > 0
            messages.extend(multi_inter_message)
        messages.append({"role": "user", "content": question + mcq_question_format})
        messages.append({"role": "assistant", "content": "<choice>"})
    elif model_type == "llama":
        if multi_inter_message is None:
            assert turn_number <= 0
            multi_inter_message = ""
        messages = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{preference}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{pref_generation}
<|eot_id|>
{multi_inter_message}
<|start_header_id|>user<|end_header_id|>
{question}{mcq_question_format}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    elif model_type == "mistral":
        if multi_inter_message is None:
            assert turn_number <= 0
            multi_inter_message = ""
        messages = f"""<s>[INST]
{system_prompt}
{preference}
[/INST]
{pref_generation}</s>
{multi_inter_message}
[INST]
{question}{mcq_question_format}
[/INST]
<choice>
"""
    elif model_type == "gpt":
        user_message = {"role": "user", "content": preference}
        system_prompt = {"role": "system", "content": system_prompt}
        messages = [
            system_prompt,
            user_message,
            {"role": "assistant", "content": pref_generation},
        ]
        if multi_inter_message:
            assert turn_number > 0
            messages.extend(multi_inter_message)
        messages.append({"role": "user", "content": question + mcq_question_format})
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return messages


def get_self_critic_prompt_critic_mcq(
    args,
    critic_request,
    preference,
    pref_generation,
    response_to_q,
    multi_inter_message,
    question,
    system_prompt,
    options,
):
    mcq_question_format = get_mcq_question_format(options)
    question = question + mcq_question_format
    if "claude" in args.model:
        critic_messages = [
            {"role": "user", "content": preference},
            {"role": "assistant", "content": pref_generation},
        ]
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
        critic_messages = f"""<|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {system_prompt}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {preference}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        {pref_generation}
        <|eot_id|>{multi_inter_message}
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
        critic_messages = f"""<s>[INST]
{system_prompt}
{preference}
[/INST]
{pref_generation}</s>
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


def get_self_critic_prompt_response_mcq(
    args,
    critic_request,
    preference,
    pref_generation,
    response_to_q,
    multi_inter_message,
    question,
    critic,
    revision_request,
    options,
    system_prompt,
):
    mcq_question_format = get_mcq_question_format(options)
    question = question + mcq_question_format
    revision_request += mcq_question_format
    if "claude" in args.model:
        critic_messages = [
            {"role": "user", "content": preference},
            {"role": "assistant", "content": pref_generation},
        ]
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
        critic_messages = f"""<|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {system_prompt}
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
        critic_messages = f"""<s>[INST]
{system_prompt}
{preference}
[/INST]
{pref_generation}</s>
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


def get_implicit_question_prompt_mcq(
    conversation_messages,
    question,
    multi_inter_message,
    model_type,
    turn_number,
    remind,
    cot,
    options,
):
    mcq_question_format = get_mcq_question_format(options)
    if remind:
        question = question + REMINDER
    if cot:
        question = COT_PROMPT + question
    question = question + mcq_question_format
    if model_type == "claude":
        messages = conversation_messages
        if turn_number > 0:
            messages.extend(multi_inter_message)
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": "<choice>"})
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
        {question}
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
{question}
[/INST]
<choice>
"""
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return messages
