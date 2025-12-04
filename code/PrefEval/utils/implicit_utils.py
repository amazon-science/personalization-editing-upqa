import os
import json
import time
from .common_utils import COT_PROMPT, REMINDER
from .chat_template_utils import (
    format_conversation_for_model,
    create_messages_with_system,
    create_messages_without_system,
    get_tokenizer_for_model_type,
    apply_chat_template_if_available
)


def extract_implicit_messages(args, conversation, model_type):
    if args.pref_type == "choice":
        conversation_messages, _ = extract_conversation_to_messages(conversation, model_type)
        conversation_list, _ = extract_conversation_to_messages(conversation, "claude")
    elif args.pref_type == "persona":
        conversation_messages = extract_conversation_to_msg_persona(conversation, model_type)
        conversation_list = []
        for key, item in conversation.items():
            conversation_list.append({"role": "user", "content": item["user"]})
            conversation_list.append({"role": "assistant", "content": item["assistant"]})
    return conversation_messages, conversation_list


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
    if args.topic == "prefeval_pro_balanced":
        # Special case for prefeval_pro_balanced topic
        with open(
            "/home/personalization-editing/data/prefeval_pro/prefeval_pro_balanced.json",
            "r",
        ) as infile:
            topic_data = json.load(infile)
    elif args.pref_type == "choice":
        with open(
            f"{args.dir}/benchmark_dataset/implicit_preference/choice-based/{args.topic}.json",
            "r",
        ) as infile:
            topic_data = json.load(infile)
    elif args.pref_type == "persona":
        with open(
            f"{args.dir}/benchmark_dataset/implicit_preference/persona-driven/{args.topic}.json",
            "r",
        ) as infile:
            topic_data = json.load(infile)

    dir_path = f"{args.dir}/benchmark_results/implicit/{args.pref_type}/generation_results/{args.task}/{args.topic}/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    save_file = f"{args.dir}/benchmark_results/implicit/{args.pref_type}/generation_results/{args.task}/{args.topic}/{args.model}_{args.topic}_{args.inter_turns}interturn.json"
    return topic_data, save_file


def load_files_implicit_rag(args):
    if args.topic == "prefeval_pro_balanced":
        # Special case for prefeval_pro_balanced topic
        with open(
            "/home/personalization-editing/data/prefeval_pro/prefeval_pro_balanced.json",
            "r",
        ) as infile:
            topic_data = json.load(infile)
    elif args.pref_type == "choice":
        with open(
            f"{args.dir}/benchmark_dataset/implicit_preference/choice-based/{args.topic}.json",
            "r",
        ) as infile:
            topic_data = json.load(infile)
    elif args.pref_type == "persona":
        with open(
            f"{args.dir}/benchmark_dataset/implicit_preference/persona-driven/{args.topic}.json",
            "r",
        ) as infile:
            topic_data = json.load(infile)
    dir_path = f"{args.dir}/benchmark_results/implicit/{args.pref_type}/generation_results/{args.task}/{args.topic}/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    save_file = f"{args.dir}/benchmark_results/implicit/{args.pref_type}/generation_results/{args.task}/{args.topic}/{args.model}_{args.topic}_{args.inter_turns}interturn.json"
    
    if args.topic == "prefeval_pro_balanced":
        # For prefeval_pro_balanced topic, RAG data is not available
        # Return empty lists as placeholders
        rag_data = []
        msg_rag_data = []
    else:
        if args.pref_type == "choice":
            with open(
                f"{args.dir}/benchmark_dataset/rag_retrieval/simcse_implicit_{args.pref_type}/{args.topic}_overall300_topk_history_mcq.json",
                "r",
            ) as infile:
                rag_data = json.load(infile)
        else:
            with open(
                f"{args.dir}/benchmark_dataset/rag_retrieval/simcse_implicit_{args.pref_type}/{args.topic}_overall300_topk_history_{args.pref_type}.json",
                "r",
            ) as infile:
                rag_data = json.load(infile)

        with open(
            f"{args.dir}/benchmark_dataset/rag_retrieval/simcse_question_inter_conversation_similarities/{args.topic}_300_inter_similarities.json",
            "r",
        ) as infile:
            msg_rag_data = json.load(infile)
    
    return topic_data, save_file, rag_data, msg_rag_data


def extract_conversation_to_messages(conversation, model_type, model_id=None):
    """
    Extract conversation to messages for different model types.
    Uses tokenizer.apply_chat_template() when available.
    """
    messages = []
    role_list = []

    # For new models, try to use tokenizer chat template first
    if model_type in ["gemma", "qwen", "olmo"]:
        tokenizer = get_tokenizer_for_model_type(model_type, model_id)
        
        # Build standard conversation messages
        standard_messages = []
        for key, content in conversation.items():
            if key == "query" or key == "user_selection":
                standard_messages.append({"role": "user", "content": content})
            else:
                standard_messages.append({"role": "assistant", "content": content})
        
        # Try tokenizer chat template
        template_result = apply_chat_template_if_available(
            tokenizer, standard_messages, add_generation_prompt=False, tokenize=False
        )
        if template_result is not None:
            return [template_result], role_list
        
        # Fallback to manual formatting
        return [format_conversation_for_model(standard_messages, model_type, tokenizer)], role_list

    # Original logic for existing models
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


def get_implicit_question_prompt(
    conversation_messages,
    question,
    multi_inter_message,
    model_type,
    turn_number,
    max_tokens,
    remind,
    cot,
    model_id=None,
):
    """
    Get implicit question prompt for different model types.
    Uses tokenizer.apply_chat_template() when available.
    """
    if remind:
        question = question + REMINDER
    if cot:
        question = COT_PROMPT + question
    question += f" (Please respond within {max_tokens} words.)"
    
    # For new models, try to use tokenizer chat template first
    if model_type in ["gemma", "qwen", "olmo"]:
        tokenizer = get_tokenizer_for_model_type(model_type, model_id)
        system_prompt = "You are a helpful AI assistant."
        
        # Convert conversation_messages to standard format if needed
        if isinstance(conversation_messages, list) and len(conversation_messages) > 0:
            if isinstance(conversation_messages[0], dict):
                # Already in standard format
                messages = conversation_messages.copy()
            else:
                # Convert string format to standard format (simplified)
                messages = [{"role": "assistant", "content": "".join(conversation_messages)}]
        else:
            messages = []
        
        if turn_number > 0 and multi_inter_message:
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
        messages = conversation_messages
        if turn_number > 0:
            messages.extend(multi_inter_message)
        messages.append({"role": "user", "content": question})
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
        messages = f"""<s>[INST]
You are a helpful AI assistant.
{conversation_messages[6:]}
{multi_inter_message}
[INST]
{question}
[/INST]"""
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return messages


def get_self_critic_prompt_critic_implicit(
    args,
    conversation_messages,
    critic_request,
    response_to_q,
    multi_inter_message,
    question,
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


def get_self_critic_prompt_response_implicit(
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


def convert_top_k_sentences_to_msg(top_k_sentences):
    msg = (
        "Before answering my question, please consider the following context from our previous conversations. "
        f"These are the {len(top_k_sentences)} most relevant exchanges that we had previously, which may contain information about "
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
    max_tokens,
    top_k_sentences,
    system_prompt="You are a helpful asisstant.",
    model_id=None,
):
    """
    Get implicit question prompt with RAG context for different model types.
    Uses tokenizer.apply_chat_template() when available.
    """
    top_k_msg = convert_top_k_sentences_to_msg(top_k_sentences)
    question = top_k_msg + question
    question += f" (Please respond within {max_tokens} words.)"
    
    # For new models, try to use tokenizer chat template first
    if model_type in ["gemma", "qwen", "olmo"]:
        tokenizer = get_tokenizer_for_model_type(model_type, model_id)
        
        # Convert conversation_messages to standard format if needed
        if isinstance(conversation_messages, list) and len(conversation_messages) > 0:
            if isinstance(conversation_messages[0], dict):
                # Already in standard format
                messages = conversation_messages.copy()
            else:
                # Convert string format to standard format (simplified)
                messages = [{"role": "assistant", "content": "".join(conversation_messages)}]
        else:
            messages = []
        
        if multi_inter_message:
            assert turn_number > 0
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
