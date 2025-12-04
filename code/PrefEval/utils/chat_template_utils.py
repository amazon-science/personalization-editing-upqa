"""
Utility functions for handling chat templates across different model types.
Uses tokenizer.apply_chat_template() when available, with fallbacks for older models.
"""

from transformers import AutoTokenizer
from typing import List, Dict, Any, Union, Optional


def get_tokenizer_for_model_type(model_type: str, model_id: str = None) -> Optional[AutoTokenizer]:
    """
    Get a tokenizer instance for the given model type.
    Used for applying chat templates when a tokenizer instance is available.
    """
    if model_id:
        try:
            return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Could not load tokenizer for {model_id}: {e}")
            return None
    return None


def apply_chat_template_if_available(
    tokenizer: Optional[AutoTokenizer],
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
    tokenize: bool = False
) -> Union[str, List[int], None]:
    """
    Apply chat template using tokenizer if available and has chat_template.
    Returns None if tokenizer is not available or doesn't have chat_template.
    """
    if (tokenizer is not None and 
        hasattr(tokenizer, 'apply_chat_template') and 
        tokenizer.chat_template is not None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize
            )
        except Exception as e:
            print(f"Warning: Failed to apply chat template: {e}")
    return None


def create_messages_with_system(
    system_prompt: str,
    user_content: str,
    assistant_content: str = None
) -> List[Dict[str, str]]:
    """Create standard chat messages format with system prompt."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    if assistant_content:
        messages.append({"role": "assistant", "content": assistant_content})
    return messages


def create_messages_without_system(
    user_content: str,
    assistant_content: str = None
) -> List[Dict[str, str]]:
    """Create standard chat messages format without system prompt."""
    messages = [{"role": "user", "content": user_content}]
    if assistant_content:
        messages.append({"role": "assistant", "content": assistant_content})
    return messages


def format_conversation_for_model(
    messages: List[Dict[str, str]],
    model_type: str,
    tokenizer: Optional[AutoTokenizer] = None,
    system_prompt: str = "You are a helpful assistant."
) -> Union[str, List[Dict], Dict]:
    """
    Format conversation messages for different model types.
    Uses tokenizer.apply_chat_template() when available, falls back to manual formatting.
    """
    
    # Try to use tokenizer chat template first for supported models
    if model_type in ["gemma", "qwen", "olmo"] and tokenizer is not None:
        template_result = apply_chat_template_if_available(
            tokenizer, messages, add_generation_prompt=True, tokenize=False
        )
        if template_result is not None:
            return template_result
    
    # Fallback to manual formatting for each model type
    if model_type == "claude":
        return messages
    
    elif model_type == "llama":
        formatted_messages = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted_messages += f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"
        formatted_messages += "<|start_header_id|>assistant<|end_header_id|>"
        return formatted_messages
    
    elif model_type == "mistral":
        # For Mistral, we need to handle system prompt differently
        user_msgs = [msg for msg in messages if msg["role"] == "user"]
        assistant_msgs = [msg for msg in messages if msg["role"] == "assistant"]
        
        formatted = f"<s>[INST]{system_prompt}\n"
        for i, user_msg in enumerate(user_msgs):
            formatted += user_msg["content"]
            if i < len(user_msgs) - 1:
                formatted += "[/INST]"
                if i < len(assistant_msgs):
                    formatted += assistant_msgs[i]["content"] + "</s>[INST]"
        formatted += "[/INST]"
        return formatted
    
    elif model_type in ["gpt", "qwen"]:
        formatted_messages = [{"role": "system", "content": system_prompt}]
        formatted_messages.extend(messages)
        return formatted_messages
    
    elif model_type == "gemini":
        return {
            "system_instruction": system_prompt,
            "messages": [
                {"role": "user" if msg["role"] == "user" else "model", 
                 "parts": [{"text": msg["content"]}]}
                for msg in messages
            ]
        }
    
    elif model_type in ["gemma", "olmo"]:
        # Fallback manual formatting for these models if tokenizer template fails
        formatted_messages = f"System: {system_prompt}\n"
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted_messages += f"{role}: {content}\n"
        formatted_messages += "Assistant: "
        return formatted_messages
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def extend_conversation_with_multi_turn(
    base_messages: Union[str, List[Dict], Dict],
    multi_inter_message: Union[str, List[Dict]],
    model_type: str
) -> Union[str, List[Dict], Dict]:
    """
    Extend base conversation with multi-turn messages based on model type.
    """
    if not multi_inter_message:
        return base_messages
    
    if model_type == "claude":
        if isinstance(base_messages, list) and isinstance(multi_inter_message, list):
            return base_messages + multi_inter_message
    
    elif model_type in ["llama", "mistral", "qwen"]:
        if isinstance(base_messages, str) and isinstance(multi_inter_message, str):
            return base_messages + multi_inter_message
    
    elif model_type == "gpt":
        if isinstance(base_messages, list) and isinstance(multi_inter_message, list):
            return base_messages + multi_inter_message
    
    elif model_type == "gemini":
        if isinstance(base_messages, dict) and isinstance(multi_inter_message, list):
            base_messages["messages"].extend(multi_inter_message)
            return base_messages
    
    elif model_type in ["gemma", "olmo"]:
        if isinstance(base_messages, str) and isinstance(multi_inter_message, str):
            return base_messages + multi_inter_message
        elif isinstance(base_messages, list) and isinstance(multi_inter_message, list):
            return base_messages + multi_inter_message
    
    return base_messages
