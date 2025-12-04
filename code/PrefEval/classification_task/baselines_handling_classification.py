import sys, os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils_mcq import (
    generate_message,
    get_question_prompt_mcq,
    get_self_critic_prompt_critic_mcq,
    get_self_critic_prompt_response_mcq,
    get_implicit_question_prompt_mcq,
    get_question_prompt_mcq_rag,
)
from utils.mcq_implicit_utils import (
    get_implicit_question_prompt_rag_mcq,
    get_self_critic_prompt_response_mcq_implicit,
    get_self_critic_prompt_critic_mcq_implicit,
)
from utils.common_utils import generate_message


def load_turns_data(args):
    with open(f"{args.dir}/benchmark_dataset/filtered_inter_turns.json", "r") as infile:
        return json.load(infile)


def load_rag_data(args):
    with open(
        f"{args.dir}/benchmark_dataset/rag_retrieval/simcse_explicit_pref/{args.topic}_overall300_topk_history.json",
        "r",
    ) as infile:
        return json.load(infile)


def load_msg_index(args):
    with open(
        f"{args.dir}/benchmark_dataset/rag_retrieval/simcse_explicit_pref/msg_index_{args.topic}_overall300_topk_history.json",
        "r",
    ) as infile:
        return json.load(infile)


def update_task_data(
    task, pref_generation, end_generation, zero_shot_response=None, revised_messages=None, critic=None
):
    task["response_to_pref"] = pref_generation

    if zero_shot_response is not None:
        task["zero_shot_response_to_q"] = zero_shot_response
        task["response_to_q"] = revised_messages
        task["self_critic"] = critic
    else:
        task["response_to_q"] = end_generation


def save_results(save_file, topic_data):
    with open(save_file, "w") as outfile:
        json.dump(topic_data, outfile, indent=4)


def handle_client_error(err):
    message = err.response["Error"]["Message"]
    print("A client error occurred: " + format(message))


def handle_rag_task_mcq(
    args,
    preference,
    pref_generation,
    question,
    multi_inter_message,
    model_type,
    max_tokens,
    system_prompt,
    rag_data,
    msg_idx,
    task_id,
    new_options,
):

    sentence_scores = rag_data[task_id]["sentence_scores"][
        : 2 + args.inter_turns * 2
    ]  # extract offline stored sentence distance scores, 2 (stating preference, preference response) + 2 * inter turns
    sorted_sentence_scores = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    top_k_sentences = []
    for idx, score in sorted_sentence_scores[: args.topk]:
        # Note: if idx == 0 or idx == 1: RAG has extracted useful preference conversations.
        if idx == 0:
            top_k_sentences.append("User: " + preference)
        elif idx == 1:
            top_k_sentences.append("Assistant: " + pref_generation)
        else:
            top_k_sentences.append(msg_idx[str(idx)])
    assert len(top_k_sentences) == args.topk
    return get_question_prompt_mcq_rag(
        preference,
        new_options,
        pref_generation,
        question,
        multi_inter_message,
        model_type,
        top_k_sentences=top_k_sentences,
        turn_number=args.inter_turns,
        system_prompt=system_prompt,
    )


def handle_rag_task_implicit_mcq(
    args,
    conversation,
    question,
    multi_inter_message,
    model_type,
    max_tokens,
    system_prompt,
    rag_data,
    msg_rag_data,
    task_id,
    conversation_list,
    multi_turn_message,
    new_options,
):

    pref_sentence_scores = rag_data[task_id]["sentence_scores"]
    message_sentence_scores = msg_rag_data[task_id]["inter_sentence_scores"][: args.inter_turns * 2]
    combined_scores = []
    for i, score in enumerate(pref_sentence_scores):
        combined_scores.append(("pref sentence", i, score[1]))
    for i, score in enumerate(message_sentence_scores):
        combined_scores.append(("message sentence", i, score[1]))

    sorted_scores = sorted(combined_scores, key=lambda x: x[2], reverse=True)
    top_k_sentences = []

    for source, idx, score in sorted_scores[: args.topk]:
        if source == "pref sentence":
            top_k_sentences.append(conversation_list[idx]["role"] + ": " + conversation_list[idx]["content"])
        elif source == "message sentence":
            top_k_sentences.append(multi_turn_message[idx]["role"] + ": " + multi_turn_message[idx]["content"])

    assert len(top_k_sentences) == args.topk
    return get_implicit_question_prompt_rag_mcq(
        conversation,
        question,
        multi_inter_message,
        model_type,
        turn_number=args.inter_turns,
        system_prompt=system_prompt,
        top_k_sentences=top_k_sentences,
        options=new_options,
    )


def handle_selfcritic_task_mcq(
    args,
    preference,
    pref_generation,
    multi_inter_message,
    question,
    system_prompt,
    client,
    model_id,
    model_type,
    max_tokens,
    new_options,
):
    """create pref + multi turn conversations + question (get ZERO-SHOT answer)"""
    zero_shot_messages = get_question_prompt_mcq(
        preference,
        new_options,
        pref_generation,
        question,
        multi_inter_message,
        model_type,
        turn_number=args.inter_turns,
        remind=False,
        cot=False,
        system_prompt=system_prompt,
    )
    first_choice = generate_message(client, args.model, model_type, system_prompt, zero_shot_messages, max_tokens, device=args.device)
    if "mistral" in args.model or "claude" in args.model:
        first_choice = "<choice>" + first_choice
    critic_request = """Critique Request: Review your previous response to the user's question in the last conversation turn. Check if your previous choice adheres or violates to any user preferences stated earlier in the conversation that is related to this query. Provide a critique on how well those preferences were followed. Give a critic of your choice in 2 sentences. Answer in this format:
    Critic: """
    revision_request = f"""Revision Request: Based on your critic, please choose an option again to align more closely with the user's earlier stated preferences. Note that you can answer the same option if you think your previous answer is correct."""

    critic_messages = get_self_critic_prompt_critic_mcq(
        args,
        critic_request,
        preference,
        pref_generation,
        first_choice,
        multi_inter_message,
        question,
        system_prompt=system_prompt,
        options=new_options,
    )
    critic = generate_message(client, args.model, model_type, system_prompt, critic_messages, 300, device=args.device)

    """Regenerate Response"""
    revision_message = get_self_critic_prompt_response_mcq(
        args,
        critic_request,
        preference,
        pref_generation,
        first_choice,
        multi_inter_message,
        question,
        critic,
        revision_request,
        new_options,
        system_prompt=system_prompt,
    )
    end_generation = generate_message(
        client,
        args.model,
        model_type,
        system_prompt,
        revision_message,
        max_tokens,
        device=args.device,
    )

    return first_choice, end_generation, critic


def handle_selfcritic_task_implicit_mcq(
    args,
    conversation,
    multi_inter_message,
    question,
    system_prompt,
    client,
    model_id,
    model_type,
    max_tokens,
    new_options,
):
    """create pref + multi turn conversations + question (get ZERO-SHOT answer)"""
    messages = get_implicit_question_prompt_mcq(
        conversation_messages=conversation.copy(),
        question=question,
        multi_inter_message=multi_inter_message,
        model_type=model_type,
        turn_number=args.inter_turns,
        remind=False,
        cot=False,
        options=new_options,
    )
    first_choice = generate_message(client, args.model, model_type, system_prompt, messages, max_tokens, device=args.device)
    if "mistral" in args.model or "claude" in args.model:
        first_choice = "<choice>" + first_choice

    critic_request = """Critique Request: Review your previous response to the user's question in the last conversation turn. Check if your previous choice adheres or violates to any user preferences stated earlier in the conversation that is related to this query. Provide a critique on how well those preferences were followed. Give a critic of your choice in 2 sentences. Answer in this format:\nCritic: """
    revision_request = f"""Revision Request: Based on your critic, please choose an option again to align more closely with the user's earlier stated preferences. Note that you can answer the same option if you think your previous answer is correct."""

    critic_messages = get_self_critic_prompt_critic_mcq_implicit(
        args,
        conversation_messages=conversation.copy(),
        critic_request=critic_request,
        response_to_q=first_choice,
        multi_inter_message=multi_inter_message,
        question=question,
        options=new_options,
    )

    critic = generate_message(client, args.model, model_type, system_prompt, critic_messages, 300, device=args.device)
    revision_message = get_self_critic_prompt_response_mcq_implicit(
        args,
        new_options,
        model_type,
        conversation_messages=conversation.copy(),
        critic_request=critic_request,
        response_to_q=first_choice,
        multi_inter_message=multi_inter_message,
        question=question,
        critic=critic,
        revision_request=revision_request,
    )

    end_generation = generate_message(
        client,
        args.model,
        model_type,
        system_prompt,
        revision_message,
        max_tokens,
        device=args.device,
    )

    return first_choice, end_generation, critic
