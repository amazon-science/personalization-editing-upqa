import sys, os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.explicit_utils import (
    get_question_prompt_rag,
    get_question_prompt,
    get_self_critic_prompt_critic,
    get_self_critic_prompt_response,
)
from utils.implicit_utils import (
    get_self_critic_prompt_critic_implicit,
    get_self_critic_prompt_response_implicit,
    get_implicit_question_prompt,
    get_implicit_question_prompt_rag,
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


def handle_rag_task(
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
    return get_question_prompt_rag(
        preference,
        pref_generation=pref_generation,
        question=question,
        multi_inter_message=multi_inter_message,
        model_type=model_type,
        max_tokens=max_tokens,
        turn_number=args.inter_turns,
        system_prompt=system_prompt,
        top_k_sentences=top_k_sentences,
    )


def handle_rag_task_implicit(
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
    return get_implicit_question_prompt_rag(
        conversation,
        question,
        multi_inter_message,
        model_type,
        max_tokens=max_tokens,
        turn_number=args.inter_turns,
        system_prompt=system_prompt,
        top_k_sentences=top_k_sentences,
    )


def handle_selfcritic_task(
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
):
    """create pref + multi turn conversations + question (get ZERO-SHOT answer)"""
    zero_shot_messages = get_question_prompt(
        preference=preference,
        pref_generation=pref_generation,
        question=question,
        multi_inter_message=multi_inter_message,
        model_type=model_type,
        max_tokens=max_tokens,
        turn_number=args.inter_turns,
        remind=False,
        cot=False,
        system_prompt=system_prompt,
        args=args,
    )
    response_to_q = generate_message(client, args.model, model_type, system_prompt, zero_shot_messages, max_tokens, device=args.device)
    critic_request = "Critique Request: Review your previous response to the user's question in the last conversation turn. Check if the response adheres or violates to any user preferences stated earlier in the conversation that is related to this query. Provide a critique on how well those preferences were followed. Give a critic of your response in 2 sentences. Answer in this format:\nCritic: "

    critic_messages = get_self_critic_prompt_critic(
        args, critic_request, preference, pref_generation, response_to_q, multi_inter_message, question, system_prompt
    )
    critic = generate_message(client, args.model, model_type, system_prompt, critic_messages, max_tokens, device=args.device)

    revision_request = f"Revision Request: Based on your critic, please rewrite your previous response to align more closely with the user's earlier stated preferences. Answer the question again: {question}. Answer in this format (just give updated response, don't include critic or any explanation):\nResponse: "

    revision_prompt = get_self_critic_prompt_response(
        args=args,
        critic_request=critic_request,
        preference=preference,
        pref_generation=pref_generation,
        response_to_q=response_to_q,
        multi_inter_message=multi_inter_message,
        question=question,
        critic=critic,
        revision_request=revision_request,
        system_prompt=system_prompt,
    )
    final_response = generate_message(
        client,
        args.model,
        model_type,
        system_prompt=system_prompt,
        messages=revision_prompt,
        max_tokens=500,
        device=args.device,
    )

    return response_to_q, final_response, critic


def handle_selfcritic_task_implicit(
    args,
    conversation,
    multi_inter_message,
    question,
    system_prompt,
    client,
    model_id,
    model_type,
    max_tokens,
):
    """create pref + multi turn conversations + question (get ZERO-SHOT answer)"""
    zero_shot_messages = get_implicit_question_prompt(
        conversation_messages=conversation.copy(),
        question=question,
        multi_inter_message=multi_inter_message,
        model_type=model_type,
        max_tokens=max_tokens,
        turn_number=args.inter_turns,
        remind=False,
        cot=False,
    )
    response_to_q = generate_message(client, args.model, model_type, system_prompt, zero_shot_messages, max_tokens, device=args.device)
    critic_request = "Critique Request: Review your previous response to the user's question in the last conversation turn. Check if the response adheres or violates to any user preferences stated earlier in the conversation that is related to this query. Provide a critique on how well those preferences were followed. Give a critic of your response in 2 sentences. Answer in this format:\nCritic: "

    critic_messages = get_self_critic_prompt_critic_implicit(
        args,
        conversation_messages=conversation.copy(),
        critic_request=critic_request,
        response_to_q=response_to_q,
        multi_inter_message=multi_inter_message,
        question=question,
    )
    critic = generate_message(client, args.model, model_type, system_prompt, critic_messages, max_tokens, device=args.device)
    revision_request = f"Revision Request: Based on your critic, please rewrite your previous response to align more closely with the user's earlier stated preferences. Answer the question again: {question}. Answer in this format (just give updated response, don't include critic or any explanation):\nResponse: "

    revision_message = get_self_critic_prompt_response_implicit(
        args,
        conversation.copy(),
        critic_request,
        response_to_q,
        multi_inter_message,
        question,
        critic,
        revision_request,
    )
    self_critic_response = generate_message(
        client,
        args.model,
        model_type,
        system_prompt,
        revision_message,
        max_tokens=500,
        device=args.device,
    )

    return response_to_q, self_critic_response, critic
