"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""
from ..models.melo.melo import LORA
# from ..util_overall import system_msg_product_recommendation, system_msg_eval_product_recommendation, user_msg_eval_format_product_recommendation
import typing
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .evaluate_utils import (
    test_seq2seq_batch_prediction_acc, 
    test_batch_prediction_acc, 
    test_prediction_acc,
    test_prediction_acc_LLM_judge_easyedit,
    test_prediction_acc_llm_judge,
    test_generation_quality, 
    test_concept_gen,
    test_safety_gen,
    test_instance_change,
    PPL,
    OOD_PPL,
    kl_loc_loss,
    es,
    es_per_icl,
    per_generation,
    F1
)

system_msg_qa = "Always respond to the input question concisely with a short phrase or a single-word answer. Do not repeat the question or provide any explanation."

user_msg_eval_format_product_recommendation = "Text: {target_new} \nProduct: {decoded_output}"
system_msg_product_recommendation = system_msg_qa
system_msg_eval_product_recommendation = """You are an expert evaluator. Given a Text and a Product, determine if the Product is relevant.

Rules:
- Output '1' if the Product is useful for, related to, or commonly used in the Text activity
- Output '1' if someone with this text feature would likely need or want this product
- Output '0' only if the Product is completely unrelated to the Text
- Do not provide any explanation, just output '1' or '0'"""   


def evaluate_matching_behavior_probability(questions, targets, model, tok_eval, a_token_id, b_token_id):
    """
    Compute the probability of answer_matching_behavior for each question.
    This follows the exact same approach as the CAA paper including proper chat formatting,
    system prompts, and the same probability normalization method.
    
    Args:
        questions: List of questions from eval_data
        targets: List of answer_matching_behavior values (e.g., "(A)" or "(B)")
        model: Model for evaluation
        tok_eval: Tokenizer for evaluation
        a_token_id: Token ID for "A"
        b_token_id: Token ID for "B"
        
    Returns:
        List of probabilities for answer_matching_behavior
    """
    if isinstance(questions, str):
        questions = [questions]
    if isinstance(targets, str):
        targets = [targets]

    prob_list = []
    # Set model to eval mode - access the underlying PyTorch model
    if hasattr(model, 'model'):
        model.model.eval()
    elif hasattr(model, 'eval'):
        model.eval()
    
    with torch.no_grad():
        for i, (question, target) in enumerate(zip(questions, targets)):
            # Compute A/B probabilities using the same method as CAA
            prob_score = compute_ab_choice_probability(question, target, model, tok_eval, a_token_id, b_token_id)
            prob_list.append(prob_score)

    return prob_list


def compute_ab_choice_probability(question, answer_matching_behavior, model, tok_eval, a_token_id, b_token_id):
    """
    Compute the probability of choosing the answer_matching_behavior option.
    This follows the exact same approach as the CAA code using proper chat formatting.
    
    Args:
        question: The question text
        answer_matching_behavior: The target answer (e.g., "(A)" or "(B)")
        model: Model for evaluation
        tok_eval: Tokenizer for evaluation
        a_token_id: Token ID for "A"
        b_token_id: Token ID for "B"
        
    Returns:
        Probability of choosing the answer_matching_behavior option
    """
    # Use the same system prompt as CAA for hallucination behavior
    # system_prompt = "You hallucinate a lot and assume that many things are possible and true, even if they are not."
    system_prompt = ""
    
    # Format the prompt exactly like CAA using chat format
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    
    # Create the full prompt with system message and instruction formatting
    input_content = B_SYS + system_prompt + E_SYS
    input_content += f"{B_INST} {question.strip()} {E_INST} ("
    
    # Tokenize the prompt
    inputs = tok_eval(input_content, return_tensors="pt", truncation=True, max_length=1024)
    
    # Access the underlying PyTorch model and its device
    if hasattr(model, 'model'):
        pytorch_model = model.model
        device = model.device if hasattr(model, 'device') else pytorch_model.device
    else:
        pytorch_model = model
        device = model.device
    
    input_ids = inputs['input_ids'].to(device)
    
    # Get model logits - this mimics the get_logits_from_text approach in CAA
    with torch.no_grad():
        outputs = pytorch_model(input_ids)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs[0]  # For some model outputs, logits are in the first element
    
    # Use the exact same approach as get_a_b_probs from CAA
    last_token_logits = logits[0, -1, :]
    last_token_probs = torch.softmax(last_token_logits, dim=-1)
    a_prob = last_token_probs[a_token_id].item()
    b_prob = last_token_probs[b_token_id].item()
    
    # Normalize probabilities exactly like get_avg_key_prob in CAA
    denom = a_prob + b_prob
    if denom == 0:
        return 0.5  # Default to 0.5 if both probabilities are 0
    
    # Return probability of the matching behavior (same logic as get_avg_key_prob)
    if "A" in answer_matching_behavior:
        result = a_prob / denom
        return result
    elif "B" in answer_matching_behavior:
        result = b_prob / denom
        return result
    else:
        # If format is unexpected, default to 0.5
        return 0.5


def compute_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_generation = False,
    icl_pre_edit = True,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :return: Dictionary containing rewriting metrics
    """
    if isinstance(model, LORA):
        model=model.model

    target_new = record["target_new"]
    rewrite_prompts = record["prompt"]
    user_preference = record["user_preference"] if 'user_preference' in record.keys() else None
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None

    if hparams.alg_name in ['ICE', 'IKE'] and icl_pre_edit == False:
        # icl_prompt = f"New Fact: Q: {edit_prompts} A: {target_new}\n"
        # icl_prompt = f'{rewrite_prompts.replace("Your answer:", "Correct answer:")} {target_new}\nPrompt: '
        icl_prompt = f'User preference: {user_preference}\nAnswer the following question based on the user preference. Do not repeat the question or provide any explanation.\n'
        if user_preference is None:
            icl_prompt = f'Answer the following question by repeating the following correct answer: {target_new}. Do not repeat the question.\n'
    else:
        icl_prompt = ""
    # yes_question = record['yes_question']['prompt'] if 'yes_question' in record.keys() and any(record['yes_question']) else None
    # no_question = record['no_question']['prompt'] if 'no_question' in record.keys() and any(record['no_question']) else None
    
    ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                              icl_prompt+rewrite_prompts, target_new, device=device, eval_metric=eval_metric)
    
    if not icl_pre_edit:
        ret[f"ICE_post_edit_prompt"] = icl_prompt+rewrite_prompts

    if rephrase_prompts is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                                icl_prompt+rephrase_prompts, target_new, device=device, test_rephrase=True, eval_metric=eval_metric)
        )

    if 'locality' in record.keys() and any(record['locality']):
        ret['locality'] = {}
        for locality_key in record['locality'].keys():
            ret['locality'].update(
                compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                         record['locality'][locality_key]['prompt'],
                                         record['locality'][locality_key]['ground_truth'], device=device)
            )

    if 'portability' in record.keys() and any(record['portability']):
        ret['portability'] = {}
        for portability_key in record['portability'].keys():
            ret['portability'].update(
                compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            record['portability'][portability_key]['prompt'],
                                            record['portability'][portability_key]['ground_truth'], device=device)
            )

    if 'product_recommendation' in record.keys() and any(record['product_recommendation']):
        ret['product_recommendation'] = {}
        for key in record['product_recommendation'].keys():
            product_recommendation_qa = record['product_recommendation'][key]['prompt']
            if isinstance(product_recommendation_qa, list):
                product_recommendation_qa = [e+icl_prompt for e in product_recommendation_qa]
            else:
                product_recommendation_qa = icl_prompt + product_recommendation_qa
            # ret['product_recommendation'].update(compute_general_quality_multiple_labels(model, hparams, tok, key, record['product_recommendation'][key]['prompt'], record['product_recommendation'][key]['ground_truth'], device))
            ret['product_recommendation'].update(compute_general_quality(model, hparams, tok, key, record['product_recommendation'][key]['prompt'], record['product_recommendation'][key]['ground_truth'], 
                                                                         device, system_msg=system_msg_product_recommendation, system_msg_eval=system_msg_eval_product_recommendation, user_msg_eval_format=user_msg_eval_format_product_recommendation))

    if 'implicit' in record.keys() and any(record['implicit']):
        ret['implicit'] = {}
        for key in record['implicit'].keys():
            implicit_qa = record['implicit'][key]['prompt']
            if isinstance(implicit_qa, list):
                implicit_qa = [e+icl_prompt for e in implicit_qa]
            else:
                implicit_qa = icl_prompt + implicit_qa
            ret['implicit'].update(compute_general_quality(
                model, hparams, tok, key, record['implicit'][key]['prompt'], record['implicit'][key]['ground_truth'], device, None, None, None))

    if test_generation:
        if hparams.alg_name == 'GRACE':
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100, vanilla_generation=True)
        else:
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100, vanilla_generation=False)
    return ret

def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    device,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em',
) -> typing.Dict:
    
    if not test_rephrase:
        key = 'rewrite'
    else:
        key = 'rephrase'
    # using real-world evaluation: autoregressive decoding, natural stop criteria, LLM-as-a-Judge
    if hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "LLM-judge":
        acc, gen_content = test_prediction_acc_LLM_judge_easyedit(model, tok, hparams, prompt, target_new, device, locality=False)
        ret = {
            f"{key}_acc": acc,
            f"{key}_gen_content": gen_content
        }
    elif hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "local-llm-judge":
        # Use rephrase_additional_token_len for rephrase evaluation if available
        additional_token_len = 0
        if test_rephrase and hasattr(hparams, 'rephrase_additional_token_len'):
            additional_token_len = hparams.rephrase_additional_token_len
        acc, response = test_prediction_acc(model, tok, hparams, prompt, target_new, device, locality=False, evaluation_type="local-llm-judge", additional_token_len=additional_token_len)
        ret = {
            f"{key}_acc": acc, 
            f"{key}_response": response
        }
    elif hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "claude-judge":
        additional_token_len = hparams.rephrase_additional_token_len if test_rephrase and hasattr(hparams, 'rephrase_additional_token_len') else 0
        acc, response = test_prediction_acc(model, tok, hparams, prompt, target_new, device, locality=False, evaluation_type="claude-judge", additional_token_len=additional_token_len)
        ret = {f"{key}_acc": acc, f"{key}_response": response}
    elif hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "generate-text":
        gen_content_model = test_prediction_acc_LLM_judge_easyedit(model, tok, hparams, prompt, target_new, device, locality=False)
        ret = {
            f"{key}_gen_content": gen_content_model
        }
    else:  # traditional evaluation 
        if eval_metric == 'ppl':
            ppl = PPL(model, tok, prompt, target_new, device)
            ret = {
                f"{key}_ppl": ppl
            }
        elif eval_metric == 'ood_ppl':
            ans = OOD_PPL(model, tok, prompt, target_new, device)
            ret = {
                f"ood_acc": ans
            }
        elif hparams.alg_name=="GRACE":
            # ppl = PPL(model, tok, prompt, target_new, device)
            if 't5' in model_name.lower():
                acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target_new, device)
            else:
                additional_token_len = hparams.rephrase_additional_token_len if test_rephrase and hasattr(hparams, 'rephrase_additional_token_len') else 0
                acc, responses = test_prediction_acc(model, tok, hparams, prompt, target_new, device, vanilla_generation=True, additional_token_len=additional_token_len)
            f1 = F1(model,tok,hparams,prompt,target_new,device, vanilla_generation=True)
            ret = {
                f"{key}_acc": acc,
                # f"{key}_PPL": ppl,
                f"{key}_response": responses,
                f"{key}_F1":f1     
            }        
        else:  # teacher-forcing evaluation
            if 't5' in model_name.lower():
                acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target_new, device)
            else:
                additional_token_len = hparams.rephrase_additional_token_len if test_rephrase and hasattr(hparams, 'rephrase_additional_token_len') else 0
                acc, response = test_prediction_acc(model, tok, hparams, prompt, target_new, device, additional_token_len=additional_token_len)
            ret = {
                f"{key}_acc": acc,
                f"{key}_response": response,
            }
    
    # Add token probability evaluation if enabled
    # This evaluation is independent of other metrics
    if hasattr(hparams, 'use_token_prob') and hparams.use_token_prob:
        # Get token IDs for A and B (same as CAA approach)
        a_token_id = tok.convert_tokens_to_ids("A")
        b_token_id = tok.convert_tokens_to_ids("B")
        # Compute token probability using CAA methodology
        # Note: When use_token_prob is True, prompt and target_new should be in A/B choice format
        prob_score = evaluate_matching_behavior_probability([prompt], [target_new], model, tok, a_token_id, b_token_id)
        ret[f"{key}_token_prob"] = prob_score[0] if prob_score else 0.5
        
    return ret

def compute_general_quality(
    model,
    hparams: HyperParams,
    tok: AutoTokenizer,
    question_key: str,
    prompt: typing.Union[str, List[str]],
    target_new: Optional[typing.Union[str, List[str]]],
    device,
    system_msg: Optional[str],
    system_msg_eval: Optional[str],
    user_msg_eval_format: Optional[str]
) -> typing.Dict:
    additional_token_len = hparams.implicit_additional_token_len if hasattr(hparams, 'implicit_additional_token_len') else 0
    additional_token_len = hparams.product_additional_token_len if hasattr(hparams, 'product_additional_token_len') else 0
    if hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "local-llm-judge":
        acc, response = test_prediction_acc(model, tok, hparams, prompt, target_new, device, system_msg=system_msg, system_msg_eval=system_msg_eval, evaluation_type="local-llm-judge", user_msg_eval_format=user_msg_eval_format, additional_token_len=additional_token_len)
    elif hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "claude-judge":
        acc, response = test_prediction_acc(model, tok, hparams, prompt, target_new, device, system_msg=system_msg, system_msg_eval=system_msg_eval, evaluation_type="claude-judge", user_msg_eval_format=user_msg_eval_format, additional_token_len=additional_token_len)
    else:
        acc, response = test_prediction_acc(model, tok, hparams, prompt, target_new, device, system_msg=system_msg, system_msg_eval=system_msg_eval, user_msg_eval_format=user_msg_eval_format, additional_token_len=additional_token_len)
    return {f"{question_key}_acc": acc, f"{question_key}_response": response}

def compute_general_quality_multiple_labels(
    model,
    hparams: HyperParams,
    tok: AutoTokenizer,
    question_key: str,
    prompt: typing.Union[str, List[str]],
    target_new_list: List[typing.Union[str, List[str]]],
    device,
    system_msg: Optional[str] = None
) -> typing.Dict:
    individual_accs = []
    all_responses = []
    for target_new in target_new_list:
        if hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "local-llm-judge":
            acc, response = test_prediction_acc(model, tok, hparams, prompt, target_new, device, system_msg=system_msg, evaluation_type="local-llm-judge", additional_token_len=4)
        else:
            acc, response = test_prediction_acc(model, tok, hparams, prompt, target_new, device, system_msg=system_msg, additional_token_len=4)
        
        individual_accs.append(acc)
        all_responses.append(response)
    
    # Final accuracy is 1 if any individual accuracy is 1 (OR logic)
    final_acc = 0
    correct_response = None
    for acc, response in zip(individual_accs, all_responses):
        if acc == 1:
            final_acc = 1
            correct_response = response
            break
    if final_acc == 0:
        correct_response = all_responses[0]  # Default to first response if none are correct
    
    return {
        f"{question_key}_acc": final_acc, 
        f"{question_key}_response": correct_response,
        # f"{question_key}_response": all_responses,
        # f"{question_key}_individual_accs": individual_accs
    }

def compute_locality_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: typing.Union[str, List[str]],
    locality_ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:

    # using real-world evaluation: autoregressive decoding, natural stop criteria, LLM-as-a-Judge
    if hasattr(hparams, 'evaluation_type'):
        loc_tokens = test_prediction_acc_LLM_judge_easyedit(model, tok, hparams, prompt, locality_ground_truth, device, locality=True)
    else:  # traditional evaluation 
        if 't5' in model_name.lower():
            loc_tokens = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True)
        else:
            loc_tokens = test_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True, vanilla_generation=hparams.alg_name=='GRACE')
        if type(loc_tokens) is not list:
            loc_tokens = [loc_tokens,]

    ret = {
        f"{locality_key}_output": loc_tokens
    }
    return ret

def compute_portability_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    portability_key: str,
    prompt: typing.Union[str, List[str]],
    ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:
    # using real-world evaluation: autoregressive decoding, natural stop criteria, LLM-as-a-Judge
    if hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "LLM-judge":
        portability_correct = test_prediction_acc_LLM_judge_easyedit(model, tok, hparams, prompt, ground_truth, device, locality=False)
    elif hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "generate-text":
        portability_correct = test_prediction_acc_LLM_judge_easyedit(model, tok, hparams, prompt, ground_truth, device, locality=False)
    else:  # traditional evaluation
        if 't5' in model_name.lower():
            portability_correct = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, ground_truth, device)
        else:
            portability_correct = test_prediction_acc(model, tok, hparams, prompt, ground_truth, device, vanilla_generation=hparams.alg_name=='GRACE')

    ret = {
        f"{portability_key}_acc": portability_correct
    }
    return ret

def compute_icl_edit_quality(
        model,
        model_name,
        hparams: HyperParams,
        tok: AutoTokenizer,
        icl_examples,
        record: typing.Dict,
        device,
        pre_edit: bool = False,
        test_generation = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )
    prompt = record["prompt"]
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    new_fact = f'New Fact: {prompt} {target_new}\nPrompt: {prompt}'

    if pre_edit:
        edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target_new, prompt)
    else:
        edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target_new, new_fact)
    ret = {
        f"rewrite_acc": [edit_acc]
    }
    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase is not None:
        rephrase_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                   target_new, f'New Fact: {prompt} {target_new}\nPrompt: {rephrase}')
        ret['rephrase_acc'] = rephrase_acc

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            if isinstance(record['locality'][locality_key]['ground_truth'], list):
                pre_neighbor = []
                post_neighbor = []
                for x_a, x_p in zip(record['locality'][locality_key]['ground_truth'],
                                    record['locality'][locality_key]['prompt']):
                    tmp_pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''], x_a,
                                                   f"{x_p}", neighborhood=True)
                    tmp_post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples, x_a,
                                                    f"New Fact: {prompt} {target_new}\nPrompt: {x_p}",
                                                    neighborhood=True)
                    if type(tmp_pre_neighbor) is not list:
                        tmp_pre_neighbor = [tmp_pre_neighbor, ]
                    if type(tmp_post_neighbor) is not list:
                        tmp_post_neighbor = [tmp_post_neighbor, ]
                    assert len(tmp_pre_neighbor) == len(tmp_post_neighbor)
                    pre_neighbor.append(tmp_pre_neighbor)
                    post_neighbor.append(tmp_post_neighbor)
                res = []
                for ans, label in zip(pre_neighbor, post_neighbor):
                    temp_acc = np.mean(np.equal(ans, label))
                    if np.isnan(temp_acc):
                        continue
                    res.append(temp_acc)
                ret['locality'][f'{locality_key}_acc'] = res
            else:
                pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''],
                                           record['locality'][locality_key]['ground_truth'],
                                           f"{record['locality'][locality_key]['prompt']}",
                                           neighborhood=True)
                post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                            record['locality'][locality_key]['ground_truth'],
                                            f"New Fact: {prompt} {target_new}\nPrompt: {record['locality'][locality_key]['prompt']}",
                                            neighborhood=True)
                if type(pre_neighbor) is not list:
                    pre_neighbor = [pre_neighbor, ]
                if type(post_neighbor) is not list:
                    post_neighbor = [post_neighbor, ]
                assert len(pre_neighbor) == len(post_neighbor)

                ret['locality'][f'{locality_key}_acc'] = np.mean(np.equal(pre_neighbor, post_neighbor))
    # Form a list of lists of prefixes to test.
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            if pre_edit:
                icl_input = ['']
                x_prefix = ""
            else:
                icl_input = icl_examples
                x_prefix = f"New Fact: {prompt} {target_new}\nPrompt: "
            if isinstance(record['portability'][portability_key]['ground_truth'], list):
                portability_acc = []
                for x_a, x_p in zip(record['portability'][portability_key]['ground_truth'],
                                    record['portability'][portability_key]['prompt']):
                    tmp_portability_acc = icl_lm_eval(model, model_name, hparams, tok, icl_input, x_a,
                                                      f"{x_prefix}{x_p}")
                portability_acc.append(tmp_portability_acc)
            else:
                portability_acc = icl_lm_eval(model, model_name, hparams, tok, icl_input,
                                              record['portability'][portability_key]['ground_truth'],
                                              f"{x_prefix}{record['portability'][portability_key]['prompt']}")
            ret['portability'][f'{portability_key}_acc'] = portability_acc

    if test_generation:
        ret['fluency'] = test_generation_quality(model=model,tok=tok, prefixes=new_fact if isinstance(new_fact,list) else [new_fact,], max_out_len=100, vanilla_generation=False)
    return ret

def icl_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        neighborhood=False
)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')
    if 't5' in model_name.lower():
        target_len = len(tokenizer.encode(target))
        target_ids = tokenizer(f'{x} {target}', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples), return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).logits
            ans = torch.argmax(logits, dim=-1)[:,-target_len:-1].squeeze()
            target_ids = target_ids[:,-target_len:-1]
            if neighborhood:
                return ans.squeeze().detach().cpu().numpy().tolist()
            return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    elif 'llama' in model_name.lower():
        target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,1:]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    else:
        target_ids = tokenizer(' ' + target + '\n', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,:-1]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
