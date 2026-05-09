# prompt_builders/score_math_builder.py

import sys
sys.path.append('/')
sys.path.append('envs/')

import os
import json
from functools import partial
from collections.abc import Iterable

import torch.nn as nn

class PromptBuilder(nn.Module):
    """
    An OOP-style prompt builder for a math/score domain (like your original code).
    It inherits from BasePromptBuilder, using more robust method names.
    """

    ## SCORE Prompts

    # initial_instructions = (
    #     "You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step."
    #     " At the end of the Solution, when you give your final answer, write it in the form"
    #     ' \"Final Answer: The final answer is $answer$\"'
    # )

    # # The instructions for correction
    # correction_instructions = (
    #     "There might be an error in the solution above because of lack of understanding of the question."
    #     " Please correct the error, if any, and rewrite the solution. Only output the final solution!"
    #     ' At the end of the Solution, when you give your final answer, write it in the form'
    #     ' \"Final Answer: The final answer is $answer$.\"'
    # )


    initial_instructions = None

    # # The instructions for correction
    correction_instructions = (
        "There might be an error in the solution above because of lack of understanding of the question."
        " Please correct the error, if any, and rewrite the solution. Put your final answer within \\boxed{{}}"
    )

    init_system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}"


    def build_initial_prompt_from_asm(
        self,
        sample,
        tokenizer,
        question_col="question",
        few_shot_prompts=None,
        tokenize=False,
        *args,
        **kwargs
    ):
        """
        Returns a single prompt string for initial generation.
        """

        # The user question
        question_text = sample.get(question_col, "")
        user_question = self._create_user_question(question_text)

        # Build the message sequence
        messages = compose_chat_messages(
            system_prompt=self.init_system_prompt,
            instructions=self.initial_instructions,
            user_question=user_question,
            few_shot_prompts=few_shot_prompts
        )

        # Return the final merged prompt
        return tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=True
        )

    def build_initial_prompt_from_pscode(
        self,
        sample,
        tokenizer,
        question_col="question",
        few_shot_prompts=None,
        tokenize=False,
        *args,
        **kwargs
    ):
        """
        Returns a single prompt string for initial generation.
        """

        # The user question
        question_text = sample.get(question_col, "")
        user_question = self._create_user_question(question_text)

        # Build the message sequence
        messages = compose_chat_messages(
            system_prompt=self.init_system_prompt,
            instructions=self.initial_instructions,
            user_question=user_question,
            few_shot_prompts=few_shot_prompts
        )

        # Return the final merged prompt
        return tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=True
        )

    def build_syntax_correction_prompt(
        self,
        sample,
        tokenizer,
        question_col="question",
        initial_answer_col="inital_answer",
        few_shot_prompts=None,
        tokenize=False,
        *args,
        **kwargs
    ):
        """
        Returns a list of final prompt strings (one per initial answer).
        """

        question_text = sample.get(question_col, "")
        user_question = self._create_user_question(question_text)
        initial_answers = sample.get(initial_answer_col, [])

        all_correction_prompts = []
        for init_ans in initial_answers:
            messages = [
                # score type
             #   {"role": "user", "content": self.initial_instructions},
            # qwen2.5 type
                {"role": "system", "content": self.init_system_prompt},
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": init_ans},
                {"role": "user", "content": self.correction_instructions},
            ]


            # Convert messages to final text
            final_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=True
            )
            all_correction_prompts.append(final_prompt)

        return all_correction_prompts

    def build_dfg_correction_prompt(
        self,
        question,
        init_answer,
        correction,
        *args,
        **kwargs
    ):
        """
        Creates a short conversation with the final corrected answer
        as the assistant's last response.
        Returns a list of message dicts.
        """
        user_question = self._create_user_question(question)

        messages = [
            # qwen type system prompt
            {"role": "system", "content": self.init_system_prompt},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": init_answer},
            {"role": "user", "content": self.correction_instructions},
            {"role": "assistant", "content": correction},
        ]

        return messages

    def build_format_correction_prompt(
        self,
        question,
        init_answer,
        correction,
        *args,
        **kwargs
    ):
        """
        Creates a short conversation with the final corrected answer
        as the assistant's last response.
        Returns a list of message dicts.
        """
        user_question = self._create_user_question(question)

        messages = [
            # qwen type system prompt
            {"role": "system", "content": self.init_system_prompt},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": init_answer},
            {"role": "user", "content": self.correction_instructions},
            {"role": "assistant", "content": correction},
        ]

        return messages

    def build_compile_correction_prompt(
        self,
        question,
        init_answer,
        correction,
        *args,
        **kwargs
    ):
        """
        Creates a short conversation with the final corrected answer
        as the assistant's last response.
        Returns a list of message dicts.
        """
        user_question = self._create_user_question(question)

        messages = [
            # qwen type system prompt
            {"role": "system", "content": self.init_system_prompt},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": init_answer},
            {"role": "user", "content": self.correction_instructions},
            {"role": "assistant", "content": correction},
        ]

        return messages

    def build_format_correction_prompt(
        self,
        question,
        init_answer,
        correction,
        *args,
        **kwargs
    ):
        """
        Creates a short conversation with the final corrected answer
        as the assistant's last response.
        Returns a list of message dicts.
        """
        user_question = self._create_user_question(question)

        messages = [
            # qwen type system prompt
            {"role": "system", "content": self.init_system_prompt},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": init_answer},
            {"role": "user", "content": self.correction_instructions},
            {"role": "assistant", "content": correction},
        ]

        return messages

    def build_execute_correction_prompt(
        self,
        question,
        init_answer,
        correction,
        *args,
        **kwargs
    ):
        """
        Creates a short conversation with the final corrected answer
        as the assistant's last response.
        Returns a list of message dicts.
        """
        user_question = self._create_user_question(question)

        messages = [
            # qwen type system prompt
            {"role": "system", "content": self.init_system_prompt},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": init_answer},
            {"role": "user", "content": self.correction_instructions},
            {"role": "assistant", "content": correction},
        ]

        return messages



# ==================================================================================================================================================================


# prompt_builders/star_qa_builder.py

import sys

sys.path.append('/')
sys.path.append('envs/')

import os
import json
from functools import partial
from collections.abc import Iterable

from .base import BasePromptBuilder
from prompts.prompt_schemas import compose_chat_messages


class QAPromptBuilder(BasePromptBuilder):
    # For the *initial generation* step
    system_prompt_init = (
        "You are a helpful reasoning assistant in general domain question answering. "
        "Please reason through the question step by step very shortly before giving a final answer."
    )
    initial_instructions = (
        "Generate a short chain-of-thought rationale very shortly, and then provide the final answer.\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )

    # For the *correction* step
    system_prompt_corr = (
        "You are a helpful reasoning assistant in general domain question answering. "
        "Your task is to correct the initial response if it is incorrect."
    )
    correction_instructions = (
        "Consider the question and the initial answer. "
        "Generate a correction to the initial answer if it is incorrect. "
        "Disregard the information you already have, look for other options. "
        "Do not use the information that does not match your criteria."
        "Think about your correction step-by-step and output answer in following format:\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )

    def _create_user_question(self, question_text: str):
        return (
            f"Question:\n{question_text}\n\n"
            "Reason step by step very shortly, then conclude with the answer. "
            "Strictly follow format Step-by-step reasoning: and Final Answer:"
        )

    def build_initial_generation_prompt(
            self,
            sample,
            tokenizer,
            question_col="question",
            few_shot_prompts=None,
            tokenize=False,
            *args,
            **kwargs
    ):
        # Build user question
        question_text = sample.get(question_col, "")
        user_question = self._create_user_question(question_text)

        # Compose messages
        messages = compose_chat_messages(
            system_prompt=self.system_prompt_init,
            instructions=self.initial_instructions,
            user_question=user_question,
            few_shot_prompts=few_shot_prompts
        )

        # Return merged prompt
        return tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=True
        )

    def build_correction_prompt(
            self,
            sample,
            tokenizer,
            question_col="question",
            initial_answer_col="inital_answer",
            few_shot_prompts=None,
            tokenize=False,
            *args,
            **kwargs
    ):
        question_text = sample.get(question_col, "")
        user_question = self._create_user_question(question_text)
        initial_answers = sample.get(initial_answer_col, [])

        all_correction_prompts = []
        for init_ans in initial_answers:
            messages = [
                {"role": "user", "content": self.initial_instructions},
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": init_ans},
                {"role": "user", "content": self.correction_instructions},
            ]

            # Convert messages to final text
            final_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=True
            )
            all_correction_prompts.append(final_prompt)

        return all_correction_prompts

    def build_correction_messages_with_final_answer(
            self,
            question,
            init_answer,
            correction,
            few_shot_prompts=None,
            *args,
            **kwargs
    ):
        """
        New helper method for a single conversation that ends
        with the final correction as the assistant's last message.

        The 'question' is optional but let's incorporate it so we have context.
        The idea is to mirror the logic of 'build_correction_prompt'
        but actually include the final corrected answer as the assistant response.
        """

        # 1) system -> self.system_prompt_corr
        # messages.append({"role": "system", "content": self.system_prompt_corr})

        user_question = self._create_user_question(question)

        messages = [
            {"role": "user", "content": self.initial_instructions},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": init_answer},
            {"role": "user", "content": self.correction_instructions},
            {"role": "assistant", "content": correction},
        ]

        return messages


# ===================================================================================================================================================


from collections.abc import Iterable
import os
import json


def load_few_shot_prompts(few_shot_dir, prompt_type):
    """
    Loads a single few-shot file from `few_shot_dir/` based on the prompt_type.
    For example, if prompt_type="critic", we look for "critic.json" in that dir.

    Returns a list of role/content dicts, or None if no file is found.
    """
    file_name = f"{prompt_type}.json"  # e.g. "critic.json"
    file_path = os.path.join(few_shot_dir, file_name)
    if not os.path.exists(file_path):
        print(f"[INFO] No few-shot file found for prompt_type='{prompt_type}' at '{file_path}'. Using none.")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Few-shot file must contain a JSON list of role/content dicts. Found: {file_path}")

    print(f"[INFO] Loaded {len(data)} few-shot messages from {file_path}")
    return data


def flatten_list(nested_list):
    """
    Recursively flatten a nested list (ignores strings as iterables).
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, Iterable) and not isinstance(item, str):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def compose_chat_messages(
        system_prompt: str,
        instructions: str,
        user_question: str,
        few_shot_prompts=None
):
    """
    Constructs the chat messages in the following order:
       1) system_prompt  (role: system)
       2) instructions   (role: user)
       3) few_shot_prompts (list of roles: user/assistant)
       4) user_question  (role: user)
    """
    messages = []

    # 1) System prompt
    if system_prompt:
        messages.append({"role": "user", "content": system_prompt})

    # 2) Instructions
    if instructions:
        messages.append({"role": "user", "content": instructions})

    # 3) Few-shot prompts
    if few_shot_prompts and isinstance(few_shot_prompts, list):
        messages.extend(few_shot_prompts)

    # 4) Actual user question
    messages.append({"role": "user", "content": user_question})

    return messages


def generate_prompt(
        sample,
        tokenizer,
        use_context_col="none",
        question_col="question",
        few_shot_prompts=None,
        *args,
        **kwargs
):
    """
    Generates a Q&A prompt, optionally including context. Uses `question_col` for question text.
    """
    system_prompt = "You are a helpful assistant tasked with answering questions."

    instructions = (
        "Answer the following question concisely.\n"
    )
    if use_context_col != "none":
        instructions = (
                "Use the following documents if they are relevant:\n\n"
                + instructions
        )

    # Potential context
    context_str = ""
    if use_context_col != "none":
        all_context = sample.get(use_context_col, "")
        if isinstance(all_context, list):
            all_context = [f"Passage {i}: {passage}" for i, passage in enumerate(all_context)]
            all_context = "\n".join(all_context)
        context_str = f"{all_context}\n\n"

    question_text = sample.get(question_col, "No question provided.")
    user_question = (
        f"{context_str}"
        f"Question: {question_text}\n\n"
        "Answer briefly:"
    )

    messages = compose_chat_messages(
        system_prompt=system_prompt,
        instructions=instructions,
        user_question=user_question,
        few_shot_prompts=few_shot_prompts
    )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def generate_when_to_retrieve_prompt(
        sample,
        tokenizer,
        use_context_col="none",
        question_col="question_text",
        few_shot_prompts=None,
        *args,
        **kwargs
):
    """
    If context is provided, include it. Otherwise just rely on internal knowledge.
    By default, we assume the question is in `question_text` (but you can override).
    """
    system_prompt = "You are a helpful assistant."

    instructions = ""
    context_str = ""
    if use_context_col != "none":
        all_context = sample.get(use_context_col, "")
        if isinstance(all_context, list):
            all_context = [f"Passage {i}: {passage}" for i, passage in enumerate(all_context)]
            all_context = "\n".join(all_context)
        instructions = (
            "Use the provided information or your internal knowledge to answer:\n"
        )
        context_str = f"{all_context}\n\n"
    else:
        instructions = (
            "Answer based on your internal knowledge:\n"
        )

    question_text = sample.get(question_col, "No question provided.")
    user_question = f"{context_str}Question: {question_text}\nAnswer in one or few words:"

    messages = compose_chat_messages(
        system_prompt=system_prompt,
        instructions=instructions,
        user_question=user_question,
        few_shot_prompts=few_shot_prompts
    )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def critic_prompt(
        sample,
        tokenizer,
        answer_col,
        question_col="question",
        few_shot_prompts=None,
        *args,
        **kwargs
):
    """
    Critiques an existing answer, focusing on factual correctness.
    `question_col` determines which field has the question text.
    """
    system_prompt = (
        "You are a helpful critic assistant. Identify factual errors in the given answer, "
        "or confirm correctness if there are no errors."
    )
    instructions = (
        "Explain any factual inaccuracies succinctly (in bullet points). If correct, say so briefly. "
        "Do not provide a revised answer yourself."
    )
    question_text = sample.get(question_col, "No question provided.")
    user_question = (
        f"QUESTION: {question_text}\n"
        f"ANSWER: {sample.get(answer_col, 'No answer provided.')}"
    )

    messages = compose_chat_messages(
        system_prompt=system_prompt,
        instructions=instructions,
        user_question=user_question,
        few_shot_prompts=few_shot_prompts
    )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def revision_prompt(
        sample,
        tokenizer,
        critic_col,
        answer_col,
        question_col="question",
        few_shot_prompts=None,
        *args,
        **kwargs
):
    """
    Generates revision prompts based on criticisms found in `critic_col`.
    Each criticism yields a sub-prompt.
    Uses `question_col` to fetch the question text.
    """
    system_prompt = (
        "You are a helpful assistant tasked with revising answers based on provided criticisms."
    )
    instructions = (
        "Revise the INITIAL ANSWER in one sentence based on the given CRITICISM."
    )

    criticisms = sample.get(critic_col, [])
    if not isinstance(criticisms, list):
        criticisms = [criticisms]  # In case it's a single string

    result_prompts = []
    question_text = sample.get(question_col, "No question provided.")
    initial_answer = sample.get(answer_col, "No initial answer provided.")

    for criticism in criticisms:
        user_question = (
            f"QUESTION: {question_text}\n\n"
            f"INITIAL ANSWER: {initial_answer}\n\n"
            f"CRITICISM: {criticism}\n\n"
            "REVISED ANSWER:"
        )

        messages = compose_chat_messages(
            system_prompt=system_prompt,
            instructions=instructions,
            user_question=user_question,
            few_shot_prompts=few_shot_prompts
        )

        result_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        result_prompts.append(result_prompt)

    return result_prompts


def verbalized_confidence_1s(
        sample,
        tokenizer,
        question_col="question",
        few_shot_prompts=None,
        *args,
        **kwargs
):
    """
    Prompts the model to provide a step-by-step solution and probability (0.0 to 1.0).
    Uses `question_col` to extract the question from `sample`.
    """
    system_prompt = (
        "You are a helpful assistant tasked with answering questions and providing "
        "a confidence level for your answers."
    )

    instructions = (
        'Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question.'
        'Give ONLY the guess and probability, no other words orexplanation.'
        'For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>'
        'Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n\n'
    )

    question_text = sample.get(question_col, "No question provided.")
    user_question = f"Question: {question_text}"

    messages = compose_chat_messages(
        system_prompt=system_prompt,
        instructions=instructions,
        user_question=user_question,
        few_shot_prompts=few_shot_prompts
    )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def verbalized_confidence_1s_topk(
        sample,
        tokenizer,
        k,
        question_col="question",
        few_shot_prompts=None,
        *args,
        **kwargs
):
    """
    Prompts the model to provide the top k guesses with probabilities (0.0 to 1.0) for the given question.
    Uses `question_col` to extract the question from `sample`.
    """

    # System prompt: basic introduction about the assistant
    system_prompt = (
        "You are a helpful assistant tasked with answering questions and providing "
        "a confidence level for each of your answers."
    )

    # Instructions for the model
    instructions = (
        f"Provide your {k} best guesses and the probability that each is correct (0.0 to 1.0) for the following question. "
        "Give ONLY the guesses and probabilities, no other words or explanation. For example:\n\n"
    )

    # Dynamically build the instruction part for the top k guesses
    top_k_instructions = ""
    for i in range(1, k + 1):
        top_k_instructions += f"G{i}: <the {i}-th most likely guess, as short as possible; not a complete sentence, just the guess!>\n"
        top_k_instructions += f"P{i}: <the probability between 0.0 and 1.0 that G{i} is correct, without any extra commentary whatsoever; just the probability!>\n"

    # Add the question text as the final prompt
    question_text = sample.get(question_col, "No question provided.")
    user_question = f"The question is: {question_text}"

    # Construct messages for the tokenizer
    messages = compose_chat_messages(
        system_prompt=system_prompt,
        instructions=instructions,
        user_question=user_question,
        few_shot_prompts=few_shot_prompts
    )

    # Return the formatted prompt using the tokenizer
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


# A dictionary to easily retrieve prompt functions by name
prompt_mapper = {
    "generate": generate_prompt,
    "critic": critic_prompt,
    "revise": revision_prompt,
    "when_to_retrieve": generate_when_to_retrieve_prompt,
    "verbalized_1s": verbalized_confidence_1s,
    "verbalized_1s_topk": verbalized_confidence_1s_topk
}
