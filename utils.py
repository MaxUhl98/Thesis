import os
import pandas as pd
import requests
import yaml
import json
from jinja2 import Template
from huggingface_hub import hf_hub_download
import torch
import gc
from transformers import AutoTokenizer

SGLANG_URL = os.getenv("SGLANG_URL", "http://localhost:30000/generate")
LLM = 'meta-llama/Llama-3.1-8B-Instruct'
qa_template = "Frage: {question}\n\nAntwort: {answer}"
yes_no_system_prompt = """Du bist ein medizinischer Experte.

AUFGABE: Bewerte, ob die Antwort für die gegebene Frage relevant ist.

FORMAT: Antworte nur mit "Ja" oder "Nein" - keine anderen Wörter, keine Erklärungen.

RELEVANZ-KRITERIEN:
- "Ja": Frage wurde korrekt beantwortet.
- "Nein": Frage wurde inkorrekt beantwortet.

""".strip()


def load_chat_template(model_id: str) -> Template:
    """
    Downloads and loads a Jinja2 chat template from a Hugging Face tokenizer config file
    in either JSON or YAML format.

    :param model_id: The Hugging Face model identifier.
    :return: A Jinja2 Template object representing the chat template.
    :raises ValueError: If the chat template cannot be found or loaded.
    """
    # Try both JSON and YAML formats
    for ext in ["json", "yaml"]:
        try:
            config_path = hf_hub_download(model_id, filename=f"tokenizer_config.{ext}")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f) if ext == "json" else yaml.safe_load(f)
            if "chat_template" in config:
                return Template(config["chat_template"])
        except Exception:
            continue
    raise ValueError(f"Failed to load chat template for model {model_id}!")


def apply_chat_template(chat_template: Template, messages: list[dict[str, str]]) -> str:
    """
    Renders a chat template with a list of user/system messages.

    :param chat_template: A Jinja2 template with placeholders for chat messages.
    :param messages: A list of dictionaries representing chat messages.
    :return: The rendered prompt string.
    """
    rendered = chat_template.render(messages=messages, add_generation_prompt=True)
    return rendered

def get_user_system_format_messages(prompt: str, behaviour: str) -> list[dict[str, str]]:
    """
    Constructs a message list suitable for a chat-based LLM input, with a system
    prompt followed by a user prompt.

    :param prompt: The user prompt or question.
    :param behaviour: The system prompt describing model behavior.
    :return: A list of formatted messages.
    """
    messages = [{"role": "system", "content": behaviour}, {"role": "user", "content": prompt}]
    return messages


class ComputeClient:
    """
    Client interface for communicating with LLM and embedding servers.

    Attributes:
        url (str): The endpoint URL for the language model (LLM).
    """

    def __init__(
        self, url=SGLANG_URL
    ):
        self.url = url

    def call_llm(
        self, behaviour: str|list[str], prompts: list[str]|str, parameters: dict = None
    ) -> list[str]:
        """
        Calls the LLM to generate responses for a batch of prompts, applying the specified behavior.

        :param behaviour: The system prompt (or prompts) describing the model's behavior.
        :param prompts: A list of user input strings.
        :param parameters: Optional dictionary of sampling parameters for generation (e.g., temperature, max_tokens).
        :return: A list of generated responses from the LLM.
        """
        template = load_chat_template(LLM)
        if parameters is None:
            parameters = {}
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(behaviour, str):
            system_prompts = [behaviour for _ in prompts]
        else:
            system_prompts = behaviour
        prompts_in_message_format = [get_user_system_format_messages(prompt, system_prompt) for prompt, system_prompt in zip(prompts, system_prompts)]
        formated_prompts = [
            apply_chat_template(template, prompts)
            for prompts in prompts_in_message_format
        ]
        data = {"text": formated_prompts, 'sampling_params': parameters}
        response = requests.post(self.url, json=data)
        response_json = response.json()
        results = [response['text'] for response in response_json]

        return results

def get_user_system_assistant_format_messages(prompt: str, behaviour: str, response:str) -> list[dict[str, str]]:
    """
    Constructs a message list suitable for a chat-based LLM input, with a system
    prompt followed by a user prompt.

    :param prompt: The user prompt or question.
    :param behaviour: The system prompt describing model behavior.
    :return: A list of formatted messages.
    """
    messages = [{"role": "system", "content": behaviour}, {"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    return messages


def convert_qa_to_prompt_answer_format(question:str, answers:dict[str|int,str], correct_answer_letter:str, system_prompt:str=yes_no_system_prompt) -> list[list[dict[str,str]]]:
    """
    Converts a multiple-choice question and its answers into a list of formatted prompt-answer message sets
    using a boolean (yes/no) classification format.

    Each answer option is turned into a prompt asking whether it's correct. The function determines whether
    each answer matches the correct one and labels it as "Ja" (yes) or "Nein" (no), returning the result in a
    format suitable for language model training.

    :param question: The question string.
    :param answers: A dictionary mapping option letters (e.g., 'A', 'B', 'C') or indices to answer strings.
    :param correct_answer_letter: The letter (e.g., 'A', 'B') indicating the correct answer in the `answers` dict.
    :param system_prompt: A predefined system instruction or context string used to structure the prompt messages.
    :return: A list where each element is a list of dicts with role-content keys representing a conversation
             (e.g., [{"role": "system", "content": ...}, {"role": "user", "content": ...}, {"role": "assistant", "content": ...}]).
    """
    prompts = [qa_template.format(question=question, answer=answer.strip('"\'').strip()) for answer in answers.values()]
    bool_answers = ['Ja' if str(letter).strip().startswith(str(correct_answer_letter).strip()) else 'Nein' for letter in answers]
    final_texts = [get_user_system_assistant_format_messages(prompt, system_prompt, answer) for prompt, answer in zip(prompts, bool_answers)]
    assert "Ja" in bool_answers, AssertionError(
        f"Answers:\n\n{answers}\n\nCorrect answer:\n{correct_answer_letter}\n{[x == correct_answer_letter for x in answers]}"
    )
    return final_texts


def convert_conversation_to_completion_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits a conversation into a prompt-completion format for training completion-based models.

    The function assumes each value in the `conversation` column is a list of messages. It treats all but the
    last message as the prompt (the conversation history) and the final message as the expected model response.

    :param df: A pandas DataFrame containing a 'conversation' column, where each entry is a list of message dicts.
    :return: The same DataFrame with two new columns:
             - 'prompt': containing all messages except the last one
             - 'completion': containing the last message as a list
    """
    df["prompt"] = df["conversation"].apply(lambda x: x[:-1])
    df['completion'] = df['conversation'].apply(lambda x: [x[-1]])
    return df

def clear_cuda():
    """
    Clears the CUDA memory.
    :return: None
    """
    # Manually delete all CUDA tensors
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
        except Exception:
            pass

    gc.collect()
    torch.cuda.empty_cache()

def extract_benchmark_prompts(df_benchmark: pd.DataFrame) -> list[list[dict[str, str]]]:
    """
    Extracts benchmark prompts from a pandas DataFrame.
    :param df_benchmark: Benchmark DataFrame.
    :return: List of Prompts in the standard conversation format of
    [{'role': 'system', 'content': system prompt}, {'role': 'user', 'content': user prompt}].
    """
    conversations = df_benchmark["conversation"].tolist()
    prompts = [data[:-1].tolist() for data in conversations]
    return prompts


def load_benchmark_prompts_and_answers(benchmark_path: str = "../data/Benchmark/german_medical_exam_boolean_questions.feather") -> tuple[list[str], list[str]]:
    """
    Loads benchmark prompt conversations and their corresponding answers from a Feather file.

    Each conversation is expected to be a list of message dictionaries, where the last message contains the answer.
    The prompts are extracted as all messages except the last one and the answers are extracted from the last message's 'content'.

    :param benchmark_path: Path to the Feather file containing the benchmark data. Defaults to
                           "../data/Benchmark/german_medical_exam_boolean_questions.feather".
    :return: A tuple of two lists:
             - prompts: List of prompt message sequences (each prompt is a list of messages except the last).
             - answers: List of answer strings corresponding to each prompt.
    """
    benchmark_data = pd.read_feather(
        benchmark_path
    )
    conversations = benchmark_data['conversation'].tolist()
    prompts = [data[:-1].tolist() for data in conversations]
    answers = [data[-1]['content'] for data in conversations]
    return prompts, answers

def get_example_prompt(model_name: str, max_input_length: int) -> str:
    """
    Generates a token-limited prompt for text generation models to produce output
    based on a provided example. The function uses a tokenizer associated with the
    specified language model to truncate the example text to fit within the maximum
    allowed token limit. The truncated text is then returned as a prompt string.

    :param model_name: Name of the language model for which the tokenizer should
      be loaded.
    :param max_input_length: Maximum number of tokens allowed for the generated
      prompt.
    :return: A truncated prompt string suitable for input to the specified
      language model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = max_input_length
    dummy_text = "lorem ipsum, dolor amet\n" * 10**5
    prompt = f"""Generate as much text similar to the example as its possible for you. Example:\n\n{dummy_text}"""
    token_limited_prompt = tokenizer.decode(
        tokenizer.encode(prompt, padding=False, truncation=True),
        skip_special_tokens=True,
    )
    return token_limited_prompt