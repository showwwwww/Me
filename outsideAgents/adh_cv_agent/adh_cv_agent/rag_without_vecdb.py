import argparse
import asyncio
import nltk
import os
import pickle
import re
import requests
import tiktoken

from dataclasses import dataclass
from io import BytesIO
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader
from typing import List, Dict, Any
from typing_extensions import TypedDict, Any

from agents import (
    Agent,
    ModelSettings,
    Runner,
    RunResult,
    function_tool,
    set_tracing_disabled,
)
from agents.models.openai_provider import OpenAIProvider


set_tracing_disabled(disabled=True)


# Global tokenizer name to use consistently throughout the code
TOKENIZER_NAME = "o200k_base"
chunks_file = 'data/document_chunks.pkl'

# model_id = "gpt-4.1-mini"
# model_id = "qwen3:8b"
# model_id = "qwen3:14b"
# model_id = "qwen3-235b-a22b"
model_id = "qwen-turbo-latest"

API_KEY = os.getenv("EXAMPLE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
if model_id.startswith("qwen3:"):
    API_KEY = "ollama"
    BASE_URL = "http://127.0.0.1:11434/v1"
    
if model_id == "gpt-4.1-mini":
    provider = OpenAIProvider(
        api_key=API_KEY,
        use_responses=False
        )
else:
    provider = OpenAIProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        use_responses=False
        )
model = provider.get_model(model_id)

extra_body = {"enable_thinking": False}
if model_id == "gpt-4.1-mini" or model_id.startswith("qwen3:"):
    extra_body = {}


system_message = """You are a senior frontend engineer. Your task is to:
1. Identify which text chunks might contain information to answer the user's question
2. Record your reasoning in a scratchpad for later reference
3. Choose chunks that are most likely relevant. Be selective, but thorough. Choose as many chunks as you need to answer the question, but avoid selecting too many.

First think carefully about what information would help answer the question, then evaluate each chunk.
"""
if model_id.startswith("qwen3:"):
    system_message = f"{system_message} /no_think"


class CustomContext:
    def __init__(self, context_variables: dict):
        self.context_variables = context_variables


@function_tool
def update_scratchpad(text: str, depth: int, scratchpad: str):
    """Record your reasoning about why certain chunks were selected.

    Keyword arguments:
      text: Your reasoning about the chunk(s) selection.
    """
    
    print(f"Invoke update_scratchpad() ...")
    scratchpad_entry = f"DEPTH {depth} REASONING:\n{text}"
    if scratchpad:
        scratchpad += "\n\n" + scratchpad_entry
    else:
        scratchpad = scratchpad_entry

    return scratchpad


def create_agent1():
    global model, system_message, extra_body
    
    my_agent = Agent[CustomContext](
        name="DocumentNavigator1",
        instructions=system_message,
        model=model,
        model_settings=ModelSettings(extra_body=extra_body),
        tools=[update_scratchpad],
        tool_use_behavior="stop_on_first_tool"
    )
    return my_agent


class OutputType(TypedDict):
    chunk_ids: list[int]


def create_agent2():
    global model, system_message, extra_body

    my_agent = Agent[CustomContext](
        name="DocumentNavigator2",
        instructions=system_message,
        output_type=OutputType,
        model=model,
        model_settings=ModelSettings(extra_body=extra_body),
    )
    return my_agent


def create_agent3():
    global model, extra_body

    system_message = """You are a little private secretary. Please answer my question based on the context information I give. The context information is located in the <context>...</context> tags. If you can't answer my question based on the context information, you should just say: "Sorry, I can't answer this question." in Chinese. The response of my question should be plain text, don't use JSON or Markdown format.
    """
    if model_id.startswith("qwen3:"):
        system_message = f"{system_message} /no_think"

    my_agent = Agent(
        name="PrivateSecretary",
        instructions=system_message,
        model=model,
        model_settings=ModelSettings(extra_body=extra_body),
    )
    return my_agent


def load_document(file_path: str) -> str:
    """Load a document from a local file and return its text content."""
    print(f"Loading document from local file: {file_path}...")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at {file_path}")

    # 检查文件是否是PDF
    if not file_path.lower().endswith('.pdf'):
        raise ValueError("Only PDF files are supported")

    # 读取本地PDF文件
    try:
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)

            full_text = ""
            max_page = 920  # Page cutoff before section 1000 (Interferences)

            for i, page in enumerate(pdf_reader.pages):
                if i >= max_page:
                    break
                page_text = page.extract_text()
                if page_text:  # 确保页面有文本内容
                    full_text += page_text + "\n"

            # 统计信息
            word_count = len(re.findall(r'\b\w+\b', full_text))
            tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
            token_count = len(tokenizer.encode(full_text))

            print(f"Document loaded: {len(pdf_reader.pages)} pages, {word_count} words, {token_count} tokens")
            return full_text

    except Exception as e:
        raise RuntimeError(f"Failed to read PDF file: {str(e)}")


def split_into_20_chunks(text: str, min_tokens: int = 500) -> List[Dict[str, Any]]:
    """
    Split text into up to 20 chunks, respecting sentence boundaries and ensuring
    each chunk has at least min_tokens (unless it's the last chunk).

    Args:
        text: The text to split
        min_tokens: The minimum number of tokens per chunk (default: 500)

    Returns:
        A list of dictionaries where each dictionary has:
        - id: The chunk ID (0-19)
        - text: The chunk text content
    """
    # First, split the text into sentences
    sentences = sent_tokenize(text)

    # Get tokenizer for counting tokens
    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)

    # Create chunks that respect sentence boundaries and minimum token count
    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0

    for sentence in sentences:
        # Count tokens in this sentence
        sentence_tokens = len(tokenizer.encode(sentence))

        # If adding this sentence would make the chunk too large AND we already have the minimum tokens,
        # finalize the current chunk and start a new one
        if (current_chunk_tokens + sentence_tokens > min_tokens * 2) and current_chunk_tokens >= min_tokens:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append({
                "id": len(chunks),  # Integer ID instead of string
                "text": chunk_text
            })
            current_chunk_sentences = [sentence]
            current_chunk_tokens = sentence_tokens
        else:
            # Add this sentence to the current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sentence_tokens

    # Add the last chunk if there's anything left
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        chunks.append({
            "id": len(chunks),  # Integer ID instead of string
            "text": chunk_text
        })

    # If we have more than 20 chunks, consolidate them
    if len(chunks) > 20:
        # Recombine all text
        all_text = " ".join(chunk["text"] for chunk in chunks)
        # Re-split into exactly 20 chunks, without minimum token requirement
        sentences = sent_tokenize(all_text)
        sentences_per_chunk = len(sentences) // 20 + (1 if len(sentences) % 20 > 0 else 0)

        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            # Get the sentences for this chunk
            chunk_sentences = sentences[i:i+sentences_per_chunk]
            # Join the sentences into a single text
            chunk_text = " ".join(chunk_sentences)
            # Create a chunk object with ID and text
            chunks.append({
                "id": len(chunks),  # Integer ID instead of string
                "text": chunk_text
            })

    # Print chunk statistics
    print(f"Split document into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        token_count = len(tokenizer.encode(chunk["text"]))
        print(f"Chunk {i}: {token_count} tokens")

    return chunks


async def route_chunks(question: str, chunks: List[Dict[str, Any]],
                 depth: int, scratchpad: str = "") -> Dict[str, Any]:
    """
    Ask the model which chunks contain information relevant to the question.
    Maintains a scratchpad for the model's reasoning.
    Uses structured output for chunk selection and required tool calls for scratchpad.

    Args:
        question: The user's question
        chunks: List of chunks to evaluate
        depth: Current depth in the navigation hierarchy
        scratchpad: Current scratchpad content

    Returns:
        Dictionary with selected IDs and updated scratchpad
    """
    global system_message

    print(f"\n==== ROUTING AT DEPTH {depth} ====")
    print(f"Evaluating {len(chunks)} chunks for relevance")

    # Build user message with chunks and current scratchpad
    user_message = f"QUESTION: {question}\n\n"

    if scratchpad:
        user_message += f"CURRENT SCRATCHPAD:\n{scratchpad}\n\n"

    user_message += "TEXT CHUNKS:\n\n"

    max_chunk_cnt = 20
    chunk_cnt = 0
    # Add each chunk to the message
    for chunk in chunks:
        user_message += f"CHUNK {chunk['id']}:\n{chunk['text']}\n\n"
        chunk_cnt += 1
        if chunk_cnt >= max_chunk_cnt:
            break

    user_message = f"{user_message}\n\nFirst, you must use the update_scratchpad function to record your reasoning."
    if model_id.startswith("qwen3:"):
        user_message = f"{user_message} /no_think"

    messages = [
        {"role": "user", "content": user_message}
    ]

    agent1 = create_agent1()

    # Process the scratchpad tool call
    new_scratchpad = scratchpad

    context_var_dict = {
        "depth": depth,
        "scratchpad": new_scratchpad,
    }
    context = CustomContext(context_variables=context_var_dict)
    result: RunResult = await Runner.run(
        agent1, 
        messages,
        context=context
    )
    # print(f"Assistant [{agent1.name}] => {result.final_output}")
    new_scratchpad = result.final_output

    for item in result.new_items:
        if item.type == 'tool_call_item':
            messages.append({
                "type": "function_call", 
                "name": "update_scratchpad",
                "call_id": item.raw_item.call_id,
                "arguments": item.raw_item.arguments
            })
            messages.append({
                "type": "function_call_output",
                "name": "update_scratchpad",
                "call_id": item.raw_item.call_id,
                "output": "Scratchpad updated successfully."
            })
    
    # Second pass: Get structured output for chunk selection
    messages.append({"role": "user", "content": "Now, select the chunks that could contain information to answer the question. Return a JSON object with the list of chunk IDs, there is only one key 'chunk_ids' in the JSON obect, for example: {\"chunk_ids\": [0, 2]}."})
    if model_id.startswith("qwen3:"):
        messages = f"{messages} /no_think"

    agent2 = create_agent2()

    context_var_dict = {
        "depth": depth,
        "scratchpad": new_scratchpad,
    }
    context = CustomContext(context_variables=context_var_dict)
    result: RunResult = await Runner.run(
        agent2, 
        messages,
        context=context
    )

    # Extract selected chunk IDs from structured output
    selected_ids = []
    if result.final_output:
        chunk_data = result.final_output
        selected_ids = chunk_data.get("chunk_ids", [])

    # Display results
    print(f"Selected chunks: {', '.join(str(id) for id in selected_ids)}")
    # print(f"Updated scratchpad:\n{new_scratchpad}")

    return {
        "selected_ids": selected_ids,
        "scratchpad": new_scratchpad
    }

    
async def navigate_to_paragraphs(document_chunks: List[Dict[str, Any]], question: str, max_depth: int = 1) -> Dict[str, Any]:
    """
    Navigate through the document hierarchy to find relevant paragraphs.

    Args:
        document_text: The full document text
        question: The user's question
        max_depth: Maximum depth to navigate before returning paragraphs (default: 1)

    Returns:
        Dictionary with selected paragraphs and final scratchpad
    """
    scratchpad = ""

    # Navigator state - track chunk paths to maintain hierarchy
    chunk_paths = {}  # Maps numeric IDs to path strings for display
    for chunk in document_chunks:
        chunk_paths[chunk["id"]] = str(chunk["id"])

    # Navigate through levels until max_depth or until no chunks remain
    for current_depth in range(max_depth + 1):
        # Call router to get relevant chunks
        result = await route_chunks(question, document_chunks, current_depth, scratchpad)

        # Update scratchpad
        scratchpad = result["scratchpad"]

        # Get selected chunks
        selected_ids = result["selected_ids"]
        selected_chunks = [c for c in document_chunks if c["id"] in selected_ids]

        # If no chunks were selected, return empty result
        if not selected_chunks:
            print("\nNo relevant chunks found.")
            return {"paragraphs": [], "scratchpad": scratchpad}

        # If we've reached max_depth, return the selected chunks
        if current_depth == max_depth:
            print(f"\nReturning {len(selected_chunks)} relevant chunks at depth {current_depth}")

            # Update display IDs to show hierarchy
            for chunk in selected_chunks:
                chunk["display_id"] = chunk_paths[chunk["id"]]

            return {"paragraphs": selected_chunks, "scratchpad": scratchpad}

        # Prepare next level by splitting selected chunks further
        next_level_chunks = []
        next_chunk_id = 0  # Counter for new chunks

        for chunk in selected_chunks:
            # Split this chunk into smaller pieces
            sub_chunks = split_into_20_chunks(chunk["text"], min_tokens=200)

            # Update IDs and maintain path mapping
            for sub_chunk in sub_chunks:
                path = f"{chunk_paths[chunk['id']]}.{sub_chunk['id']}"
                sub_chunk["id"] = next_chunk_id
                chunk_paths[next_chunk_id] = path
                next_level_chunks.append(sub_chunk)
                next_chunk_id += 1

        # Update chunks for next iteration
        document_chunks = next_level_chunks


def do_preprocess(file_path: str):
    global chunks_file

    # 确保输出目录存在
    os.makedirs(os.path.dirname(chunks_file) or '.', exist_ok=True)

    # 下载nltk数据
    try:
        nltk.download('punkt')
    except Exception as e:
        print(f"Warning: Failed to download NLTK data: {str(e)}")

    # 加载文档
    document_text = load_document(file_path)

    # 分割文档
    document_chunks = split_into_20_chunks(document_text, min_tokens=500)

    # 保存结果
    try:
        with open(chunks_file, 'wb') as f:
            pickle.dump(document_chunks, f)
        print(f"Saved processed chunks to {chunks_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to save chunks: {str(e)}")


async def answer_question_by_nav_result(navigation_result, question: str):
    context_content = "\n"
    para_len = len(navigation_result["paragraphs"])
    for i, paragraph in enumerate(navigation_result["paragraphs"][:para_len]):
        display_id = paragraph.get("display_id", str(paragraph["id"]))
        para = f"PARAGRAPH {i+1} (ID: {display_id}):\n{paragraph["text"]}"
        context_content = f"{context_content}{para}\n"

    user_message = f"<context>{context_content}</context>\n\nquestion: {question}"
    if model_id.startswith("qwen3:"):
        user_message = f"{user_message} /no_think"

    messages = [
        {"role": "user", "content": user_message}
    ]

    agent3 = create_agent3()
    result: RunResult = await Runner.run(
        agent3, 
        messages
    )

    return result.final_output


async def do_answer_question(question: str):
    global chunks_file
    
    document_chunks = None
    with open(chunks_file, 'rb') as f:
        document_chunks = pickle.load(f)

    # Run the navigation for a sample question
    navigation_result = await navigate_to_paragraphs(document_chunks, question, max_depth=2)
    if len(navigation_result["paragraphs"]) == 0:
        return "抱歉，数据源中未找到相关信息。"

    # Answer user question by navigation result
    answer = await answer_question_by_nav_result(navigation_result, question)
    return answer
    

async def main_func():
    global chunks_file

    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_pdf = os.path.join(current_dir, "SeniorFrontendEngineer_XiupingHu.pdf")

    parser = argparse.ArgumentParser(description="RAG without vector DB")
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the PDF file')
    parser.add_argument('--pdf', default=default_pdf, help='Path to PDF file')
    args = parser.parse_args()

    if args.preprocess:
        try:
            do_preprocess(args.pdf)
        except Exception as e:
            print(f"Error during preprocessing: {str(e)}")
            return 1
        return

    prompt = "你的问题："
    question = input(prompt)

    answer = await do_answer_question(question)
    print(f"问题回答：{answer}")


if __name__ == "__main__":
    asyncio.run(main_func())
