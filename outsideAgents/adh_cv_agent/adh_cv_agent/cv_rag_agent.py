from .rag_without_vecdb import do_answer_question


async def chat_with_agent(user_msg: str):
    print(f"Received user message: '{user_msg}'")
    agent_reply = await do_answer_question(user_msg)

    yield ('TEXT', agent_reply)
