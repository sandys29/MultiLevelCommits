from openai import OpenAI

def generate_commit_message_gpt(OPENAI_API_KEY,diff_text):
    prompt = f"""The following is a diff which describes the code \
                changes in a commit. Your task is to write a short commit \
                message accordingly. {diff_text} According to the diff, the commit \
                message should be
Commit Message:"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=50
    )

    return response.choices[0].message.content