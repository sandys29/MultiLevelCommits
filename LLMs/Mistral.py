from mistralai import Mistral

def getMistralResponse(MISTRAL,prompt):
    model = "mistral-large-2402"
    client = Mistral(api_key=MISTRAL)
    chat_response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    return chat_response.choices[0].message.content