import openai

openai.api_key = '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'
openai.default_headers = {"x-foo": "true"}


def ask_question_with_sources(user_query):
    global completion
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant to create awareness about dark patternBy analyzing and detecting language of the query respond to the query using the detetcted language precise and crisp example if the detetcted language is in hindi respond in hindi words, if the detetcted language is in english respond in english words and similarly make it for all languages ."},
            {"role": "user", "content": user_query},
        ],
    )
    return completion.choices[0].message.content