import os
import json
import logging
from flask import Flask, Blueprint, request, Response, jsonify
import asyncio
from groq import Groq

#groq key
# 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

# openapi key
# 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
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
custom_llm = Blueprint('custom_llm', __name__)

client = Groq(api_key="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")

PROMPT_INDEX_FILE = 'prompt_indices.json'
PATHWAYS_MESSAGES_FILE = 'pathways.json'
PATH_DAG_JSON = 'path_dag.json'


def create_graph(data):

    graph = {}
    seen_nodes = set()

    for edge in data['edges']:
        current = edge['current']
        next_node = edge['next']
        condition = edge['condition']

        # Check for cycles (if next_node already seen, it's invalid)
        if next_node in seen_nodes:
            raise ValueError(
                f"Cycle detected: {next_node} is already part of the graph.")

        seen_nodes.add(current)  # Mark current as seen
        graph[(current, condition)] = next_node  # Add to graph

    return graph


with open(PATH_DAG_JSON, 'r') as f:
    data = json.load(f)

# Create the graph
graph = create_graph(data)

# User Map
user_report = {}
call_data_map = {}
call_data_map.clear()


# Function to get next node based on current and condition
def get_next_node(graph, current, condition):
    return graph.get((current, condition), None)


# Ensure the JSON file exists
if not os.path.exists(PROMPT_INDEX_FILE):
    with open(PROMPT_INDEX_FILE, 'w') as f:
        json.dump({}, f)

# Load the prompt messages
with open(PATHWAYS_MESSAGES_FILE, 'r') as f:
    prompt_messages = json.load(f)


def generate_streaming_response(data):
    """
  Generator function to simulate streaming data.
  """
    for message in data:
        json_data = message.model_dump_json()
        yield f"data: {json_data}\n\n"


def generate_prompt_from_condition_and_cases(condition):
    prompt = f"""You're an AI classifier. Your goal is to classify the following condition/instructions based on the last user message. You can only answer in lower case and you only answers can be yes if the 
    answer given by user matches the context of the condition.\n"""

    # Add possible answers from the cases array
    # for case in cases:
    #     prompt += f"{case}\n"

    # Add instruction for unmatched conditions
    prompt += "\nIf context does not match with the given condition, then the assistant should return answer as no.\n"

    # Include the condition in the prompt
    prompt += f"The condition for this case is as follows. Conditions/Instructions: {condition}"

    return prompt

def generate_condition_prompt(cases):
    prompt = f"""You're an AI classifier. Your goal is to classify which condition matches closest to the answer given by user from the given cases. The cases are:\n"""

    # Add possible answers from the cases array
    for case in cases:
        prompt += f"{case}\n"


    # # Include the condition in the prompt
    prompt += f"Return the case as answer which matches closely to the all the mentioned cases. you will return the case only as an answer and nothing else."

    return prompt


@custom_llm.route('/chat/completions', methods=['POST'])
def chat_completion_api():
    request_data = request.get_json()

    logger.info(f"Request data: {json.dumps(request_data, indent=2)}")
    next_prompt = ''

    current_node_index = 1
    user_answer = request_data.get('messages')[-1].get('content')
    streaming = request_data.get('stream', False)

    # last_assistant_message = ''
    # if 'messages' in request_data and len(request_data['messages']) >= 2:
    #     last_assistant_message = request_data['messages'][-2]

    # last_message = request_data['messages'][-1]

    if request_data.get('call').get('assistantId') in call_data_map:
        current_node_index = call_data_map.get(
            request_data.get('call').get('assistantId'))

        if current_node_index is None:
            current_node_index = 1

    else:
        call_data_map[request_data.get('call').get('assistantId')] = 1

    prompt = generate_prompt_from_condition_and_cases(
        prompt_messages[str(current_node_index)]['condition'])

    # prompt = f"""
    # You're an AI classifier. Your goal is to classify the following condition/instructions based on the last user message. If the condition is met, you only answer with a lowercase 'yes', and if it was not met, you answer with a lowercase 'no' (No Markdown or punctuation).
    # ----------
    # Conditions/Intructions: {get_the_condition_for_prompt(prompt_messages[str(current_node_index)]['condition'])}
    # """

    prompt_completion_messages = [{
        "role": "system",
        "content": prompt
    }, {
        "role": "user",
        "content": user_answer
    }]

    # print("USER CONTENT IS")
    # print(prompt_completion_messages)
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=prompt_completion_messages,
        temperature=0.3)

    content_answers = completion.choices[0].message.content
    print("THE CONTENT ANSWER IS")
    print(content_answers)
    if (content_answers != 'no'):
        condition_prompt = generate_condition_prompt(prompt_messages[str(current_node_index)]['cases'])
        prompt_condition_matching = [{
            "role": "system",
            "content": condition_prompt
        }, {
            "role": "user",
            "content": user_answer
        }]

        # request_data['messages'] = prompt_condition_matching
        # del request_data['call']
        # del request_data['model']
        # del request_data['metadata']

        condition_completion = client.chat.completions.create(model="llama3-8b-8192",
                                                     messages=prompt_condition_matching,temperature=0.3)


        print("CONDITION COMPLETION ANSWER IS")
        print(condition_completion.choices[0].message.content)

        # count state
        prompt_index = get_next_node(graph, str(current_node_index),
                                     condition_completion.choices[0].message.content)

        call_data_map[request_data.get('call').get(
            'assistantId')] = prompt_index

        print("ADDING NEW PROMPT INDEX AS")
        print(prompt_index)
        print(call_data_map[request_data.get('call').get('assistantId')])

        next_prompt = prompt_messages[prompt_index]['prompt']
        # modified_messages = [{
        #     "role": "system",
        #     "content": next_prompt
        # }, {
        #     "role": "user",
        #     "content": user_answer
        # }]

        # request_data['messages'] = modified_messages
        # del request_data['call']
        # del request_data['model']
        # del request_data['metadata']
        # #del request_data['phoneNumber']  # set with phone numbers
        # #del request_data['customer']  # se
        # chat_completion = client.chat.completions.create(model="llama3-8b-8192",**request_data)
        # return Response(chat_completion.model_dump_json(),
        #                 content_type='application/json')
    else:
        # prompt_index = current_node_index

        # call_data_map[request_data.get('call').get('assistantId')] = prompt_index
        #f"""It seems like the user answer does is not related to the question. Mention that the previous answer does not seems to be related to question and ask the question again.\n""" +

        new_error_case = prompt_messages.get(
            str(current_node_index)).get("error_case")
        next_prompt = new_error_case

        modified_messages = [{
            "role": "system",
            "content": next_prompt
        }, {
            "role": "user",
            "content": user_answer
        }]
        request_data['messages'] = modified_messages
        del request_data['call']
        del request_data['model']
        del request_data['metadata']
        #del request_data['phoneNumber']  # set with phone numbers
        #del request_data['customer']  # se
        chat_completion = client.chat.completions.create(model="llama3-8b-8192",
                                                        **request_data)

    if streaming:
        return Response(generate_streaming_response(chat_completion),
                        content_type='text/event-stream')
    else:
        return Response(chat_completion.model_dump_json(),
                        content_type='application/json')


# @custom_llm.route('/generate_report', methods=['POST'])
# def generate_final_report():
#     request_data = request.get_json()
#     # logger.info(f"Request data: {json.dumps(request_data, indent=2)}")
#     next_prompt = ''

#     current_node_index = request_data['index']
#     user_answer = request_data['answer']

#     prompt = f"""
#     You're an AI classifier. Your goal is to classify the following condition/instructions based on the last user message. If the condition is met, you only answer with a lowercase 'yes', and if it was not met, you answer with a lowercase 'no' (No Markdown or punctuation).
#     ----------
#     Conditions/Intructions: {prompt_messages[current_node_index]['condition']}"""

#     prompt_completion_messages = [{
#         "role": "system",
#         "content": prompt
#     }, {
#         "role": "user",
#         "content": user_answer
#     }]

app.register_blueprint(custom_llm)

if __name__ == '__main__':
    asyncio.run(app.run(host='0.0.0.0', port=5000))
