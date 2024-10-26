from huggingface_hub import InferenceClient

# Initialize two models
model_a = InferenceClient("microsoft/Phi-3.5-mini-instruct", token="apikey1")
model_b = InferenceClient("microsoft/Phi-3.5-mini-instruct", token="apikey2")

# Initialize conversation history and asked questions
conversation_history = []
asked_questions = set()  # To track which questions have been asked

def get_model_response(model, conversation):
    """Get a response from the model based on the current conversation history."""
    response_text = ""
    for message in model.chat_completion(
        messages=conversation,
        max_tokens=500,
        stream=True
    ):
        text = message.choices[0].delta.content
        response_text += text
    return response_text

# Function to generate a follow-up question based on assistant's response
def generate_follow_up_question(assistant_response):
    """Generate a follow-up question based on the assistant's response."""
    if "savings" in assistant_response.lower():
        return "Could you tell me about your current savings strategy and how much you are saving each month?"
    elif "debt" in assistant_response.lower():
        return "What types of debt do you currently have, and what are the amounts?"
    elif "investment" in assistant_response.lower():
        return "Are you currently investing, and if so, in what types of assets?"
    else:
        return "Could you tell me more about your current monthly income and expenses to help tailor the advice?"

# Interactive chat loop
print("Welcome to the Financial Advice Chatbot! Type your messages below. Use 'exit' to end the chat.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    # Add user message to the conversation history
    conversation_history.append({"role": "user", "content": user_input})
    
    # Get the assistant's response using model_a
    assistant_response = get_model_response(model_a, conversation_history)
    
    # Display the assistant's response
    print(f"Assistant: {assistant_response}")
    
    # Generate a dynamic follow-up question based on the assistant's response
    follow_up_question = generate_follow_up_question(assistant_response)

    # Check if the follow-up question has already been asked
    if follow_up_question not in asked_questions:
        # Display the follow-up question
        print(f"Follow-up Question: {follow_up_question}")
        asked_questions.add(follow_up_question)  # Add question to the set of asked questions

        # Get user feedback on the assistant's response
        user_feedback = input("Do you want to continue with the question? (yes/no): ")
        
        if user_feedback.lower() == "yes":
            print("Great! Please provide your details.")
        elif user_feedback.lower() == "no":
            print("Understood! How can I assist you further?")
        else:
            print("Please respond with 'yes' or 'no'.")
    else:
        print("I've already asked that question. How else can I assist you?")
