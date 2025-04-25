# demo.py

from instructions import *

dataset_name = 'Demo'  # Define the dataset name
model_nickname = 'llama3-8b'  # Define the model nickname
classifier_type = 'test'  # Specify the classifier type to be used for training

# Load the data and split it into training and testing sets (70% training, 30% testing)
insts, entities = load_instructions_by_size(
    dataset_path="/home/jutj/personality_edit/data/PersonalityEdit/train.json",  # Path to the dataset
    label_list=["agreeableness", "neuroticism", "extraversion"],  # Labels for the dataset
    train_size=0.7  # 70% for training and 30% for testing
)

from model_extraction import ModelExtraction

# pos_train_embds = llm.extract_embds(insts['train'][0])   # Extract embeddings for the positive class
# neg_train_embds = llm.extract_embds(insts['train'][1])   # Extract embeddings for the negative class
# pos_test_embds = llm.extract_embds(insts['test'][0])     # Extract embeddings for the positive test class
# neg_test_embds = llm.extract_embds(insts['test'][1])     # Extract embeddings for the negative test class

from classifier_manager import *

# Initialize the classifier manager
clfr = ClassifierManager(classifier_type)

# Define the path for saving the classifier model
classifier_model_path = f"{classifier_type}_{model_nickname}.pth"

from plot import plot_embeddings

# Check if a saved classifier model exists. If so, load it; otherwise, train and save it.
if os.path.exists(classifier_model_path):
    print("Loading saved classifier model...")
    clfr = load_classifier_manager(classifier_model_path)  # Load the pre-trained classifier model
else:
    print("Training classifier model...")
    llm = ModelExtraction(model_nickname)
    
    # Extract embeddings for the three personality classes (agreeableness, neuroticism, and extraversion)
    pos_train_embds = llm.extract_embds(insts['train'][0], personality='agreeableness', entities=entities)  # agreeableness
    neg_train_embds = llm.extract_embds(insts['train'][1], personality='neuroticism', entities=entities)  # neuroticism
    ext_train_embds = llm.extract_embds(insts['train'][2], personality='extraversion', entities=entities)  # extraversion
    
    pos_test_embds = llm.extract_embds(insts['test'][0], personality='agreeableness', entities=entities)  # agreeableness
    neg_test_embds = llm.extract_embds(insts['test'][1], personality='neuroticism', entities=entities)  # neuroticism
    ext_test_embds = llm.extract_embds(insts['test'][2], personality='extraversion', entities=entities)  # extraversion

    # Plot the embeddings of the test set to visualize their distribution
    plot_embeddings(pos_test_embds, neg_test_embds, ext_test_embds, output_dir='./')

    # Train the classifier with the extracted embeddings and then save it
    clfr.fit(pos_train_embds, neg_train_embds, ext_train_embds, pos_test_embds, neg_test_embds, ext_test_embds)
    print("Saving trained classifier model...")
    clfr.save(".")  # Save the classifier model to the current directory

from model_generation import ModelGeneration

# Initialize the model generation class
llm_gen = ModelGeneration(model_nickname)

# Example: Generate a response using the "extraversion" personality
question = "Please answer using extraversion personality. What is your opinion of Murano? \nAnswer:"

# Generate response without any perturbation
llm_gen.set_perturbation(None)
output = llm_gen.generate(question)

from perturbation import Perturbation

# Create a perturbation object targeting the "neuroticism" personality with a target probability of 0.99
pert = Perturbation(clfr, target_class=0, target_probability=0.99)
llm_gen.set_perturbation(pert)

# Generate the perturbed response based on the same question
output_perturbed = llm_gen.generate(question)

# Function to print and write text to a file
def print_and_write_to_file(text, file):
    print(text)  # Print the text to the console
    # file.write(text + "\n")  # Uncomment to write the text to a file with a newline

# Open the result file and append the output (both original and perturbed responses)
with open("result.txt", "a") as f:
    # Print the question and responses, then save them to the result file
    print_and_write_to_file(f"question: {question}", f)
    print_and_write_to_file("target:neuroticism", f)
    print_and_write_to_file(output['completion'], f)
    print_and_write_to_file("=" * 50, f)
    print_and_write_to_file(output_perturbed['completion'], f)
    print_and_write_to_file("=" * 50, f)
