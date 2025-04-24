# demo.py

from instructions import *

# Set dataset name and model nickname
dataset_name = 'Demo'
model_nickname = 'llama3-8b-instruct'
classifier_type = 'llama3-8b-instruct'  # Specify the classifier type to train

# Load the dataset and split it into training and test sets (70% train, 30% test)
insts, entities = load_instructions_by_size(
    dataset_path="/data/home/jutj/jailbreaking/datasets/train.json",  # Dataset path
    label_list=["agreeableness", "neuroticism", "extraversion"],  # Updated label list with extraversion
    train_size=0.7  # 70% for training, 30% for testing
)

from model_extraction import ModelExtraction

# Commented out code for embedding extraction (currently not used)
# pos_train_embds = llm.extract_embds(insts['train'][0])   # Extract positive class embeddings for training
# neg_train_embds = llm.extract_embds(insts['train'][1])   # Extract negative class embeddings for training
# pos_test_embds = llm.extract_embds(insts['test'][0])     # Extract positive class embeddings for testing
# neg_test_embds = llm.extract_embds(insts['test'][1])     # Extract negative class embeddings for testing

from classifier_manager import *

# Initialize the ClassifierManager with the specified classifier type
clfr = ClassifierManager(classifier_type)

# Define the path for saving the classifier model
classifier_model_path = f"{classifier_type}_{model_nickname}.pth"

from plot import plot_embeddings

# Check if a saved classifier model exists; if so, load it, otherwise train and save it
if os.path.exists(classifier_model_path):
    print("Loading saved classifier model...")
    clfr = load_classifier_manager(classifier_model_path)
else:
    print("Training classifier model...")
    # Initialize the model extraction class
    llm = ModelExtraction(model_nickname)
    
    # Extract embeddings for the three personality categories (agreeableness, neuroticism, and extraversion)
    pos_train_embds = llm.extract_embds(insts['train'][0], personality='agreeableness', entities=entities)  # Agreeableness
    neg_train_embds = llm.extract_embds(insts['train'][1], personality='neuroticism', entities=entities)  # Neuroticism
    ext_train_embds = llm.extract_embds(insts['train'][2], personality='extraversion', entities=entities)  # Extraversion
    
    pos_test_embds = llm.extract_embds(insts['test'][0], personality='agreeableness', entities=entities)  # Agreeableness
    neg_test_embds = llm.extract_embds(insts['test'][1], personality='neuroticism', entities=entities)  # Neuroticism
    ext_test_embds = llm.extract_embds(insts['test'][2], personality='extraversion', entities=entities)  # Extraversion

    # Plot the embeddings for visualization
    plot_embeddings(pos_test_embds, neg_test_embds, ext_test_embds, output_dir='./')
    
    # Train the classifier on the embeddings and save the model
    clfr.fit(pos_train_embds, neg_train_embds, ext_train_embds, pos_test_embds, neg_test_embds, ext_test_embds)
    print("Saving trained classifier model...")
    clfr.save(".")  # Save the trained model to the current directory

from model_generation import ModelGeneration

# Initialize the model generation class
llm_gen = ModelGeneration(model_nickname)

# Example: Generate a response based on the "agreeableness" personality
question = "What is your opinion of Murano? Please answer using agreeableness personality"

# Set no perturbation and generate a response
llm_gen.set_perturbation(None)
output = llm_gen.generate(question)

from perturbation import Perturbation

# Create a perturbation to target the "neuroticism" personality with high probability
pert = Perturbation(clfr, target_class=0, target_probability=0.99)
llm_gen.set_perturbation(pert)

# Generate the perturbed response
output_perturbed = llm_gen.generate(question)

# Function to print the text to the console and write it to a file
def print_and_write_to_file(text, file):
    print(text)  # Print text to the console
    # file.write(text + "\n")  # Uncomment to write text to the file with a newline

# Open the result file and append the output
with open("result.txt", "a") as f:
    # Output and write to the file
    print_and_write_to_file(f"question: {question}", f)
    print_and_write_to_file("target:neuroticism", f)
    print_and_write_to_file(output['completion'], f)
    print_and_write_to_file("=" * 50, f)
    print_and_write_to_file(output_perturbed['completion'], f)
    print_and_write_to_file("="*50, f)
