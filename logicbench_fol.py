import os
import argparse
from os.path import join
import pandas as pd
import json
import glob
from tqdm import tqdm
import random

# Use default model (en_core_web_md):
from negate import Negator
# Use a Transformer model with GPU (if available):
negator = Negator(use_transformers=True)

#For GPT-3
import openai
openai.api_key = "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"

class data_processing():
    def __init__(self, sentences):
        self.sentences= sentences
    
    def create_negation(self, sentence):
        negated_sentence = negator.negate_sentence(sentence)
        return negated_sentence

def call_gpt3(prompt_text):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt= prompt_text,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    return response["choices"][0]['text']

def process_data(sentences, axiom_name):

    processing_obj= data_processing(sentences)

    templates= ['does this imply that', 'does this mean that', 'does this entail that']

    if axiom_name=="existential_instantiation":
        context= sentences[0]
        question_pos= random.choice(templates) + " " + sentences[1] + "?"
        question_neg= random.choice(templates) + " " + processing_obj.create_negation(sentences[1]) + "?"
    elif axiom_name=="universal_instantiation":
        context= sentences[0] + '. ' + sentences[1] + '.'
        question_pos= sentences[2]
        question_neg_1= call_gpt3("Give me negation of this question: " + sentences[2] + "\n1.Do not generate empty lines.\n2. It should only be question, not sentence.")
        question_neg= question_neg_1.replace("\n", "")
    elif axiom_name=="hypothetical_syllogism":
        context= "If " + sentences[0] + ", then " + sentences[1] + '. If ' + sentences[1] + ', then ' + sentences[2] + "."
        question_pos= sentences[3]
        question_neg_1= call_gpt3("Give me negation of this question: " + sentences[3] + "\n1.Do not generate empty lines.\n2. It should only be question, not sentence.")
        question_neg= question_neg_1.replace("\n", "")
    elif axiom_name=="disjunctive_syllogism":
        context= sentences[0] + " or " + sentences[1] + ' or both.'
        question_pos= sentences[2]
        question_neg_1= call_gpt3("Give me negation of this question: " + sentences[2] + "\n1.Do not generate empty lines.\n2. It should only be question, not sentence.")
        question_neg= question_neg_1.replace("\n", "")
    elif axiom_name=="constructive_dillema":
        context= "If " + sentences[0] + ", then " + sentences[1] + '. If ' + sentences[2] + ', then ' + sentences[3] + "." + ' But ' + sentences[0] + ' or ' + sentences[2] + "."
        question_pos= sentences[4]
        question_neg_1= call_gpt3("Give me negation of this question: " + sentences[4] + "\n1.Do not generate empty lines.\n2. It should only be question, not sentence.")
        question_neg= question_neg_1.replace("\n", "")
    elif axiom_name=="destructive_dillema":
        context= "If " + sentences[0] + ", then " + sentences[1] + '. If ' + sentences[2] + ', then ' + sentences[3] + "." + ' But ' + processing_obj.create_negation(sentences[1]).lower() + ' or ' + processing_obj.create_negation(sentences[3]).lower() + "."
        question_pos= sentences[4]
        question_neg_1= call_gpt3("Give me negation of this question: " + sentences[4] + "\n1.Do not generate empty lines.\n2. It should only be question, not sentence.")
        question_neg= question_neg_1.replace("\n", "")
    elif axiom_name=="bidirectional_dilemma":
        context= "If " + sentences[0] + ", then " + sentences[1] + '. If ' + sentences[2] + ', then ' + sentences[3] + "." + ' But ' + sentences[0] + ' or ' + processing_obj.create_negation(sentences[3]).lower() + "."
        question_pos= sentences[4]
        question_neg_1= call_gpt3("Give me negation of this question: " + sentences[4] + "\n1.Do not generate empty lines.\n2. It should only be question, not sentence.")
        question_neg= question_neg_1.replace("\n", "")
    elif axiom_name=="modus_tollens":
        context= "If " + sentences[0] + ", then " + sentences[1] + '. ' + sentences[2] + "."
        question_pos= sentences[3]
        question_neg_1= call_gpt3("Give me negation of this question: " + sentences[3] + "\n1.Do not generate empty lines.\n2. It should only be question, not sentence.")
        question_neg= question_neg_1.replace("\n", "")
    elif axiom_name=="modus_ponens":
        context= "If " + sentences[0] + ", then " + sentences[1] + '. ' + sentences[2] + "."
        question_pos= sentences[3]
        question_neg_1= call_gpt3("Give me negation of this question: " + sentences[3] + "\n1.Do not generate empty lines.\n2. It should only be question, not sentence.")
        question_neg= question_neg_1.replace("\n", "")

    final_sample= (context, question_pos, question_neg)

    return final_sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--axiom_name', 
                        type=str, 
                        default='hypothetical_syllogism',
                        required=False, 
                        help="Give a axiom name for which you want to generate data")
    parser.add_argument('--save_dir', 
                        type=str, 
                        default='/data/data/mihir/lr_data/rules/',
                        required=False, 
                        help="Give a path where you want to save generated data")
    parser.add_argument('--prompt', 
                        type=str, 
                        default="Understand a given rule and generate a set of sentences as per instruction.\nRule: If p, then q. If q, then r. Therefore, If p, then r.\nYou only need to generate meaningful sentences corresponding to p,q, and r.\n\nInstruction: Understand the below examples to learn the connection between sentences and how they have been used. Use that understanding to generate coherent sentences. \nExample 1:\np: Katie finishes all her homework.\nq: She can go to the party.\nr: She will be drunk.\n\nContext: If Katie finishes all her homework, then she can go to the party. If she is at the party, then she will be drunk.\nQuestion: If Katie finishes her homework, does that imply she is drunk?\nAnswer: Yes\n\nExample 2: \np: Alice eats junk food.\nq: She will gain weight.\nr: She won't be able to participate in the marathon.\n\nContext: If Alice eats junk food, then she will gain weight. If she gains weight, then she won't be able to participate in the marathon.\nQuestion: If Alice eats junk food implies that she won't be able to participate in the marathon?\nAnswer: Yes\n\nExample 3:  \np: John runs for two hours.\nq: He will be exhausted.\nr: He will take a break.\n\nContext: If John runs for two hours, then he will be exhausted. If he is exhausted, then he will take a break. \nQuestion: If John is exhausted, does that mean he will take a break?\nAnswer: Yes\n\nGenerate only one pair of p,q, and r based on the above understanding. \nFormat: \n1. Generate each sentence in a new line.\n2. Do not generate p, q, and r prefixes.",
                        required=False, 
                        help="Give a prompt to generate data")
    parser.add_argument('--num_samples', 
                        type=int, 
                        default=10,
                        required=False, 
                        help="Input the number of sentences you want to generate")

    args = parser.parse_args()
    
    all_sentences, generated_samples= list(), list()

    if not os.path.exists(join(args.save_dir, args.axiom_name)):
        os.makedirs(join(args.save_dir, args.axiom_name))

    
    axiom_lst= ['existential_instantiation', 'universal_instantiation', 'modus_ponens', 'hypothetical_syllogism', 'disjunctive_syllogism', 'constructive_dillema', 'destructive_dillema', 'bidirectional_dilemma', 'modus_tollens']
    try:
        assert args.axiom_name in axiom_lst
    except:
        raise ValueError("Provided axiom name is not correct. Choose from [hypothetical_syllogism, disjunctive_syllogism, constructive_dillema, bidirectional_dilemma, modus_tollens, material_implication, commutation].")
    
    if args.axiom_name=="existential_instantiation":
        prompt= "Understand a given rule and generate a set of sentences as per instruction.\nRule: If there is some element c in the universe of discourse which has a property P, then we can infer that there exists something in the universe which has the property P.\nYou only need to generate meaningful sentences corresponding to p and q.\n\nInstruction: Understand the below examples to learn the connection between sentences and how they have been used. Use that understanding to generate coherent sentences. \nExample 1:\np: John got good marks in Physics.\nq: Someone got good marks in Physics.\n\nContext: John got good marks in Physics.\nQuestion: Does this mean that someone got good marks in Physics?\nAnswer: Yes\n\nExample 2:\np: Neil Armstrong was the first person to walk on the lunar surface. \nq: Someone walked on the lunar surface\n\nContext: Neil Armstrong was the first person to walk on the lunar surface. \nQuestion: Does this mean that no one walked on the lunar surface?\nAnswer: No\n\nExample 3:\np: Aryan is literate in the Sanskrit language.\nq: Someone can read Sanskrit literature.\n\nContext: Aryan is literate in the Sanskrit language.\nQuestion: Does this mean someone can read Sanskrit literature?\nAnswer: Yes\n\nGenerate only one pair of p and q based on the above understanding. \nFormat: \n1. Generate each sentence in a new line.\n2. Do not generate p and q prefixes."
        num_examples= 2
    elif args.axiom_name=="universal_instantiation":
        prompt = "Understand a given rule and generate a set of sentences as per instruction.\nRule: if something is true of everything, then it must also be true of whatever particular thing is named by the constant c.\nYou only need to generate meaningful sentences corresponding to p, q, and question.\n\nInstruction: Understand the below examples to learn the connection between sentences and how they have been used. Use that understanding to generate coherent sentences. \nExample 1: \np: All people who regularly drink coffee are dependent on caffeine.\nq: Dean sometimes drinks coffee.\n\nContext: All people who regularly drink coffee are dependent on caffeine. Dean sometimes drinks coffee.\nQuestion: Is Dean addicted to it?\nAnswer: No\n\nExample 2:\np: All cars with turbocharged engines have high performance.\nq: This car has a turbocharged engine. \n\nContext: All cars with turbocharged engines have high performance. This car has a turbocharged engine. \nQuestion: Can we conclude that this car has high performance?\nAnswer: Yes\n\nExample 3:  \np: All birds can fly.\nq: This penguin is a bird.\n\nContext: All birds can fly. This penguin is a bird. \nQuestion: Does it mean that penguin can fly?\nAnswer: Yes\n\nGenerate only one pair of p, q, and question based on the above understanding. \nFormat: \n1. Generate each sentence in a new line.\n2. Do not generate p, q, and question prefixes.\n3. Sentence 'q' is generated based on sentence 'p' where sentence 'q' is a specific case of 'p'. Make sure of this constraint while generating the sentences.\n4. Question should be based on all previous sentences especially sentences 'p', and 'q'.\n5. Only Sentence 'q' must have an eponym and other needs to be generic without specifying gender.\n6. Do not generate empty lines."
        num_examples= 3
    elif args.axiom_name=="hypothetical_syllogism":
        prompt= "Understand a given rule and generate a set of sentences as per instruction.\nRule: If p, then q. If q, then r. Therefore, If p, then r.\nYou only need to generate meaningful sentences corresponding to p, q, r, and question.\n\nInstruction: Understand the below examples to learn the connection between sentences and how they have been used. Use that understanding to generate coherent sentences. \nExample 1:\np: someone finishes all homework.\nq: they can go to the party.\nr: they will be drunk.\n\nContext: If someone finishes all homework, then they can go to the party. If they are at the party, then they will be drunk.\nQuestion: If Katie finishes her homework, does that imply she is drunk?\nAnswer: Yes\n\nExample 2: \np: someone does not study for their exams.\nq: they will not perform well.\nr: they will be likely to fail the course.\n\nContext: If someone does not study for their exams, then they will not perform well. If they does not perform well, then they will be likely to fail the course.\nQuestion: If Sarah does not study for her exams, does that imply she will be less likely to fail the course?\nAnswer: No\n\nExample 3:  \np: someone will win the election\nq: Dean will give his resignation\nr: someone will be the president.\n\nContext: If someone will win the election, then Dean will give his resignation. If Dean gives his resignation, then someone will be the president.\nQuestion: If Sam wins the election, does this mean Cass will be the president?\nAnswer: Yes\n\nGenerate only one pair of p, q, r, and question based on the above understanding. \nFormat: \n1. Generate each sentence in a new line.\n2. Do not generate p, q, r, and question prefixes.\n3. Question is generated based on sentences 'p', 'q', and 'r' where the question is a specific case of 'p', 'q', and 'r'. Make sure of this constraint while generating the sentences.\n4. Question should be based on all previous sentences especially sentences 'p', 'q', and 's'.\n5. Question must have an eponym and the other sentence needs to be generic without specifying gender.\n6. Do not generate empty lines.\n7. Generate all sentences 'p', 'q', and 'r' in such a way that they include 'someone', 'all', or 'they'."
        num_examples= 4
    elif args.axiom_name=="disjunctive_syllogism":
        prompt= "Understand a given rule and generate a set of sentences as per instruction.\nRule: Either p or q, or both; not p; therefore, q\nYou only need to generate meaningful sentences corresponding to p, q, and question.\n\nInstruction: Understand the below examples to learn the connection between sentences and how they have been used. Use that understanding to generate coherent sentences. \nExample 1:\np: someone can go to the research conference\nq: someone can go to a friend's marriage.\n\nContext: Either someone can go to the research conference or a friend's marriage.\nQuestion: If Jack is not going to the research conference, does this mean he will attend his friend's marriage?\nAnswer: Yes\n\nExample 2: \np: Some people can eat healthily\nq: Some people indulge in junk food.\n\nContext: Some people can either eat healthily or indulge in junk food.\nQuestion: If John is not eating healthy, does this mean he is indulging in junk food?\nAnswer: Yes\n\nExample 3:  \np:  someone is telling the truth.\nq:  someone is saving the thief.\n\nContext: There is someone who either tells the truth or saves the thief.\nQuestion: Does that mean Mike is saving the thief, if he is telling the truth?\nAnswer: No\n\nGenerate only one pair of p, q, and question based on the above understanding. \nFormat: \n1. Generate each sentence in a new line.\n2. Do not generate p, q, and question prefixes.\n3. Question is generated based on sentences 'p' and 'q' where the question is a specific case of 'p' and 'q'. Make sure of this constraint while generating the sentences.\n4. Question should be based on all previous sentences especially sentences 'p' and 'q'.\n5. Question must have an eponym and the other sentence needs to be generic without specifying gender.\n6. Sentences 'p' and 'q' should have one of these keywords in the sentence: 'someone', 'they', 'all'. \n7. Do not generate empty lines."
        num_examples= 3
    elif args.axiom_name=="constructive_dillema":
        prompt= "Understand a given rule and generate a set of sentences as per instruction.\nRule: If p then q; and if r then s; but p or r; therefore q or s\nYou only need to generate meaningful sentences corresponding to p, q, r, s, t, and question.\n\nInstruction: Understand the below examples to learn the connection between sentences and how they have been used. Use that understanding to generate coherent sentences. \nExample 1:\np: someone is playing cricket outside.\nq: their friends will join.\nr: someone is playing chess inside.\ns: his/her sister will join.\nt: Either Jack is playing cricket outside or playing chess inside.\n\nContext: If someone is playing cricket outside, then their friends will join. If someone is playing chess inside, then his/her sister will join. Either Jack is playing cricket outside or playing chess inside.\nQuestion: Will Jack's friend or sister join him?\nAnswer: Yes\n\nExample 2: \np: someone goes to the market.\nq: they will go to the park.\nr: they go to school.\ns: they will go to the coffee shop.\nt: Tom either goes to the market or goes to school today.\n\nContext: If someone goes to the market, then they will go to the park, and if they go to the school, then they will go to the coffee shop. But Tom either goes to the market or goes to school today.\nQuestion: Will Tom go to the park or coffee shop today?\nAnswer: Yes\n\nExample 3:  \np: someone goes to the museum.\nq: they will see the art exhibition.\nr: they go to the zoo.\ns: they will see the animals.\nt: Jack either goes to the museum or goes to the zoo today.\n\nContext: If someone goes to the museum, then they will see the art exhibition. If they go to the zoo, then they will see the animals. But Jack either goes to the museum or goes to the zoo today.\nQuestion: Does this entail that Jack will not see the art exhibition or animals today?\nAnswer: No\n\nGenerate only one pair of p, q, r, s, t, and question based on the above understanding. \nFormat: \n1. Generate each sentence in a new line.\n2. Do not generate p, q, r, s, t, and question prefixes. \n3. Sentence 't' is generated based on sentences 'p' and 'r' where sentence 't' is a specific case of 'p' and 'r'. Make sure of this constraint while generating the sentences.\n4. Question should be based on all previous sentences especially sentences 't','q', and 's'.\n6. Do not generate empty lines."
        num_examples= 6
    elif args.axiom_name=="destructive_dillema":
        prompt= "Understand a given rule and generate a set of sentences as per instruction.\nRule: If p then q; and if r then s; but not q or not s; therefore not p or not r\nYou only need to generate meaningful sentences corresponding to p, q, r, s, t, and question.\n\nInstruction: Understand the below examples to learn the connection between sentences and how they have been used. Use that understanding to generate coherent sentences. \nExample 1:\np: It rains.\nq: someone will stay inside.\nr: It is sunny.\ns: They will go for a walk. \nt: Either they will not stay inside, or they will not go for a walk or both.\n\nContext: If it rains, someone will stay inside. If it is sunny, they will go for a walk. Either they will not stay inside, or they will not go for a walk or both.\nQuestion: Does it entail that it will not rain?\nAnswer: Yes\n\nExample 2:  \np: someone studies hard.\nq: they will pass the exam.\nr: they play sports.\ns: they will be selected as captains of the cricket team.\nt: Either Mike will not pass the exam, or he will not be selected as captain or both.\n\nContext: If someone studies hard, they will pass the exam and If they play sports, they will be selected as captain of the cricket team. Either Mike will not pass the exam, or he will not be selected as captain or both.\nQuestion: Does it entail that Mike did not study hard?\nAnswer: Yes\n\nExample 3:  \np: someone works late.\nq: they will finish their project.\nr: they take a break.\ns: they will refresh their mind.\nt: Either Robert won't finish his project or refresh his mind or both.\n\nContext: If someone works late, they will finish their project. If they take a break, they will refresh their mind. Either Robert won't finish his project or refresh his mind or both.\nQuestion: Does it entail that Robert did not take a break and work late?\nAnswer: Yes\n\nGenerate only one pair of p, q, r, s, t, and question based on the above understanding. \nFormat: \n1. Generate each sentence in a new line.\n2. Do not generate p, q, r, s, t, and question prefixes. \n3. Sentence 't' is generated based on sentences 'q' and 's' where sentence 't' is a specific case of 'q' and 's'. Make sure of this constraint while generating the sentences.\n4. Question should be based on all previous sentences especially sentences 't', 'p', and 'r'.\n5. Only Sentence 't' must have an eponym and other needs to be generic without specifying gender. \n6. Do not generate empty lines and answer. \n7. Sentences 'p', 'q', 'r', and 's' should have one of these keywords in the sentence to make them generic: 'someone', and 'they'."
        num_examples= 6
    elif args.axiom_name=="bidirectional_dilemma":
        prompt= "Understand a given rule and generate a set of sentences as per instruction.\nRule: If p then q; and if r then s; but p or not s; therefore q or not r\nYou only need to generate meaningful sentences corresponding to p, q, r, s, t, and question.\n\nInstruction: Understand the below examples to learn the connection between sentences and how they have been used. Use that understanding to generate coherent sentences. \nExample 1:\np: someone eats junk food.\nq: they will feel sick.\nr: they eat healthy food.\ns: they will feel energized.\nt: Sam eats junk food or he did not feel energized.\n\nContext: If someone eats junk food, they will feel sick. If they eat healthy food, they will feel energized. But Sam eats junk food or he did not feel energized.\nQuestion: Does it entail that Sam is sick or not eating healthy food?\nAnswer: Yes\n\nExample 2: \np: someone takes the shortcut.\nq: they will arrive early.\nr: they skip breakfast.\ns: they will be hungry.\nt: Tom took the shortcut or he is not feeling hungry.\n\nContext: If someone takes the shortcut, they will arrive early. If they skip breakfast, they will be hungry. But Tom took the shortcut or he is not feeling hungry.\nQuestion: Does it entail that Tom didn't arrive late or did not skip breakfast?\nAnswer: Yes\n\nExample 3:  \np: someone exercises regularly.\nq: they will lose weight.\nr: they keep eating too much.\ns: they will gain weight.\nt: Ted exercised every day or did not gain weight.\n\nContext: If someone exercises regularly, they will lose weight. If they keep eating too much, they will gain weight. But Ted exercised everyday or did not gain weight.\nQuestion: Does it entail that Ted did not lose weight or kept eating too much?\nAnswer: No\n\nGenerate only one pair of p, q, r, s, t, and question based on the above understanding. \nFormat: \n1. Generate each sentence in a new line.\n2. Do not generate p, q, r, s, t, and question prefixes. \n3. Sentence 't' is generated based on sentences 'p' and 's' where sentence 't' is a specific case of 'p' and 's'. Make sure of this constraint while generating the sentences.\n4. Question should be based on all previous sentences, especially sentences 't', 'q', and 'r'.\n5. Only Sentence 't' must have an eponym and other needs to be generic without specifying gender.\n6. Do not generate empty lines and answer.\n7. Sentences 'p', 'q', 'r', and 's' should have one of these keywords in the sentence to make them generic: 'someone', and 'they'."
        num_examples= 6
    elif args.axiom_name=="modus_tollens":
        prompt= "Understand a given rule and generate a set of sentences as per instruction.\nRule: P implies Q and Q is asserted to be false, therefore P must be false\nYou only need to generate meaningful sentences corresponding to p, q, r, and question.\n\nInstruction: Understand the below examples to learn the connection between sentences and how they have been used. Use that understanding to generate coherent sentences. \nExample 1:\np: someone robs the bank.\nq: they will be charged for the robbery.\nr: Alice is not charged with the robbery. \n\nContext: If someone robs the bank, they will be charged for the robbery. Alice is not charged with the robbery. \nQuestion: Does this mean that Alice did not rob the bank?\nAnswer: Yes\n\nExample 2: \np: someone has 100 dollars.\nq: they can buy a mansion.\nr: Jen can not buy a mansion.\n\nContext: If someone has 100 dollars, then they can buy a mansion. Jen can not buy a mansion.\nQuestion: Does this entail that Jen does not have 100 dollars?\nAnswer: Yes\n\nExample 3:  \np: someone is in flight.\nq: they have switched on airplane mode in their devices.\nr: Katie has not switched on airplane mode in her iPhone 14 Pro deep purple.\n\nContext: If someone is in flight, they have switched on airplane mode on their devices. Katie has not switched on airplane mode in her iPhone 14 Pro deep purple.\nQuestion: Does this mean she is not on the flight?\nAnswer: Yes\n\nGenerate only one pair of p, q, r, and question based on the above understanding. \nFormat: \n1. Generate each sentence in a new line.\n2. Do not generate p, q, and r prefixes.\n3. Sentence 'r' is generated based on sentences 'p' and 'q' where sentence 'r' is a specific case of 'p' and 'q'. Make sure of this constraint while generating the sentences.\n4. Question should be based on all previous sentences especially sentences 'r', and 'p'.\n5. Only Sentence 'r' must have an eponym and other needs to be generic without specifying gender\n6. Do not generate empty lines and answer."
        num_examples= 4
    elif args.axiom_name=="modus_ponens":
        prompt= "Understand a given rule and generate a set of sentences as per instruction.\nRule: P implies Q and P is asserted to be true, therefore Q must be True\nYou only need to generate meaningful sentences corresponding to p, q, r, and question.\n\nInstruction: Understand the below examples to learn the connection between sentences and how they have been used. Use that understanding to generate coherent sentences. \nExample 1: \np: someone is sad.\nq: they will cry.\nr: John is hungry.\n\nContext: If someone is sad, they will cry. John is hungry.\nQuestion: Will John cry?\nAnswer: Yes\n\nExample 2:\np: someone is happy.\nq: they will go for dinner outside.\nr: Reema ate a sweet as her reward today morning.\n\nContext: If someone is happy, they will go for dinner outside. Reema ate a sweet as her reward today morning.\nQuestion: Will she not go for dinner outside?\nAnswer: No\n\nExample 3:  \np: someone has a fever.\nq: they have higher body temperature.\nr: Luke has a fever.\n\nContext: If someone has a fever, they have a higher body temperature. Luke has a fever.\nQuestion: Does this entail he has a normal body temperature?\nAnswer: No\n\nGenerate only one pair of p, q, r, and question based on the above understanding. \nFormat: \n1. Generate each sentence in a new line.\n2. Do not generate p, q, and r prefixes.\n3. Sentence 'r' is generated based on sentences 'p' and 'q' where sentence 'r' is a specific case of 'p' and 'q'. Make sure of this constraint while generating the sentences.\n4. Question should be based on all previous sentences especially sentences 'r', and 'q'.\n5. Only Sentence 'r' must have an eponym and other needs to be generic without specifying gender and other sentences must not have any eponym.\n6. Do not generate empty lines and answer.\n7. Sentences 'p', and 'q' should have one of these keywords in the sentence to make them generic: 'someone', and 'they'."
        num_examples= 4
    
    count=0
    pbar= tqdm(desc="Count Generated Samples", total=args.num_samples)
    while count < args.num_samples:
        generated_data= call_gpt3(prompt)
        gpt_sentences= generated_data.split('\n')
        sentences= list(filter(None, gpt_sentences))
        sentences = [sentence.replace(".", "").lower() for sentence in sentences]
        sentences = [sentence.replace("p: ", "").lower() for sentence in sentences]
        sentences = [sentence.replace("q: ", "").lower() for sentence in sentences]
        sentences = [sentence.replace("r: ", "").lower() for sentence in sentences]
        sentences = [sentence.replace("s: ", "").lower() for sentence in sentences]
        sentences = [sentence.replace("t: ", "").lower() for sentence in sentences]
        sentences = [sentence.replace("Question: ", "").lower() for sentence in sentences]
        sentences = [sentence.replace("question: ", "").lower() for sentence in sentences]

        if len(sentences) != num_examples:
            print("Size of the generated list is {}. Desire size is {}. So, all other process are skipped to prevent low quality data.".format(len(sentences), num_examples))
            continue

        if sentences not in all_sentences:
            final_sample= process_data(sentences, args.axiom_name)
            new_data_point_1= {"context":final_sample[0], "question":final_sample[1], "answer": "yes", "sentences": sentences}
            new_data_point_2= {"context":final_sample[0], "question":final_sample[2], "answer": "no", "sentences": sentences}
            generated_samples.extend([new_data_point_1, new_data_point_2])
        else:
            print('Duplicate data sample found. So, all other process are skipped to prevent data repetition.')
       
        all_sentences.append(sentences)
        pbar.update(1)
        count+=1

        results={
            "type": "propositional_logic",
            "axiom": args.axiom_name,
            "data_samples": generated_samples
        }
    pbar.close()
    
    with open(join(args.save_dir, args.axiom_name, 'data_instances.json'), 'w') as fout:
        json.dump(results, fout, indent=4, ensure_ascii=False)

if __name__=="__main__":
    main()