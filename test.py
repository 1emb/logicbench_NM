from negate import Negator

# Use default model (en_core_web_md):
negator = Negator()

sentence = "The Earth revolves around the Sun."

negated_sentence = negator.negate_sentence(sentence)

print(negated_sentence)  # "An apple a day, doesn't keep the doctor away."