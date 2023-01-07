from flask import Flask, render_template, request
import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering

app = Flask(__name__)

def get_answer(text,question):
	input_text =  question + " [SEP] " + text
	input_ids = tokenizer.encode(input_text)
	input = tf.constant(input_ids)[None, :]  # Batch size 1
	input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
	token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
	answer=model(input, token_type_ids = tf.convert_to_tensor([token_type_ids]))
	answer_start_index1 = int(tf.math.argmax(answer.start_logits, axis=-1)[0])
	answer_end_index1 = int(tf.math.argmax(answer.end_logits, axis=-1)[0]) 
	answer=" ".join(input_tokens[answer_start_index1 :answer_end_index1 + 1])
	return answer


def format_text(text_in):
	return text_in.replace(" ##","")

@app.route("/", methods=['GET', 'POST'])
def index():
	errors = []
	results = []
	answer=""
	inputtext = r"""The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain "Amazonas" in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species."""
	question =r"""What percentage does the Amazon represents in rainforests on the planet?"""
			
	if request.method == 'POST':
		try:
			inputtext = request.form.get('textvalue')
			question = request.form.get('questionvalue')
			if (inputtext!="" and question!=""):
				answer=get_answer(inputtext,question)
				if (answer==""):
					answer="Not a valid question"
				
				answer=format_text(answer)

			if (inputtext=="" and question==""):
				answer="Please provide valid input text and question"
			elif (inputtext==""):
				answer="Please provide input text"
			elif (question==""):
				answer="Please provide question"

		except:
			errors.append(
                "Unable to process POST data"
            )
	elif request.method == 'GET':
		print("No Post Back Call")

	results.append(inputtext)
	results.append(question)
	results.append(answer)
	return render_template("start.html",errors=errors, results=results)

if __name__ == '__main__':
	global tokenizer
	global model
	modelName = 'bert-large-uncased-whole-word-masking-finetuned-squad'
	tokenizer = BertTokenizer.from_pretrained(modelName)
	model = TFBertForQuestionAnswering.from_pretrained(modelName)
	app.run(host='0.0.0.0', port=8080)