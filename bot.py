import numpy as np
from scratch import DNN, LoadingData

# Load model once and for all (using load data function)
data = LoadingData()
cat_to_tag,tag_to_response = data.cat_to_tag,data.tag_to_response

X_shape,Y_shape = data.X_train,data.Y_train # Need data for the shaped to be used

model = DNN()
model.build(X_shape,Y_shape)
X,Y = model.get_input_array(X_shape),Y_shape
model.load(path="model/scratch.h5")

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        # preprocess the question (cleaning and vectorizing using bag of words)
        input_array = model.get_input_array([inp])
        results_1 = model.predict(input_array)
        # Extract the highest probability from the prediction
        results_index_1 = np.argmax(results_1[0])
        percentage_1 = results_1[0][results_index_1]*100
        # Extract the second highest probability
        results_2 = [x if x*100<percentage_1 else 0 for x in results_1[0]]
        results_index_2 = np.argmax(results_2)
        percentage_2 = results_2[results_index_2]*100
        # Extract the label of the predictions
        pred_1 = cat_to_tag[results_index_1]
        pred_2 = cat_to_tag[results_index_2]
        # save the values in "information"
        information = "%s: %s(%.2f%%), %s(%.2f%%)"%(model._type,pred_1,percentage_1,pred_2,percentage_2)

        if percentage_1>75: # in %
            # Get the recommandations associated for the highest probability intent
            responses = tag_to_response[pred_1]
            if pred_1 in ['greeting']:
                text=responses[0]
            else:
                text = ("I understand that you like %s. These are my recommandations: \n"%pred_1 + "\n".join(responses))

        elif percentage_1>30: # in %
            # if the probability is quit low (between 75% and 30%) give two recommandations
            responses_1 = tag_to_response[pred_1]
            responses_2 = tag_to_response[pred_2]
            text = ("I understand that you like %s and %s. These are my recommandations: \n"%(pred_1,pred_2) + "\n".join(responses_1+responses_2))

        else:
            # Else dont give any recommandation
            text='Sorry, I didn\'t understand. Can you reformulate?'

        print(text)
        print(information)

chat()
