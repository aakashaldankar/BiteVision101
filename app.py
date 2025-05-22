import gradio as gr
import bitevision_model 
import torch
from timeit import default_timer as timer
import os

effnetb2, effnetb2_transforms=bitevision_model.create_effnetb2(101,42)

effnetb2.load_state_dict(torch.load("BiteVision101_e20.pth", map_location=torch.device("cpu"), weights_only=True))

classes=['apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare','beet_salad','beignets',
 'bibimbap','bread_pudding','breakfast_burrito','bruschetta','caesar_salad','cannoli','caprese_salad','carrot_cake','ceviche','cheese_plate',
 'cheesecake','chicken_curry','chicken_quesadilla','chicken_wings','chocolate_cake','chocolate_mousse','churros','clam_chowder',
 'club_sandwich','crab_cakes','creme_brulee','croque_madame','cup_cakes','deviled_eggs','donuts','dumplings','edamame','eggs_benedict',
 'escargots','falafel','filet_mignon','fish_and_chips','foie_gras','french_fries','french_onion_soup','french_toast','fried_calamari',
 'fried_rice','frozen_yogurt','garlic_bread','gnocchi','greek_salad','grilled_cheese_sandwich','grilled_salmon','guacamole','gyoza','hamburger','hot_and_sour_soup',
 'hot_dog','huevos_rancheros','hummus','ice_cream','lasagna','lobster_bisque','lobster_roll_sandwich','macaroni_and_cheese','macarons',
 'miso_soup','mussels','nachos','omelette','onion_rings','oysters','pad_thai','paella','pancakes','panna_cotta','peking_duck','pho','pizza',
 'pork_chop','poutine','prime_rib','pulled_pork_sandwich','ramen','ravioli','red_velvet_cake','risotto','samosa','sashimi','scallops','seaweed_salad','shrimp_and_grits','spaghetti_bolognese',
 'spaghetti_carbonara','spring_rolls','steak','strawberry_shortcake','sushi','tacos','takoyaki','tiramisu','tuna_tartare','waffles']

def make_pred_and_timeit(img):

    start_timer=timer()
    
    transformed_image=effnetb2_transforms(img).unsqueeze(dim=0)
    
    effnetb2.eval()
    with torch.inference_mode():
        pred_logits=effnetb2(transformed_image)
        probs=torch.softmax(pred_logits, dim=1).squeeze()
        
    pred_probs={classes[i]: round(probs[i].item(),3) for i in range(len(probs))}
    pred_time=timer()-start_timer
    
    return pred_probs, pred_time

examples=[['examples/'+example] for example in os.listdir("examples")]

title="BiteVision101: FoodImage Classification Model 🍔 🍕 🥑"
description="🍔 BiteVision101 🍕 is the ultimate food detective! 🕵️‍♂️🍽️ Simply upload an image of any food, and it will instantly identify the dish and tell us its name! 🥑✨ From a mouthwatering pizza 🍕 to a juicy burger 🍔 or a vibrant salad 🥗, BiteVision101 has got it covered! 🎉📸 No need to guess, just snap a pic and let the magic happen! ✨🙌"
article="BiteVision101 is a feature extraction model trained on the Food-101 dataset, leveraging EfficientNetB2 as its backbone. With a total of 7843303 parameters, BiteVision101 delivers powerful performance in food image recognition."
demo=gr.Interface(fn=make_pred_and_timeit, inputs=gr.Image(type="pil"), outputs=[gr.Label(num_top_classes=3, label="Predictions"), 
                                                                     gr.Number(label="Prediction Time(s)")], examples=examples, 
                  title=title, description=description, article=article)

demo.launch()


