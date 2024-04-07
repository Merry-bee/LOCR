from PIL import Image,ImageOps,ImageDraw,ImageColor
import torch
import os
import json
import subprocess
from pathlib import Path
import json

def visual_box(png_path,boxes,save_path,color=(255,0,0),image_size = [672,896],texts=None,fill=True):
    img = Image.open(png_path).resize(image_size)
    img=img.convert('RGBA')
    transp = Image.new('RGBA', image_size, (0,0,0,0))
    draw = ImageDraw.Draw(transp, "RGBA")   

    boxes = boxes.reshape(-1,2,2)
    if not isinstance(color,tuple):
        color = ImageColor.getrgb(color)
    if fill:
        fill_color = color + (80,)   # 一半的透明度
    else:
        fill_color = (255,255,255,0)    # 透明
    
    try:
        for i,box in enumerate(boxes):
            if box[1][0] < box[0][0]: 
                print('x2<x1')
                box[1][0] = box[0][0]
            if box[1][1] < box[0][1]:
                print('y2<y1')
                box[1][1] = box[0][1]
            resized_box = torch.empty_like(box)
            resized_box[:,0] = box[:,0]*image_size[0]   # x
            resized_box[:,1] = box[:,1]*image_size[1]   # y
            
            
            
            draw.rectangle([tuple(resized_box[0]),tuple(resized_box[1])],outline=color,fill=fill_color)
            if texts:
                if color == 'blue':
                    draw.text((resized_box[0][0]-10,resized_box[0][1]-20),texts[i].encode("utf-8").decode("latin1"),fill=color)
                else:
                    draw.text((resized_box[0][0]-10,resized_box[0][1]-10),texts[i].encode("utf-8").decode("latin1"),fill=color)
        img.paste(Image.alpha_composite(img, transp))
        img.save(save_path)   
    except Exception as e:
        print(e)
        


def interact_with_human(prompt_pred,flask_png_path,save_path,color ='red',image_size = [672,896]):
   
    img = Image.open(flask_png_path).resize(image_size)
    draw = ImageDraw.Draw(img)
    prompt_pred[:,0] *= image_size[0]
    prompt_pred[:,1] *= image_size[1]
    if prompt_pred[0,0] < prompt_pred[1,0] and prompt_pred[0,1] < prompt_pred[1,1]:
        draw.rectangle([tuple(prompt_pred[0]),tuple(prompt_pred[1])],outline=color)
   
    img.save(flask_png_path)     
    
    os.chdir('flask-image-annotator')
    process = subprocess.Popen(['python','app.py'])
    os.chdir('../')
    process.terminate()
    with open('flask-image-annotator/out.json','r') as fi:
        user_input = fi.readline()
    if user_input: 
        dct = json.loads(user_input)
        token_user = dct['name'].replace('\\n','\n')  
        prompt_user = [[dct['x1']/image_size[0],dct['y1']/image_size[1]],[dct['x2']/image_size[0],dct['y2']/image_size[1]]]
        visual_box(png_path=save_path,boxes=torch.tensor(prompt_user),save_path=save_path,color='blue',image_size = [672,896],fill=True)      
        return prompt_user,token_user
    else:
        return [],''
    
