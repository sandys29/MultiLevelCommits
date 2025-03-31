import pandas as pd
import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import argparse

def calc_score(reference, predicted):
    coco = COCO(reference)
    coco_result = coco.loadRes(predicted)
    coco_eval = COCOEvalCap(coco, coco_result)
    try:
        coco_eval.evaluate()
    except:
        pass
    for metric, score in coco_eval.eval.items():
        try:
            print(f'{metric}: {score:.3f}')
        except:
            pass

if __name__ == '__main__':
    #Get all Files
    parser = argparse.ArgumentParser(description='Process necessary API keys.')
    parser.add_argument('--FILE1', type=str, required=True, help='Required File 1 location')
    parser.add_argument('--FILE2', type=str, required=True, help='Required File 2 location')
    parser.add_argument('--FILE3', type=str, required=True, help='Required File 3 location')
    parser.add_argument('--FILE4', type=str, required=True, help='Required File 4 location')
    parser.add_argument('--FILE5', type=str, required=True, help='Required File 5 location')
    
    #Store Dataframes
    args = parser.parse_args()
    df1 = pd.read_csv(args.FILE1)
    df2 = pd.read_csv(args.FILE2)
    df3 = pd.read_csv(args.FILE3)
    df4 = pd.read_csv(args.FILE4)
    df5 = pd.read_csv(args.FILE5)
    
    #concat dataframes
    df=pd.concat([df1, df2,df3,df4,df5], ignore_index=True)
    
    #Store actual captions
    captions = list(df['label'])
    
    #Store Predicted Captions per model
    cap1 = list(df['llama-70b-output'])
    cap2 = list(df['llama3.1-8b-output'])
    cap3 = list(df['mistral-large-output'])
    cap4 = list(df['gpt-4o-output'])
    new_cap1, new_cap2, new_cap3, new_cap4=[], [], [], []
    for k,v in enumerate(cap1):
        new_cap1.append({'image_id': k, 'caption': v})

    for k,v in enumerate(cap2):
        new_cap2.append({'image_id': k, 'caption': v})

    for k,v in enumerate(cap3):
        new_cap3.append({'image_id': k, 'caption': v})

    for k,v in enumerate(cap4):
        new_cap4.append({'image_id': k, 'caption': v})

    new_ref = {'images': [], 'annotations': []}
    for k, refs in enumerate(captions):
        new_ref['images'].append({'id': k})
        new_ref['annotations'].append({'image_id': k, 'id': k, 'caption': refs})

    #Create Json files
    with open('references.json', 'w') as fgts:
        json.dump(new_ref, fgts)
    with open('captions1.json', 'w') as fres:
        json.dump(new_cap1, fres)

    with open('captions2.json', 'w') as fres:
        json.dump(new_cap2, fres)

    with open('captions3.json', 'w') as fres:
        json.dump(new_cap3, fres)

    with open('captions4.json', 'w') as fres:
        json.dump(new_cap4, fres)
    
    #Calculate Scores of each model 
    print('Llama3.1 70B Score -------')
    calc_score('references.json','captions1.json')
    
    print('Llama3.1 8B Score -------')
    calc_score('references.json','captions2.json')
    
    print('Mistral-Large Score -------')
    calc_score('references.json','captions3.json')
    
    print('GPT-4o Score -------')
    calc_score('references.json','captions4.json')