#! /usr/bin/env python
import openai
import argparse
import collections
import json
import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm as progressbar
import requests
import time
import base64

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def draw_bboxes(bboxes, load_path, save_path, verbose=False):
    """Draw bounding boxes given the screenpath and list of bboxes.

    Args:
        bboxes: List of all bounding boxes
        load_path: Path to load the screenshot
        save_path: Path to save the screenshot with bounding boxes
        verbose: Print out status statements
    """
    # Read images and draw rectangles.
    image = Image.open(load_path)
    draw = ImageDraw.Draw(image)
    # Get a font.
    font = ImageFont.truetype("arial.ttf", 25)
    offset = 2
    object_names = []
    for index, bbox_datum in enumerate(bboxes):
        object_index = bbox_datum.get("index", index)
        verts = bbox_datum["bbox"]
        name = bbox_datum["prefab_path"]
        draw.rectangle(
            [(verts[0], verts[1]), (verts[0] + verts[3], verts[1] + verts[2])], outline="red", width=2
        )
        # Draw text with black background.
        # text = str(object_index)
        # text_width, text_height = font.getsize(text)
        # draw.rectangle(
        #     (
        #         verts[0] + offset,
        #         verts[1] + offset,
        #         verts[0] + 2 * offset + text_width,
        #         verts[1] + 2 * offset + text_height,
        #     ),
        #     fill="black",
        # )
        draw.text(
            (verts[0] + verts[3]/2-1, verts[1] + verts[2]/2-1),
            str(object_index),  # str(index)
            fill="red",
            font=font,
        )
        object_names.append(name)
    # Save the image with bbox drawn.
    if verbose:
        print("Saving: {}".format(save_path))
    image.save(save_path, "PNG")
    # for idx, name in enumerate(object_names):
    #     print (idx, ":", name)
    
def get_screenshot_save_path(screenshot_path):
    file_name, extension = screenshot_path.rsplit(".", 1)
    return "{}_bbox_draw.{}".format(file_name, extension)

def count_fashion_attributes(meta_fashion_file):
    with open(meta_fashion_file, "r") as file_id:
        meta_fashion = json.load(file_id)
    # Count the number of attributes.
    all_attributes = {}
    for item_name, item_attributes in meta_fashion.items():
        for attribute in item_attributes.keys():
            if attribute not in all_attributes:
                if attribute == "availableSizes":
                    all_attributes[attribute] = item_attributes[attribute]
                elif attribute == "color":
                    all_attributes[attribute] = [i.strip() for i in item_attributes[attribute].split(",")]
                elif item_attributes[attribute] == "":
                    continue
                else:
                    all_attributes[attribute] = [item_attributes[attribute]]
            else:
                if attribute == "availableSizes":
                    for size in item_attributes[attribute]:
                        if size not in all_attributes[attribute]:
                            all_attributes[attribute] += [size]
                elif attribute == "color":
                    for color in [i.strip() for i in item_attributes[attribute].split(",")]:
                        if color not in all_attributes[attribute]:
                            all_attributes[attribute] += [color]
                elif item_attributes[attribute] == "":
                    continue
                else:
                    if item_attributes[attribute] not in all_attributes[attribute]:
                        all_attributes[attribute] += [item_attributes[attribute]]
    # Print the counts.
    for attribute in all_attributes.keys():
        print(f"{attribute}: {len(all_attributes[attribute])}")
        print(all_attributes[attribute])
    print(f"Total attributes: {len(all_attributes.keys())}")

def count_furniture_attributes(meta_furniture_file):
    with open(meta_furniture_file, "r") as file_id:
        meta_fashion = json.load(file_id)
    # Count the number of attributes.
    all_attributes = {}
    for item_name, item_attributes in meta_fashion.items():
        for attribute in item_attributes.keys():
            if attribute not in all_attributes:
                if attribute == "color":
                    all_attributes[attribute] = [i.strip() for i in item_attributes[attribute].split(",")]
                elif item_attributes[attribute] == "":
                    continue
                else:
                    all_attributes[attribute] = [item_attributes[attribute]]
            else:
                if attribute == "color":
                    for color in [i.strip() for i in item_attributes[attribute].split(",")]:
                        if color not in all_attributes[attribute]:
                            all_attributes[attribute] += [color]
                elif item_attributes[attribute] == "":
                    continue
                else:
                    if item_attributes[attribute] not in all_attributes[attribute]:
                        all_attributes[attribute] += [item_attributes[attribute]]
    # Print the counts.
    for attribute in all_attributes.keys():
        print(f"{attribute}: {len(all_attributes[attribute])}")
        print(all_attributes[attribute])
    print(f"Total attributes: {len(all_attributes.keys())}")

def progress_scene(scene_names, scene_jsons_root, scene_images_root, save_root):
    for scene in progressbar(scene_names):
        # print ("Scene: ", scene)
        json_path = os.path.join(scene_jsons_root, f"{scene}_scene.json")
        # Check if file exists, else try with "m_"
        if not os.path.exists(json_path):
            json_path = os.path.join(scene_jsons_root, f"m_{scene}_scene.json")
            assert os.path.exists(json_path), f"{json_path} not found!"
        with open(json_path, "r") as file_id:
            scene_json = json.load(file_id)
        object_bboxes = scene_json["scenes"][0]["objects"]

        # Image load and save paths.
        trimmed_scene_name = scene[2:] if scene[:2] == "m_" else scene
        screenshot_load_path = os.path.join(
            scene_images_root, f"{trimmed_scene_name}.png"
        )
        screenshot_save_path = os.path.join(
            save_root, f"{trimmed_scene_name}_bbox.png"
        )
        draw_bboxes(object_bboxes, screenshot_load_path, screenshot_save_path)
        # try:
        #     draw_bboxes(object_bboxes, screenshot_load_path, screenshot_save_path)
        # except:
        #     print ("Error: ", scene)
        #     continue

def query_gpt4v(scene_names, scene_jsons_root, scene_images_root):
    api_key = os.environ.get("OPENAI_API_KEY")
    bbox_generated_info = {}
    for scene in progressbar(scene_names[:10]):
        trimmed_scene_name = scene[2:] if scene[:2] == "m_" else scene
        screenshot_load_path = os.path.join(
            scene_images_root, f"{trimmed_scene_name}_bbox.png"
        )
        json_path = os.path.join(scene_jsons_root, f"{scene}_scene.json")
        # Check if file exists, else try with "m_"
        if not os.path.exists(json_path):
            json_path = os.path.join(scene_jsons_root, f"m_{scene}_scene.json")
            assert os.path.exists(json_path), f"{json_path} not found!"
        with open(json_path, "r") as file_id:
            scene_json = json.load(file_id)
        object_bboxes = scene_json["scenes"][0]["objects"]
        bbox_info = ""
        for index, bbox_datum in enumerate(object_bboxes):
            object_index = bbox_datum.get("index", index)
            name = bbox_datum["prefab_path"]
            bbox_info += f"{object_index}: {name}\n"
        prompt = "This is a picture with various types of clothes hanging. Several boxes are drawn on the image, and each box encloses a piece of clothing, marked with a corresponding number. Describe the clothes according to the number sign, focusing on the color, type, and pattern. Below is the name of each piece of clothing. It is auxiliary information to help you identify the clothes:[" + bbox_info[:-1] + "]"

        base64_image = encode_image(screenshot_load_path)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
                }
        ],
        "max_tokens": 1000
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        content = response.json()["choices"][0]["message"]["content"]
        bbox_generated_info[f"{trimmed_scene_name}_bbox"] = {
            "image_name": f"{trimmed_scene_name}_bbox.png",
            "content": content
        }
        time.sleep(1)
    return bbox_generated_info

def main():
    fashion_meta_file = "./original_data/fashion_prefab_metadata_all.json"
    # count_fashion_attributes(fashion_meta_file)
    furniture_meta_file = "./original_data/furniture_prefab_metadata_all.json"
    # count_furniture_attributes(furniture_meta_file)
    
    train_scene_jsons_root = "./original_data/train_scene_jsons/"
    train_scene_images_root = "./original_data/train_scene_images/"
    test_scene_jsons_root = "./original_data/test_scene_jsons/"
    test_scene_images_root = "./original_data/test_scene_images/"
    save_boxed_train_images_root = "./generated_info/train_boxed_images/"
    save_boxed_test_images_root = "./generated_info/test_boxed_images/"
    
    train_scene_names = sorted(list(set([i.rsplit("_", 1)[0] for i in os.listdir(train_scene_jsons_root)])))
    test_scene_names = sorted(list(set([i.rsplit("_", 1)[0] for i in os.listdir(test_scene_jsons_root)])))
    print ("Train scenes: ", len(train_scene_names))
    print ("Test scenes: ", len(test_scene_names))
    
    # progress_scene(train_scene_names, train_scene_jsons_root, train_scene_images_root, save_boxed_train_images_root)
    # progress_scene(test_scene_names, test_scene_jsons_root, test_scene_images_root, save_boxed_test_images_root)
    train_snapshot_json_file_path = "./generated_info/train_generated_info.json"
    train_generated_info = query_gpt4v(train_scene_names, train_scene_jsons_root, save_boxed_train_images_root)
    with open(train_snapshot_json_file_path, "w") as file_id:
        json.dump(train_generated_info, file_id, indent=4)
    # test_snapshot_json_file_path = "./generated_info/test_generated_info.json"
    # test_generated_info = query_gpt4v(test_scene_names, test_scene_jsons_root, save_boxed_test_images_root)
    # with open(test_snapshot_json_file_path, "w") as file_id:
    #     json.dump(test_generated_info, file_id, indent=4)
    

if __name__ == "__main__":
    main()
