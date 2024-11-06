# -*- coding: utf-8 -*-
import re
import random
import os
from typing import List
import json
import statistics
from PIL import Image
import io
from colorama import Fore, Back, Style, init
def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)

def reformate_dialog_data(seed_dialog_data):
    reformated_data = []
    for idx, dialog_data in enumerate(seed_dialog_data):
        if idx % 2 == 0:
            reformated_data.append({
                "role": "user",
                "utterance": dialog_data,
            })
        else:
            reformated_data.append({
                "role": "assistant",
                "utterance": dialog_data,
            })
    return reformated_data

def random_select_scene(image_pool_path, image_info_pool_path, fashion_metadata, furniture_metadata, seed_dialog_data, scene_begin_id=None):
    if scene_begin_id is not None:
        selected_dialog_data = seed_dialog_data[scene_begin_id]
    else:
        selected_dialog_data = random.choice(seed_dialog_data)
    domain = selected_dialog_data["domain"]
    
    scene_images_paths = []
    scene_images_info_paths = []
    for turn_ids, scene_id in selected_dialog_data["scene_ids"].items():
        trimmed_scene_name = scene_id[2:] if scene_id[:2] == "m_" else scene_id
        screenshot_load_path = os.path.join(image_pool_path, f"{trimmed_scene_name}.png")

        json_path = os.path.join(image_info_pool_path, f"{trimmed_scene_name}_scene.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(image_info_pool_path, f"m_{trimmed_scene_name}_scene.json")
        
        if not os.path.exists(json_path) or not os.path.exists(screenshot_load_path):
            continue
        
        scene_images_paths.append(screenshot_load_path)
        scene_images_info_paths.append(json_path)
    
    assert len(scene_images_paths) == len(scene_images_info_paths)
    assert len(scene_images_paths) > 0
    
    if domain == "fashion":
        metadata = fashion_metadata
    elif domain == "furniture":
        metadata = furniture_metadata
    else:
        raise ValueError(f"Invalid domain: {domain}")
    dialog = reformate_dialog_data(selected_dialog_data["dialogue"])
    
    # Collect all objects info in the first scene and select another scene have different classes of objects
    object_info_in_select_scene = extract_object_info(scene_images_info_paths[0], metadata)
    collected_classes_info = collect_type_info(object_info_in_select_scene)
    
    # select another scene with different classes of objects
    while True:
        another_selected_dialog_data = random.choice(seed_dialog_data)
        if another_selected_dialog_data["domain"] != domain:
            continue
        another_scene_images_paths = []
        another_scene_images_info_paths = []
        for turn_ids, scene_id in another_selected_dialog_data["scene_ids"].items():
            trimmed_scene_name = scene_id[2:] if scene_id[:2] == "m_" else scene_id
            screenshot_load_path = os.path.join(image_pool_path, f"{trimmed_scene_name}.png")
            json_path = os.path.join(image_info_pool_path, f"{trimmed_scene_name}_scene.json")
            if not os.path.exists(json_path):
                json_path = os.path.join(image_info_pool_path, f"m_{trimmed_scene_name}_scene.json")
            if not os.path.exists(json_path) or not os.path.exists(screenshot_load_path):
                continue
            another_scene_images_paths.append(screenshot_load_path)
            another_scene_images_info_paths.append(json_path)
        
        assert len(another_scene_images_paths) == len(another_scene_images_info_paths)
        assert len(another_scene_images_paths) > 0
        
        if another_scene_images_info_paths[0] == scene_images_info_paths[0]:
            continue
    
        another_object_info = extract_object_info(another_scene_images_info_paths[0], metadata)
        another_collected_classes_info = collect_type_info(another_object_info)
        
        has_different_type = False
        different_type_name = ""
        for type_info in another_collected_classes_info:
            if type_info not in collected_classes_info:
                has_different_type = True
                different_type_name = type_info
                break
                
        if has_different_type:
            break
        else:
            continue
    
    return scene_images_paths, scene_images_info_paths, metadata, dialog, domain, another_scene_images_paths, another_scene_images_info_paths, different_type_name

def collect_type_info(object_info):
    all_type = set()
    for k, v in object_info.items():
        if "type" in v:
            all_type.add(v["type"])
    return list(all_type)
    
def sample_profile(profile_slots, exclude_name=None):
    sampled_profile = {}
    for slot_key, slot_values in profile_slots.items():
        if slot_key == "Name" and exclude_name is not None:
            sampled_value = random.choice([v for v in slot_values if v != exclude_name])
        else:
            sampled_value = random.choice(slot_values)
        sampled_profile[slot_key] = sampled_value
    return sampled_profile

def extract_object_info(scene_image_info_path, metadata):
    with open(scene_image_info_path, "r", encoding="utf-8") as f:
        scene_info = json.load(f)
    scene_json = scene_info["scenes"][0]["objects"]
    return_obj_info = {}
    for datum in scene_json:
        object_name = datum["prefab_path"]
        if object_name in metadata:
            return_obj_info[object_name] = metadata[object_name]
    return return_obj_info

def simulate_user_preference(objects_info, domain, strict_preference=None):
    if domain == "fashion":
        all_assetType = set()
        all_customerReview = set()
        all_availableSizes = set()
        all_color = set()
        all_pattern = set()
        all_brand = set()
        all_sleeveLength = set()
        all_type = set()
        all_price = set()
        for k, v in objects_info.items():
            all_assetType.add(v.get("assetType", "NULL"))
            all_customerReview.add(v.get("customerReview", "NULL"))
            all_availableSizes.update(set(v.get("availableSizes", "NULL")))
            all_color.add(v.get("color", "NULL"))
            all_pattern.add(v.get("pattern", "NULL"))
            all_brand.add(v.get("brand", "NULL"))
            all_sleeveLength.add(v.get("sleeveLength", "NULL"))
            all_type.add(v.get("type", "NULL"))
            all_price.add(v.get("price", "NULL"))
        
        
        # remove "NULL" from all sets
        all_assetType.discard("NULL")
        all_customerReview.discard("NULL")
        all_availableSizes.discard("NULL")
        all_color.discard("NULL")
        all_pattern.discard("NULL")
        all_brand.discard("NULL")
        all_sleeveLength.discard("NULL")
        all_type.discard("NULL")
        all_price.discard("NULL")
        
        # convert all set to list
        all_assetType = list(all_assetType)
        all_customerReview = list(all_customerReview)
        all_availableSizes = list(all_availableSizes)
        all_color = list(all_color)
        all_pattern = list(all_pattern)
        all_brand = list(all_brand)
        all_sleeveLength = list(all_sleeveLength)
        all_type = list(all_type)
        all_price = list(all_price)
        
        # convert all_customerReview and all_price to float
        all_customerReview = [float(cr) for cr in all_customerReview]
        all_price = [float(p) for p in all_price]
        return_string = ""
        
        if len(all_assetType) != 0:
            assertType_preference = "Assert Type Preference: "
            for i in all_assetType:
                assertType_preference += f"{i}({random.choice(['favor', 'aversion','neutral'])}) "
            assertType_preference = assertType_preference.strip() + ";\n"
            return_string += assertType_preference
        
        if len(all_customerReview) != 0:
            median_value = round(statistics.median(all_customerReview), 1)
            customerReview_preference = f"Review Preference: greater than {median_value};\n"
            return_string += customerReview_preference
        
        if len(all_availableSizes) != 0:
            size_preference = f"Size Preference: {random.choice(list(all_availableSizes))};\n"
            return_string += size_preference
        
        if len(all_color) != 0:
            color_preference = f"Color Preference: "
            for i in all_color:
                color_preference += f"{i}({random.choice(['favor', 'aversion','neutral'])}) "
            color_preference = color_preference.strip() + ";\n"
            return_string += color_preference
        
        if len(all_pattern) != 0:
            pattern_preference = f"Pattern Preference: "
            for i in all_pattern:
                pattern_preference += f"{i}({random.choice(['favor', 'aversion','neutral'])}) "
            pattern_preference = pattern_preference.strip() + ";\n"
            return_string += pattern_preference
        
        if len(all_brand) != 0:
            brand_preference = f"Brand Preference: "
            for i in all_brand:
                brand_preference += f"{i}({random.choice(['favor', 'aversion','neutral'])}) "
            brand_preference = brand_preference.strip() + ";\n"
            return_string += brand_preference
        
        if len(all_sleeveLength) != 0:
            sleeveLength_preference = f"Sleeve Length Preference: "
            for i in all_sleeveLength:
                sleeveLength_preference += f"{i}({random.choice(['favor', 'aversion','neutral'])}) "
            sleeveLength_preference = sleeveLength_preference.strip() + ";\n"
            return_string += sleeveLength_preference
        
        if len(all_type) != 0:
            type_preference = f"Type Preference: "
            for i in all_type:
                if strict_preference is not None and i == strict_preference:
                    type_preference += f"{i}({random.choice(['favor', 'neutral'])}) "
                else:
                    type_preference += f"{i}({random.choice(['favor', 'aversion','neutral'])}) "
            type_preference = type_preference.strip() + ";\n"
            return_string += type_preference
        
        if len(all_price) != 0:
            median_value = round(statistics.median(all_price), 2)
            price_preference = f"Price Preference: less than {median_value};\n"
            return_string += price_preference
        
        return return_string
    elif domain == "furniture":
        all_brand = set()
        all_color = set()
        all_customerRating = set()
        all_materials = set()
        all_price = set()
        all_type = set()
        for k, v in objects_info.items():
            all_brand.add(v.get("brand", "NULL"))
            all_color.add(v.get("color", "NULL"))
            all_customerRating.add(v.get("customerRating", "NULL"))
            all_materials.add(v.get("materials", "NULL"))
            all_price.add(v.get("price", "NULL"))
            all_type.add(v.get("type", "NULL"))
        
        all_brand.discard("NULL")
        all_color.discard("NULL")
        all_customerRating.discard("NULL")
        all_materials.discard("NULL")
        all_price.discard("NULL")
        all_type.discard("NULL")
        
        all_brand = list(all_brand)
        all_color = list(all_color)
        all_customerRating = list(all_customerRating)
        all_materials = list(all_materials)
        all_price = list(all_price)
        all_type = list(all_type)
        
        all_customerRating = [float(cr) for cr in all_customerRating]
        all_price = [float(p[1:]) for p in all_price] # remove the $ sign
        
        return_string = ""
        
        if len(all_brand) != 0:
            brand_preference = f"Brand Preference: "
            for i in all_brand:
                brand_preference += f"{i}({random.choice(['favor', 'aversion','neutral'])}) "
            brand_preference = brand_preference.strip() + ";\n"
            return_string += brand_preference
        
        if len(all_color) != 0:
            color_preference = f"Color Preference: "
            for i in all_color:
                color_preference += f"{i}({random.choice(['favor', 'aversion','neutral'])}) "
            color_preference = color_preference.strip() + ";\n"
            return_string += color_preference
            
        if len(all_customerRating) != 0:
            median_value = round(statistics.median(all_customerRating), 1)
            customerRating_preference = f"Review Preference: greater than {median_value};\n"
            return_string += customerRating_preference
        
        if len(all_materials) != 0:
            materials_preference = f"Materials Preference: "
            for i in all_materials:
                materials_preference += f"{i}({random.choice(['favor', 'aversion','neutral'])}) "
            materials_preference = materials_preference.strip() + ";\n"
            return_string += materials_preference
        
        if len(all_price) != 0:
            median_value = round(statistics.median(all_price), 2)
            price_preference = f"Price Preference: less than {median_value};\n"
            return_string += price_preference
            
        if len(all_type) != 0:
            type_preference = f"Type Preference: "
            for i in all_type:
                if strict_preference is not None and i == strict_preference:
                    type_preference += f"{i}({random.choice(['favor', 'neutral'])}) "
                else:
                    type_preference += f"{i}({random.choice(['favor', 'aversion','neutral'])}) "
            type_preference = type_preference.strip() + ";\n"
            return_string += type_preference
        
        return return_string
        
        
def reformat_objects_info(objects_info, domain):
    return_reformat_objects_info = ""
    if domain == "fashion":
        for k, v in objects_info.items():
            return_reformat_objects_info += v["assetType"] + ": customer review (" + str(v["customerReview"]) + "), available sizes (" + " ".join(v["availableSizes"]) + "), color (" + v["color"] + "), pattern (" + v["pattern"] + "), brand (" + v["brand"] + "), sleeve length (" + v["sleeveLength"] + "), type (" + v["type"] + "), price (" + str(v["price"]) + ");\n"
    elif domain == "furniture":
        for k, v in objects_info.items():
            return_reformat_objects_info += v["type"] + ": customer rating (" + str(v["customerRating"]) + "), color (" + v["color"] + "), brand (" + v["brand"] + "), materials (" + v["materials"] + "), price (" + v["price"] + ");\n"
    return return_reformat_objects_info

# 将 PNG 图片转换为 JPEG，并压缩到100KB以内
def compress_image(input_path, output_path, target_size_kb=100):
    # 打开原始图片
    with Image.open(input_path) as img:
        # 转换为 JPEG 格式
        img = img.convert('RGB')

        # 逐步调整质量来压缩图片大小
        quality = 85  # 初始压缩质量
        while True:
            # 将图片保存到字节流中
            with io.BytesIO() as buffer:
                img.save(buffer, format='JPEG', quality=quality)
                size_kb = buffer.tell() / 1024  # 计算图片的KB大小
                
                if size_kb <= target_size_kb or quality <= 10:  # 满足条件则停止
                    buffer.seek(0)
                    # 将压缩后的图片保存到文件中
                    with open(output_path, 'wb') as f:
                        f.write(buffer.read())
                    break
                # 降低质量继续压缩
                quality -= 5

def create_instruct(
        scene_image_path,
        scene_image_info_path,
        metadata,
        domain,
        user_profile,
        assistant_profile,
        user_personality,
        assistant_personality,
        conversation_end_or_continue_sample,
        max_interaction_step,
        second_scene_image_path,
        second_scene_image_info_path,
        different_type_name
    ):
    """Create instructions about the conversation environment and roles."""
    # Describe the environment (shared by all roles)
    env_desc = f"You are now participating in a conversation happening in a {domain} store."
    if scene_image_path is not None and second_scene_image_path is not None:
        env_desc += f"The image provided is the snapshot of this store."
    
    # Describe the user
    user_desc = "You are {}, ".format(user_profile["Name"])
    profile_desc = ""
    
    if user_profile["Occupation"] == "Student":
        if user_profile["Gender"] == "Male":
            profile_desc = "a male student in the age range of {}, living in {}".format(user_profile["Age Range"].lower(), user_profile["Residence"])
        else:
            profile_desc = "a female student in the age range of {}, living in {}".format(user_profile["Age Range"].lower(), user_profile["Residence"])
    elif user_profile["Occupation"] == "Employed":
        if user_profile["Gender"] == "Male":
            profile_desc = "a man in the age range of {}, working in a company and living in {}".format(user_profile["Age Range"].lower(), user_profile["Residence"])
        else:
            profile_desc = "a woman in the age range of {}, working in a company and living in {}".format(user_profile["Age Range"].lower(), user_profile["Residence"])
    else:
        if user_profile["Gender"] == "Male":
            profile_desc = "a retired man in the age range of {}, living in {}".format(user_profile["Age Range"].lower(), user_profile["Residence"])
        else:
            profile_desc = "a retired woman in the age range of {}, living in {}".format(user_profile["Age Range"].lower(), user_profile["Residence"])
    user_desc += profile_desc + ".\n\n"
    
    # extract_object_info
    objects_info = extract_object_info(scene_image_info_path, metadata)
    second_objects_info = extract_object_info(second_scene_image_info_path, metadata)
    # user_desc += "You are currently in a {} store, where you can see the following objects:\n".format(domain)
    user_desc += "Based on your past experiences, you have the following preferences:\n"
    
    simulate_preference = simulate_user_preference(objects_info, domain=domain)
    second_simulate_preference = simulate_user_preference(second_objects_info, domain=domain, strict_preference=different_type_name)
    
    user_desc_original = user_desc + simulate_preference + "\n"
    user_desc_transition = user_desc + second_simulate_preference + "\n"
    
    # print (Fore.GREEN + f"user_desc_original: {user_desc_original}" + Style.RESET_ALL)
    # print (Fore.GREEN + f"user_desc_transition: {user_desc_transition}" + Style.RESET_ALL)
    
    user_desc_original += "Based on the Big-5 personality traits, your personality is measured as:\n"
    user_desc_transition += "Based on the Big-5 personality traits, your personality is measured as:\n"
    for k, v in user_personality.items():
        user_desc_original += "For {}, you are {}.\n".format(k, v)
        user_desc_transition += "For {}, you are {}.\n".format(k, v)
    user_desc_original += "\n"
    user_desc_transition += "\n"
    
    user_desc_original += "Your response should match your profile and personality, and be concise (no longer than 30 words).\n"
    user_desc_original += "You don't need to recommend anything, but feel free to express your personal interests."
    
    user_desc_transition += "Your response should match your profile and personality, and be concise (no longer than 30 words).\n"
    user_desc_transition += "You don't need to recommend anything, but feel free to express your personal interests."
    
    user_desc_at_transition_turn = user_desc_transition + " In this round, you must seek recommendations of type {0}, and you should shift the topic of the conversation to the content of type {0}.".format(different_type_name)
    user_desc_at_transition_turn += "For example: 'That sounds good, but I would like to buy a type {0} instead.'".format(different_type_name)

    user_dict = {
        "name": user_profile["Name"],
        "role_desc": user_desc_original,
        "role_desc_in_transition_turn": user_desc_at_transition_turn,
        "role_desc_after_transition_turn": user_desc_transition
    }
    
    assistant_name = assistant_profile["Name"]
    assistant_desc = f"You are {assistant_name}, a saleperson in a {domain} store.\n"
    assistant_desc += "You are here to converse and assist {} with {} shopping.\n".format(user_profile["Name"], domain)
    assistant_desc += "Your goal is to capture user preferences and make recommendations based on the products in the store."
    assistant_desc += "Below are the products available in the store:\n\n"
    
    assistant_desc_original = assistant_desc + reformat_objects_info(objects_info, domain=domain) + "\n"
    assistant_desc_transition = assistant_desc + reformat_objects_info(second_objects_info, domain=domain) + "\n"
    
    assistant_desc_original += "Be informative and engaging while providing insights to arouse {}'s interest.\n".format(user_profile["Name"])
    assistant_desc_original += "Your words at each turn should be concise (no longer than 30 words).\n\n"
    
    assistant_desc_transition += "Be informative and engaging while providing insights to arouse {}'s interest.\n".format(user_profile["Name"])
    assistant_desc_transition += "Your words at each turn should be concise (no longer than 30 words).\n\n"
    
    assistant_dict = {
        "name": assistant_profile["Name"],
        "role_desc": assistant_desc_original,
        "role_desc_in_transition_turn": assistant_desc_original,
        "role_desc_after_transition_turn": assistant_desc_transition
    }

    # Describe the moderator
    moderator_desc = "You are the moderator of a conversation. You need to determine whether the discussion between Role-S and Role-U should come to an immediate end.\n"
    moderator_desc += "The conversation should conclude under the following two conditions:\n"
    moderator_desc += "(1) If Role-S completes the recommendation, the user accepts the recommendation\n"
    moderator_desc += "(2) If Role-U explicitly rejects Role-S's recommendation when Role-S has tried to recommend it for the second time.\n"
    moderator_desc += f"(3) The conversation between the user and the system reaches the maximum number of rounds limit ({max_interaction_step} rounds)"
    
    moderator_desc += "In either of these cases, the conversation should be brought to an immediate end.\n\n"

    moderator_desc += "For example, here is a conversation:\n## {}".format(conversation_end_or_continue_sample["seed_continue"])
    moderator_desc += "Should the conversation end? The answer is no.\n\n"
    moderator_desc += "Here is another conversation:\n## {}".format(conversation_end_or_continue_sample["seed_end"])
    moderator_desc += "Should the conversation end? The answer is yes."

    terminal_condition = "Now, for the above discussion between {} (Role-S) and {} (Role-U), should the conversation end? Answer yes or no.".format(user_profile["Name"], assistant_name)

    moderator_dict = {
        "role_desc": moderator_desc,
        "terminal_condition": terminal_condition
    }

    return env_desc, user_dict, assistant_dict, moderator_dict, objects_info, simulate_preference, second_objects_info, second_simulate_preference