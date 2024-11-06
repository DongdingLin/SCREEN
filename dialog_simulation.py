# -*- coding: utf-8 -*-
import time
import json
import os
import random
import argparse
from tqdm import tqdm
from chatarena.agent import Player, Moderator
from chatarena.backends import OpenAIChat
from chatarena.environments.conversation import ModeratedConversation
from chatarena.arena import Arena
from data_utils import find_word_in_string, random_select_scene, sample_profile, create_instruct, compress_image
from colorama import Fore, Back, Style, init


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Scene Information Pool
    parser.add_argument("--train_scenes_images_pool_path", type=str, default="./scene_info_pool/original_data/train_scene_images", help="The training scenes images pool.")
    parser.add_argument("--test_scenes_images_pool_path", type=str, default="./scene_info_pool/original_data/test_scene_images", help="The test scenes images pool.")
    parser.add_argument("--train_scenes_images_info_pool_path", type=str, default="./scene_info_pool/original_data/train_scene_jsons", help="The training scenes images info pool.")
    parser.add_argument("--test_scenes_images_info_pool_path", type=str, default="./scene_info_pool/original_data/test_scene_jsons", help="The test scenes images info pool.")
    parser.add_argument("--fashion_metadata_path", type=str, default="./scene_info_pool/original_data/fashion_prefab_metadata_all.json", help="The fashion metadata file.")
    parser.add_argument("--furniture_metadata_path", type=str, default="./scene_info_pool/original_data/furniture_prefab_metadata_all.json", help="The furniture metadata file.")
    parser.add_argument("--user_profiles_path", type=str, default="./seed_dataset/caches/db_slot/slot_profiles_filtered.json", help="The user profiles slot-values file.")
    
    parser.add_argument("--seed_data_path", type=str, default="./scene_info_pool/original_data/all_data.json", help="The seed data path.")
    
    # Generate Parameters
    parser.add_argument("--max_generated_dialogs", type=int, default=500, help="The max number of dialogs to generate.")
    parser.add_argument("--max_interaction_step", type=int,default=30, help="The max number of interaction steps, i.e., 2 * max rounds.")
    parser.add_argument("--min_transition_step", type=int, default=3, help="The min number of transition steps.")
    parser.add_argument("--max_transition_step", type=int, default=5, help="The max number of transition steps.")
    parser.add_argument("--max_system_tokens", type=int, default=100, help="The max number of tokens to generate for the system.")
    parser.add_argument("--max_user_tokens", type=int, default=80, help="The max number of tokens to generate for the user.")
    parser.add_argument("--max_moderator_tokens", type=int, default=10, help="The max number of tokens to generate for the moderator.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="The chat model to use.")
    parser.add_argument("--output_dir", type=str, default="data/SCREEN", help="The output directory to save the simulated dialog data.")
    parser.add_argument("--temperature", type=float, default=0.75, help="The temperature to use in sampling.")
    parser.add_argument("--small_img_cache_dir", type=str, default="./cache/images", help="The directory to save the small image cache.")
    
    # Output Control
    parser.add_argument("--show_message", type=str2bool, default="true", help="Whether to show the conversation messages.")
    parser.add_argument("--show_description", type=str2bool, default="false", help="Whether to show the role description.")
    
    parser.add_argument("--random_seed", type=int, default=1135)
    return parser.parse_args()

def str2bool(v):
    if v.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif v.lower() in ('false',' no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")

def prompt_conversation(conversation):
    """Prompt the conversation context."""
    conversation_ctx = ""
    for utterance_dict in conversation:
        if utterance_dict["role"] == "user":
            conversation_ctx += f"[Role-U]: {utterance_dict['utterance']}<EOS>\n\n"
        elif utterance_dict["role"] == "assistant":
            conversation_ctx += f"[Role-S]: {utterance_dict['utterance']}<EOS>\n\n"
        else:
            raise ValueError("Invalid role in conversation.")
    return conversation_ctx

def sample_continue_or_end_conversation(conversation):
    """Sample seed conversations (continue | end)."""
    conv_lens = len(conversation)
    continue_len = random.choice(range(1, int(conv_lens * 0.6)))
    conv_continue = prompt_conversation(conversation[:continue_len])
    conv_end = prompt_conversation(conversation)
    seed_conv = {
        "seed_continue": conv_continue,
        "seed_end": conv_end
    }
    return seed_conv

def sample_assistant_role(profile_slots, user_profile):
    """Sample an assistant role."""
    all_names = profile_slots["Name"]
    user_name = user_profile["Name"]
    sampled_name = random.choice(all_names)
    while find_word_in_string(sampled_name, user_name):
        sampled_name = random.choice(all_names)
    return sampled_name

def sample_personality():
    """Sample a personality based on Big Five personality traits."""
    personalities = {
        "agreeableness": ["trustworthy, straightforward, and generous", "unreliable, complicated, meager, and boastful"],
        "conscientiousness": ["efficient, organized, and careful", "inefficient, careless, and sloppy"],
        "extraversion": ["outgoing, energetic, and talkative", "shy, reserved, and quiet"],
        "neuroticism": ["sensitive, nervous, and insecure", "secure, confident, and calm"],
        "openness": ["intellectual, imaginative, and curious", "unimaginative, uncreative, and conventional"]
    }
    sampled_personality = {}
    for trait, values in personalities.items():
        sampled_personality[trait] = random.choice(values)
    return sampled_personality

#### generate dialog data
def generate_dialog_data(
    train_scenes_images_pool_path,
    test_scenes_images_pool_path,
    train_scenes_images_info_pool_path,
    test_scenes_images_info_pool_path,
    fashion_metadata_path,
    furniture_metadata_path,
    user_profiles_path,
    seed_data_path,
    max_generated_dialogs=2,
    max_interaction_step=10,
    min_transition_step=3,
    max_transition_step=5,
    max_system_tokens=100,
    max_user_tokens=80,
    max_moderator_tokens=10,
    model_name="gpt-4o-mini",
    temperature=0.75,
    output_dir=os.path.join("GeneratedData", "SCREEN"),
    show_description=True,
    show_message=True,
    small_image_cache_dir="./cache/images"
):
    if not os.path.exists(seed_data_path):
        raise ValueError(f"Few-shot data path {seed_data_path} does not exist.")
    else:
        with open(seed_data_path, "r", encoding='utf-8') as f:
            seed_dialog_data = json.load(f)
        # print (Fore.RED + f"Loaded Few-shot Data" + Style.RESET_ALL, flush=True)
        # print (Fore.GREEN + f"Total samples in Few-shot Data: {len(seed_dialog_data)}" + Style.RESET_ALL, flush=True)
    
    # load user profiles
    with open(user_profiles_path, "r", encoding='utf-8') as f:
        profile_slots = json.load(f)
    # print (Fore.RED + f"Loaded User Profiles" + Style.RESET_ALL, flush=True)
    # print (Fore.GREEN + f"Total Profiles Keys: {len(profile_slots)}" + Style.RESET_ALL, flush=True)
    # for key, value in profile_slots.items():
    #     print (f"Key: {key} - Num of Value: {len(value)}", flush=True)
        
    # load fashion metadata
    with open(fashion_metadata_path, "r", encoding='utf-8') as f:
        fashion_metadata = json.load(f)
    # print (Fore.RED + f"Loaded Fashion Metadata" + Style.RESET_ALL, flush=True)
    # print (Fore.GREEN + f"Total Fashion Metadata: {len(fashion_metadata)}" + Style.RESET_ALL, flush=True)
    
    # load furniture metadata
    with open(furniture_metadata_path, "r", encoding='utf-8') as f:
        furniture_metadata = json.load(f)
    # print (Fore.RED + f"Loaded Furniture Metadata" + Style.RESET_ALL, flush=True)
    # print (Fore.GREEN + f"Total Furniture Metadata: {len(furniture_metadata)}" + Style.RESET_ALL, flush=True)

    # Create the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(small_image_cache_dir):
        os.makedirs(small_image_cache_dir)
        
    # Create the output file with the timestamp
    time_flag = time.strftime("%Y%m%d%H%M%S", time.localtime())
    output_path = os.path.join(output_dir, f"simulated_dialogs_{time_flag}.json")
    
    
    with open(output_path, "w", encoding='utf-8') as fw:
        for i in range(max_generated_dialogs):
            print (Fore.RED + f"Generating Dialog {i+1}/{max_generated_dialogs}" + Style.RESET_ALL, flush=True)
            
            scene_image_paths, scene_image_info_paths, metadata, dialog_example, domain, second_scene_images_paths, second_scene_images_info_paths, different_type_name = random_select_scene(train_scenes_images_pool_path, train_scenes_images_info_pool_path, fashion_metadata, furniture_metadata, seed_dialog_data, scene_begin_id=None)
            # print (Fore.GREEN + f"Selected Scene: {scene_image_paths}" + Style.RESET_ALL)
            # print (Fore.GREEN + f"Selected Scene Info: {scene_image_info_paths}" + Style.RESET_ALL)
            # print (Fore.GREEN + f"Selected Domain: {domain}" + Style.RESET_ALL)
            # print (Fore.GREEN + f"Selected Dialog Example: {dialog_example}" + Style.RESET_ALL)
            
            # dict return {"seed_continue", "seed_end"}
            conversation_end_or_continue_sample = sample_continue_or_end_conversation(dialog_example)

            # randomly sample a personality
            # {"agreeableness": ["trustworthy], "conscientiousness": ["efficient"], "extraversion": ["outgoing"], "neuroticism": ["sensitive"], "openness": ["intellectual"]}
            simulated_user_personality = sample_personality()
            simulated_assistant_personality = sample_personality()
            
            # randomly sample a user profile and a assistant profile
            simulated_user_profile = sample_profile(profile_slots)
            simulated_assistant_profile = sample_profile(profile_slots, exclude_name=simulated_user_profile["Name"])
            
            # print (Fore.GREEN + f"Simulated User Profile: {simulated_user_profile}" + Style.RESET_ALL)
            # print (Fore.GREEN + f"Simulated Assistant Profile: {simulated_assistant_profile}" + Style.RESET_ALL)
            
            small_image_path = os.path.join(small_image_cache_dir, os.path.basename(scene_image_paths[0]).replace(".png", "_small.jpg"))
            compress_image(scene_image_paths[0], small_image_path, target_size_kb=100)
            second_small_image_path = os.path.join(small_image_cache_dir, os.path.basename(second_scene_images_paths[0]).replace(".png", "_small.jpg"))
            compress_image(second_scene_images_paths[0], second_small_image_path, target_size_kb=100)
            
                    
            env_desc, user_dict, assistant_dict, moderator_dict, objects_info_in_scene, simulate_user_preference, second_objects_info, second_simulate_preference = create_instruct(
                scene_image_path=small_image_path,
                scene_image_info_path=scene_image_info_paths[0],
                metadata=metadata,
                domain=domain,
                user_profile=simulated_user_profile,
                assistant_profile=simulated_assistant_profile,
                user_personality=simulated_user_personality,
                assistant_personality=simulated_assistant_personality,
                conversation_end_or_continue_sample=conversation_end_or_continue_sample,
                max_interaction_step=max_interaction_step,
                second_scene_image_path=second_small_image_path,
                second_scene_image_info_path=second_scene_images_info_paths[0],
                different_type_name=different_type_name
            )
            # print (Fore.GREEN + f"Environment Description: {env_desc}" + Style.RESET_ALL)
            # print (Fore.GREEN + f"User Dict: {user_dict}" + Style.RESET_ALL)
            # print (Fore.GREEN + f"Assistant Dict: {assistant_dict}" + Style.RESET_ALL)
            # print (Fore.GREEN + f"Moderator Dict: {moderator_dict}" + Style.RESET_ALL)
            
            transition_turn = random.choice(range(min_transition_step, max_transition_step))
            
            # print (Fore.GREEN + f"Transition Turn: {transition_turn}" + Style.RESET_ALL)
            # print (Fore.GREEN + f"Different Type Name: {different_type_name}" + Style.RESET_ALL)
            
            # print (Fore.BLUE + f"User Dict: {user_dict}" + Style.RESET_ALL)
            # print (Fore.RED + f"Assistant Dict: {assistant_dict}" + Style.RESET_ALL)
            
            assistant = Player(
                name=assistant_dict["name"], backend=OpenAIChat(model=model_name, temperature=temperature, max_tokens=max_system_tokens), role_desc=assistant_dict["role_desc"], role_desc_in_transition_turn=assistant_dict["role_desc_in_transition_turn"], role_desc_after_transition_turn=assistant_dict["role_desc_after_transition_turn"], transition_turn=transition_turn, global_prompt=env_desc, visual_path=small_image_path, second_visual_path=second_small_image_path)
            user = Player(
                name=user_dict["name"], backend=OpenAIChat(model=model_name, temperature=temperature, max_tokens=max_user_tokens), role_desc=user_dict["role_desc"], role_desc_in_transition_turn=user_dict["role_desc_in_transition_turn"], role_desc_after_transition_turn=user_dict["role_desc_after_transition_turn"], transition_turn=transition_turn, global_prompt=env_desc, visual_path=small_image_path, second_visual_path=second_small_image_path)
            moderator = Moderator(
                backend=OpenAIChat(model=model_name, temperature=temperature, max_tokens=max_moderator_tokens), 
                role_desc=moderator_dict["role_desc"], 
                role_desc_in_transition_turn=moderator_dict["role_desc"], 
                role_desc_after_transition_turn=moderator_dict["role_desc"], 
                transition_turn=transition_turn,
                terminal_condition=moderator_dict["terminal_condition"])
            # let assistant start the conversation
            env = ModeratedConversation(player_names=[p.name for p in [assistant, user]], moderator=moderator, moderator_period="round")
            arena = Arena(players=[assistant, user], environment=env, global_prompt=env_desc)
            
            arena.launch_cli(max_steps=max_interaction_step, show_description=show_description, show_message=show_message, interactive=False)

            print("Save? (y/n)")
            if input() == "n":
               continue
            
            # save the simulated dialog to file
            messages = env.get_observation()
            simulated_convs = []
            for msg in messages:
                if msg.agent_name == assistant.name:
                    utt = {"system": msg.content}
                else:
                    utt = {"user": msg.content}
                simulated_convs.append(utt)
            
            write_line = {
                "id": time.strftime("%Y%m%d%H%M%S", time.localtime()),
                "scene_image_path": scene_image_paths[0],
                "scene_image_info_path": scene_image_info_paths[0],
                "second_scene_image_path": second_scene_images_paths[0],
                "second_scene_image_info_path": second_scene_images_info_paths[0],
                "transition_turn": transition_turn,
                "domain": domain,
                "seed_dialog": dialog_example,
                "user_profile": simulated_user_profile,
                "assistant_profile": simulated_assistant_profile,
                "user_personality": simulated_user_personality,
                "objects_info_in_scene": objects_info_in_scene,
                "simulate_user_preference": simulate_user_preference,
                "second_objects_info": second_objects_info,
                "second_simulate_preference": second_simulate_preference,
                "conversation": simulated_convs
            }
            fw.write(json.dumps(write_line, ensure_ascii=False, indent=4) + "\n")
            fw.flush()

        #     print("Sleeping for 5 seconds...")
        #     time.sleep(5)
        #     exit()

            #print("Continue? (y/n)")
            #if input() == "n":
            #    break


if __name__ == '__main__':
    init(autoreset=True)
    args = parse_args()
    random.seed(args.random_seed)
    generate_dialog_data(
        args.train_scenes_images_pool_path,
        args.test_scenes_images_pool_path,
        args.train_scenes_images_info_pool_path,
        args.test_scenes_images_info_pool_path,
        args.fashion_metadata_path,
        args.furniture_metadata_path,
        args.user_profiles_path,
        args.seed_data_path,
        max_generated_dialogs=args.max_generated_dialogs,
        max_interaction_step=args.max_interaction_step,
        min_transition_step=args.min_transition_step,
        max_transition_step=args.max_transition_step,
        max_system_tokens=args.max_system_tokens,
        max_user_tokens=args.max_user_tokens,
        max_moderator_tokens=args.max_moderator_tokens,
        model_name=args.model_name,
        temperature=args.temperature,
        output_dir=args.output_dir,
        show_description=args.show_description,
        show_message=args.show_message,
        small_image_cache_dir=args.small_img_cache_dir
    )
