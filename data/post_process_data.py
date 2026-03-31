"""
This script is for post-processing the dataset for tobacco samples.

One item in a particular image sample ideally is given as:

{
    'item': str
    'description': str
    'flavors': str
    'marketing': str
    'shape': str
    'color': str
    'text': str
}
"""
import json
from json_repair import repair_json
import os
import pandas as pd
from tqdm import tqdm

attributes = ['item', 'description', 'flavors', 'marketing', 'shape', 'color', 'text']
attributes_to_probe = ['item', 'description', 'marketing', 'text']

product_type_keywords = { # TODO: fine synonyms and associated slang for each prod type
    "nicotine_pouch": ["nicotine pouch", "pouch"], # NEW, TODO: I think we should add energy and coffee pouches, as supplementary samples
    "smokeless_tobacco": ["snuff", "smokeless", "dip"],
    "pipe_tobacco": ["pipe"],
    "cigars": ["cigar"],
    "hookah": ["hookah"],
    "patches": ["patch"],
    "gum": ["gum"],
    "lozenges": ["lozenge"],
    "heated_tobacco": ["heated tobacco"],
    "cigarettes": ["cigarette"],
    "e-cigarettes": ["e-cigarette", "vape"],
    "uncategorized": [] # TODO: do later
}

known_product_names = { # removed names that may be mistaken for other things--animals, objects, etc.
    "smokeless_tobacco": ["copenhagen", "goat", "husky", "klint", "longhorn", "rogue", "stokers", "velo", "zyn", "creek", "grizzly", "kayak", "kodiak", "red seal", "skoal", "timber wolf", "vild", 'nykd', 'xqs', 'on!', 'baow', 'lynx', '77', 'volt', 'maverick', 'vid', 'fumi', 'lyft', 'smokey mountain', 'dynamite', 'kavak', "taylor's pride", "wakey"], # NOTE: -> {smokeless tobacco, nicotine pouches}
    "pipe_tobacco": ['peterson', 'cornell and diehl', 'john cotton', 'hearth & home'],
    "cigars": ['davidoff', 'perdomo', 'ashton', 'arturo fuente', 'la aroma de cuba', 'romeo y julieta', 'oliva', 'gold & mild', 'black & mild'],
    "hookah": ['trifecta', 'nakhla', 'khalil mamoon', 'kaloud', 'al fakher', 'starbuzz', 'tangiers'],
    "patches": ['nicotex', 'habitrol', 'puzeku', 'leader', 'walgreens', 'apotex', 'fekux', 'sorelax', 'ximonth', 'nicotouch', 'nicoderm', 'nicassist', 'valleylux', 'kroger', 'cvs', 'aroamas', 'rugby', 'sunmark', 'nicotinell', 'nicotrol', 'blue point', 'nicorette', 'nicabate', 'quitx', 'rite aid', 'deboob', 'sparsha', 'major pharmaceuticals', 'equate', 'telanshare', 'up&up', 'niquitin', 'bluepoint'],
    "gum": ['meijer', 'rugby', 'nicorelief', 'members mark', 'goodsense', 'amazon', 'blip', 'kroger', 'sainsburys', 'equate', 'zonnic', 'lucy', 'rite aid', 'nicotinell', 'leader', 'cvs', 'up&up', 'nicotrol', 'nicassist', 'sunmark'],
    "lozenges": ['kirkland signature', 'wellness basics', 'signature care', 'publix', 'amazon', 'walgreens', 'nixit', 'jones', 'rubicon', 'foster & thrive', 'nicosure', 'rising health', 'heb', 'kroger', 'zonnic', 'members mark', 'goodsense', 'cvs', 'up&up', 'nicotex', 'blip', 'sunmark', 'rogue', 'niquitin', 'meijer', 'rite aid', 'nicassist', 'rugby'],
    "heated_tobacco": ['iqos', 'ismod', 'ploom', 'pulze', 'pax', 'terea', 'evo', 'glo', 'nexone', 'buddz', 'lil', 'hitaste', 'heets', 'neostik'],
    "cigarettes": ['premier', 'eclipse', 'pall mall', 'marlboro', 'l & m', 'winston', 'american spirit', 'newport', 'camel', 'chesterfield', 'rothmans', 'sampoerna', 'padron'],
    "e-cigarettes": ['vuse solo', 'flum', 'geek bar', 'mi-pod', 'lost mary', 'tyson 20', 'smok', 'uwell', 'raz', 'vuse', 'nex'],
    "uncategorized": [] # TODO: fill in later
}

server_root = "/media/ttdat/Data2TB/manuel/tobacco/tobacco_1m_2026" # contains 'dataset''smokey mountain'

computers = {"lambda", "cviu", "home", "lab"}
computer="lambda"
if computer == "home":
    data_root = "/home/serna/Programming/smart_connect_health_neurips_2026/data" # contains 'dataset'
elif computer == "lab":
    data_root = "/home/mserna/Programming/smart_connect_health_neurips_2026/data"
elif computer == "lambda":
    data_root = "/home/mserna/projects/tobacco-projects/smart_connect_health_neurips_2026/data"
else:
    raise ValueError("Unknown computer")

simple_labels_path = os.path.join(data_root, "simple_image_labels.csv")
captions_path = os.path.join(data_root, "debug_results/result_qwen3vl_captions")
out_labels_path = os.path.join(data_root, "image_labels.csv")


def check_caption_raw_out():
    """ Check the captioning raw output in the JSON files

    :return:
    """
    files = os.listdir(captions_path)

    df = pd.read_csv(simple_labels_path, index_col=0)

    n_samples = 0
    bad_reads = {}
    n_bad = 0

    for filename in tqdm(files):
        data_path = os.path.join(captions_path, filename)
        cdf = pd.read_json(data_path)
        cdf.filepath = cdf.filepath.str.replace(server_root, data_root)
        cdf.caption = cdf.caption.apply(lambda x: repair_json(x))

        bad_infer = cdf[cdf['caption'].str.len() == 2] # empty lists:, e.g., []
        bad_reads[filename] = len(bad_infer)
        n_bad += len(bad_infer)

        n_samples += len(cdf.caption)

    print(f"{n_samples} captions were processed.")
    print(f"{len(df)} samples total")
    print()
    print(f"{n_bad} bad samples")
    for k,v in bad_reads.items():
        print(f"{k}: {v}")

    print()


def get_item_labels_one_sample(data:dict):
    """ Get refined item labels for one sample.

    :param data: (dict) caption and attribute generation output for one sample
    :return:
    """
    # TODO: follow the below steps to get refined labels
    #  DONE--1) fill in the names for all other product types
    #  2) check for different product types in the attributes (already have at this pt)
    #  3) then check for names under that product type (may have to update)
    #  4) if no match, then return '', and then I have to manually check again

    # (i) see if product type key words, not present in other types, are present first, helping us narrow down possible brands
    for pt_name, keyword_list in product_type_keywords.items():
        for keyword in keyword_list:
            # for current keyword, check if it is in attributes
            for attribute in attributes:
                if keyword in data[attribute]:
                    import pdb;pdb.set_trace()

    # (ii) if no product type match found, find based on brand name
    # TODO

    #import pdb;pdb.set_trace()

    # --------------------------------------
    # NOTE: OLD code below, the below code that returns product type and brand name will be replaced
    '''
    if data['simple_product_type'] == 'smokeless_tobacco':
        # TODO: we have to separate into 'smokeless_tobacco' and 'nicotine_pouches', and change simple labels
        pass
    else:
        #product_type = data['simple_product_type']
        pass
    # elif data["simple_product_type"] == 'uncategorized':
    #     pass
    # else:
    #     raise ValueError("Unknown product type")

    # ...get the list of known products for the given simple product type label
    pt:list = known_product_names[data['simple_product_type']]

    # ...assign product type and **real** product name
    for attribute in attributes:
        for candidate_name in pt:
            if isinstance(data[attribute], str):
                if candidate_name.lower() in data[attribute].lower(): #.contains(, case=False):
                    # if candidate_name.lower() == 'zyn':
                    #     import pdb;pdb.set_trace()

                    return "smokeless_tobacco", candidate_name
            else:
                # assume list[str] in else case
                for tmp in data[attribute]:
                    if candidate_name.lower() in tmp.lower(): #.contains(, case=False):
                        return "smokeless_tobacco", candidate_name

    # TODO: use simple product name or product type label to find case=False contains match
    #  else, loop through the other products names and keywords,
    #import pdb;pdb.set_trace()
    '''

    return "", ""


def post_process_one_product_type(captions_df, labels_df) -> list:
    """ Post-process data, e.g., for a sub-dataframe of captions for a particular product type

    :param captions_df: (pd.DataFrame) dataframe with generated captions and labels for a product type
    :param labels_df: (pd.DataFrame) dataframe with corresponding simple, naive labels
    :return:
    """
    assert len(captions_df) == len(labels_df)

    out_data = []

    # Go over rows of captions data
    for j in tqdm(range(len(captions_df))):
        caption_row = captions_df.iloc[j]
        labels_row = labels_df[labels_df["uid"] == caption_row["uid"]] # get corresponding simple labels row by uid

        # Check
        assert labels_row.filepath.item() in caption_row.filepath # check file paths match as an added sanity layer
        try:
            caption_row_atts: list[dict] = json.loads(caption_row.caption) # load dict encoded as string to list[dict]
        except json.decoder.JSONDecodeError:
            continue # this should remove empty string samples, very few and irrelevant samples will be tossed (handful)

        # The caption for current image is a list, and may include multiple detected items
        for object_idx, row_att in enumerate(caption_row_atts):
            if not isinstance(row_att, dict):
                continue # this tosses out any objects (row_att) that are not dicts that the json repair lib did repair, but is not what we want

            # For current item...
            # ...add simple labels from corresponding labels_row
            uid = labels_row.uid.item()
            filepath = labels_row.filepath.item()
            simple_tobacco_type = labels_row.tobacco_type.item()
            simple_product_type = labels_row.product_type.item()
            simple_product_name = labels_row.product_name.item()

            current_data = {
                "uid": uid,
                "object_id": object_idx,
                "filepath": filepath,
                "simple_tobacco_type": simple_tobacco_type,
                "simple_product_type": simple_product_type,
                "simple_product_name": simple_product_name,
            }

            # ...add caption info
            for att in attributes:
                # First add empty attribute if it is not present in the attribute dict
                if att not in row_att.keys():
                    current_data[att] = ""
                    continue

                if isinstance(row_att[att], list):
                    current_data[att] = ', '.join(row_att[att])
                else:
                    current_data[att] = row_att[att]

            # ...add refined labels (based caption and simple labels)
            product_type, product_name = get_item_labels_one_sample(current_data)
            current_data["product_type"] = product_type
            current_data["product_name"] = product_name

            out_data.append(current_data)

    # ------------------------------------------------------
    # import pdb;pdb.set_trace()

    # DEBUG: assuming no data leakage across the original product_type
    curr_df = pd.DataFrame(out_data)
    print(curr_df)
    raise Exception() # fix each label one at a time, then come back here

    options = ''
    current_brands = ["copenhagen", "klint", "longhorn", "rogue", "stokers", "velo", "zyn", "red seal", "skoal", "timber wolf", "vild", 'nykd', 'xqs', 'on!', 'baow', 'lynx', '77', 'volt', 'maverick', 'vid', 'fumi', 'lyft', 'smokey mountain', 'dynamite', 'kavak', "taylor's pride", "wakey"] # kavak=>kayak
    for name in current_brands:
        options += f'{name}|'
    options = options + 'smokeless|tobacco|smokeless tobacco|nicotine|pouch|nicotine pouch'
    item_df = curr_df[curr_df.item.str.contains(options, case=False)]
    print(len(item_df))
    desc_df = curr_df[curr_df.description.str.contains(options, case=False)]
    print(len(desc_df))
    market_df = curr_df[curr_df.marketing.str.contains(options, case=False)]
    print(len(market_df))
    text_df = curr_df[curr_df.text.str.contains(options, case=False, na=False)]
    print(len(text_df))

    res_df = pd.concat([desc_df, item_df, market_df, text_df])
    res_df = res_df.drop_duplicates()
    res_df = res_df.reset_index(drop=True)
    res_df.to_csv(os.path.join(data_root, "sample_labels.csv"))

    # res_df[res_df['product_name'] == ''] # should be empty!!!

    """# DEBUG: get as many accurate labels for product names in curr_df as possible, and confirm new ones
    found_pt_names = curr_df.product_name.unique().tolist()

    for fpn in found_pt_names:
        print(f"{fpn} ({len(curr_df[curr_df['product_name'] == fpn])})")

    neg_df = curr_df[curr_df['product_name'] == '']
    remain_df = neg_df[neg_df.item.str.contains('smokeless|tobacco|smokeless tobacco|nicotine|pouch|nicotine pouch', case=False)]

    pos_df = curr_df[curr_df['product_name'] != '']

    print(f'...unmatched results: {len(neg_df)}')
    print(f'...samples still to be processed by keywords in unmatched: {len(remain_df)}')
    print(f'...ready: {len(pos_df)}')"""

    import pdb;pdb.set_trace()
    # ------------------------------------------------------

    return out_data



def post_process_all_data(replace_img_path:bool=False, out_filepath:str=None):
    """ Post-process the raw captioning generation output.
    The result is *one* large CSV file with all the columns we will ever need in any experiment.

    :param replace_img_path: (bool) Replace the original image path for all samples in the df
    :param out_filepath: (str) Output file path
    :return: None
    """
    #cols = ['uid', 'filepath', 'tobacco_type', 'product_type', 'product_name']
    simple_df = pd.read_csv(simple_labels_path, index_col=0)
    files = os.listdir(captions_path)

    # DEBUG: smokeless tobacco first
    files = ["res_qwen3vl_nodist_smokeless_tobacco.json"]

    all_out_data = [] # will form output df
    #sample_attributes = {at:{} for at in attributes} # for each attribute, each dict element is uid:(str)

    for filename in files:
        data_path = os.path.join(captions_path, filename)
        cdf = pd.read_json(data_path)

        print(f"Processing file: {filename}; samples={len(cdf)}")

        if replace_img_path:
            cdf.filepath = cdf.filepath.str.replace(server_root, data_root)
        cdf.caption = cdf.caption.apply(lambda x: repair_json(x))

        pt_df = simple_df[simple_df["uid"].isin(set(cdf.uid.unique()))]

        # Process all samples for the current file/product type
        out_data = post_process_one_product_type(captions_df=cdf, labels_df=pt_df)
        all_out_data += out_data

    # Format all post-processed data
    df = pd.DataFrame(all_out_data)
    import pdb;pdb.set_trace()

    # TODO: we should assign new labels based on the captions
    # out_path = ""
    # df.to_csv(out_path,



if __name__ == '__main__':
    #check_caption_raw_out()

    post_process_all_data(replace_img_path=True) # for my local pc
    #post_process_data(False) # for server
