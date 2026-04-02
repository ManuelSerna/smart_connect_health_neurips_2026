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
import shutil
from tqdm import tqdm

attributes = ['item', 'description', 'flavors', 'marketing', 'shape', 'color', 'text']
attributes_to_probe = ['item', 'description', 'marketing', 'text']

product_type_keywords = {
    "nicotine_pouch": ["nicotine pouch", "pouch", "caffeine", "energy", "pouches", "slim", "smooth", "strong", "medium", "strength", "x-strong", "nicopod"], # NEW, TODO: I think we should add energy and coffee pouches, as supplementary samples
    "smokeless_tobacco": ["snuff", "smokeless", "dip", "chew", "snus"],
    "pipe_tobacco": ["pipe"],
    "cigars": ["cigar"],
    "hookah": ["hookah", "shisha"],
    "patches": ["patch"],
    "gum": ["gum"],
    "lozenges": ["lozenge"],
    "heated_tobacco": ["heated tobacco"],
    "cigarettes": ["cigarette"],
    "e-cigarettes": ["e-cigarette", "vape", "e-liquid", "e-juice"],
    "uncategorized": ["tobacco", "nicotine", "smoking", "cancer", "mg"] # NOTE: this is a class, but we insert general keywords here
}

known_product_names = { # removed names that may be mistaken for other things--animals, objects, etc.
    "smokeless_tobacco": ["copenhagen", "goat", "husky", "klint", "longhorn", "rogue", "stokers", "velo", "zyn", "creek", "grizzly", "kayak", "kodiak", "red seal", "skoal", "timber wolf", "vild", 'nykd', 'xqs', 'on!', 'baow', 'lynx', '77', 'volt', 'maverick', 'vid', 'fumi', 'lyft', 'smokey mountain', 'dynamite', 'kavak', "taylor's pride", "wakey"], # NOTE: -> {smokeless tobacco, nicotine pouches}
    "pipe_tobacco": ['peterson', 'cornell and diehl', 'john cotton', 'hearth & home'],
    "cigars": ['davidoff', 'perdomo', 'ashton', 'arturo fuente', 'la aroma de cuba', 'romeo y julieta', 'oliva', 'gold & mild', 'black & mild', 'arturo', 'fuente', 'romeo', 'julieta', 'mild'],
    "hookah": ['trifecta', 'nakhla', 'khalil mamoon', 'kaloud', 'al fakher', 'starbuzz', 'tangiers', 'khalil', 'fakher'],
    "patches": ['nicotex', 'habitrol', 'puzeku', 'leader', 'walgreens', 'apotex', 'fekux', 'sorelax', 'ximonth', 'nicotouch', 'nicoderm', 'nicassist', 'valleylux', 'kroger', 'cvs', 'aroamas', 'rugby', 'sunmark', 'nicotinell', 'nicotrol', 'blue point', 'nicorette', 'nicabate', 'quitx', 'rite aid', 'deboob', 'sparsha', 'major pharmaceuticals', 'equate', 'telanshare', 'up&up', 'niquitin', 'bluepoint', 'pharmaceuticals'],
    "gum": ['meijer', 'rugby', 'nicorelief', 'members mark', 'goodsense', 'amazon', 'blip', 'kroger', 'sainsburys', 'equate', 'zonnic', 'lucy', 'rite aid', 'nicotinell', 'leader', 'cvs', 'up&up', 'nicotrol', 'nicassist', 'sunmark', 'rite', 'member'],
    "lozenges": ['kirkland signature', 'wellness basics', 'signature care', 'publix', 'amazon', 'walgreens', 'nixit', 'jones', 'rubicon', 'foster & thrive', 'nicosure', 'rising health', 'heb', 'kroger', 'zonnic', 'members mark', 'goodsense', 'cvs', 'up&up', 'nicotex', 'blip', 'sunmark', 'rogue', 'niquitin', 'meijer', 'rite aid', 'nicassist', 'rugby', 'kirkland', 'signature', 'foster'],
    "heated_tobacco": ['iqos', 'ismod', 'ploom', 'pulze', 'pax', 'terea', 'evo', 'glo', 'nexone', 'buddz', 'lil', 'hitaste', 'heets', 'neostik'],
    "cigarettes": ['premier', 'eclipse', 'pall mall', 'marlboro', 'l & m', 'winston', 'american spirit', 'newport', 'camel', 'chesterfield', 'rothmans', 'sampoerna', 'padron'],
    "e-cigarettes": ['vuse solo', 'flum', 'geek bar', 'geek', 'mi-pod', 'mi pod', 'lost mary', 'mary', 'tyson', 'smok', 'uwell', 'raz', 'vuse', 'nex', 'juul', 'elf', 'puff'],
    "uncategorized": [] # TODO: fill in later
}

server_root = "/media/ttdat/Data2TB/manuel/tobacco/tobacco_1m_2026" # contains 'dataset''smokey mountain'

computers = {"lambda", "cviu", "home", "lab"}
computer="lambda"
if computer == "home":
    data_root = "/home/serna/Programming/smart_connect_health_neurips_2026/data" # contains 'dataset'
    dataset_root = "dataset"
elif computer == "lab":
    data_root = "/home/mserna/Programming/smart_connect_health_neurips_2026/data"
    dataset_root = "dataset"
elif computer == "lambda":
    data_root = "/home/mserna/projects/tobacco-projects/smart_connect_health_neurips_2026/data"
    dataset_root = "/media/ttdat/Data2TB/manuel/tobacco/tobacco_1m_2026/dataset"
else:
    raise ValueError("Unknown computer")

simple_labels_path = os.path.join(data_root, "simple_image_labels.csv")
captions_path = os.path.join(data_root, "debug_results/result_qwen3vl_captions")



def post_process_one_product_type(captions_df, labels_df) -> tuple:
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
            # TODO: do later after manual refinement
            # product_type, product_name = get_item_labels_one_sample(current_data)
            # current_data["product_type"] = product_type
            # current_data["product_name"] = product_name
            out_data.append(current_data)

    # Create negative and positive sets
    # product_type_keywords: dict[list[str]]
    # known_product_names: dict[list[str]]
    options = ""

    # ...add keys
    for pt in product_type_keywords.keys():
        options += f"{pt.replace('_', ' ')}|"
    for pn in known_product_names.keys():
        options += f"{pn.replace('_', ' ')}|"

    # add vals in list[str]
    for keyword_list in product_type_keywords.values():
        for keyword in keyword_list:
            options += f"{keyword}|"
    for names_list in known_product_names.values():
        for name in names_list:
            options += f"{name}|"

    options = options[:-1]

    curr_df = pd.DataFrame(out_data) # get all samples together into a df before splitting into neg and pos sets

    res_item_df = curr_df[curr_df.item.str.contains(options, case=False)]
    print(len(res_item_df))
    res_desc_df = curr_df[curr_df.description.str.contains(options, case=False)]
    print(len(res_desc_df))
    res_market_df = curr_df[curr_df.marketing.str.contains(options, case=False)]
    print(len(res_market_df))
    res_text_df = curr_df[curr_df.text.str.contains(options, case=False, na=False)]
    print(len(res_text_df))

    pos_df = pd.concat([res_item_df, res_item_df, res_market_df, res_text_df])
    pos_df = pos_df.drop_duplicates()
    pos_df = pos_df.reset_index(drop=True)
    pos_uids = pos_df.uid.unique()

    neg_df = labels_df[~labels_df['uid'].isin(pos_uids)]

    return neg_df, pos_df


def create_pos_neg_datasets():
    if os.path.isfile("simple_image_labels_negative.csv") and os.path.isfile("simple_image_labels_negative.csv"):
        neg_df = pd.read_csv(f"simple_image_labels_negative.csv")
        pos_df = pd.read_csv(f"image_labels_positive.csv")

        print(f"Read positive samples df: {len(pos_df)}")
        print(f"Read negative samples df: {len(neg_df)}")
    else:
        # Create negative and positive dataset folders
        subfolders = []
        for root, dirs, files in os.walk(dataset_root):
            for d in dirs:
                subfolders.append(os.path.join(root, d))

        for subfolder in subfolders:
            new_path = subfolder.replace("dataset", "positive_dataset")
            os.makedirs(new_path, exist_ok=True)
            new_path = subfolder.replace("dataset", "negative_dataset")
            os.makedirs(new_path, exist_ok=True)

        # Process files
        simple_df = pd.read_csv(simple_labels_path, index_col=0)
        files = os.listdir(captions_path)

        negative_data = []
        positive_data = []

        for filename in files:
            data_path = os.path.join(captions_path, filename)
            cdf = pd.read_json(data_path)

            print(f"Processing file: {filename}; samples={len(cdf)}")

            if computer == "lambda":
                cdf.filepath = cdf.filepath.str.replace("dataset", server_root + "/dataset")
            else:
                cdf.filepath = cdf.filepath.str.replace(server_root, data_root)

            cdf.caption = cdf.caption.apply(lambda x: repair_json(x))

            pt_df = simple_df[simple_df["uid"].isin(set(cdf.uid.unique()))]

            # Process all samples for the current file/product type
            current_negative, current_positive = post_process_one_product_type(captions_df=cdf, labels_df=pt_df)
            negative_data.append(current_negative)
            positive_data.append(current_positive)

        # Write positive and negative sample info to file
        neg_df = pd.concat(negative_data)
        pos_df = pd.concat(positive_data)

        print(f"Positive samples (objects): {len(pos_df)}; images: {len(pos_df.uid.unique())}")
        print(f"Negative samples: {len(neg_df)}")

        neg_df.to_csv(f"simple_image_labels_negative.csv", index=False)
        pos_df.to_csv(f"image_labels_positive.csv", index=False)

    # Copy files from dataset to positive or negative versions for manual review
    print("Copying positive samples to new subset")
    for pf in tqdm(pos_df.filepath):
        old_filepath = os.path.join(server_root, pf)
        new_filepath = os.path.join(server_root, pf.replace("dataset", "positive_dataset"))
        shutil.copy(old_filepath, new_filepath)

    print("Copying negative samples to new subset")
    for pf in tqdm(neg_df.filepath):
        old_filepath = os.path.join(server_root, pf)
        new_filepath = os.path.join(server_root, pf.replace("dataset", "negative_dataset"))
        shutil.copy(old_filepath, new_filepath)

    neg_df.filepath = neg_df.filepath.str.replace("dataset", "negative_dataset")
    pos_df.filepath = pos_df.filepath.str.replace("dataset", "positive_dataset")
    neg_df.to_csv(f"image_labels_negative_subset.csv", index=False) # filepaths point to new folder
    pos_df.to_csv(f"image_labels_positive_subset.csv", index=False) # filepaths point to new folder

    print("Done.")


def check_second_neg_pos_datasets():
    second_pass_dir = "sift_results"
    template = os.path.join(second_pass_dir, "res_qwen3vl_nodist_{}.json")

    pos_dfs = []
    neg_dfs = []

    for pt in known_product_names.keys():
        tmp = pd.read_json(template.format(pt))
        pos_dfs.append(tmp[tmp['caption'] == 'yes'])
        neg_dfs.append(tmp[tmp['caption'] == 'no'])

        print(f"... [{pt}] {len(tmp[tmp['caption'] == 'yes'])} positive samples")
        print(f"... [{pt}] {len(tmp[tmp['caption'] == 'no'])}  negative samples")

    pos_df = pd.concat(pos_dfs)
    neg_df = pd.concat(neg_dfs)

    print(f"positive: {len(pos_df)}")
    print(f"negative: {len(neg_df)}")

    # TODO: move samples of second positive and second negative sets to their own folders, but not in remote machine, do it locally

    import pdb;pdb.set_trace()



if __name__ == '__main__':
    ''' Our processing pipeline:
    1. Use large scale scraping from APIfy to get raw set of samples from publicly-available sources
    2. Use APIfy's tools to do a first-round screening of samples, so we can remove obvious bad samples
    3. With the remaining image-label samples, use a VLM like Qwen to caption on the set of our defined attributes.
        This will help automate differentiating images that are not tobacco or nicotine products but may be hard to filter due to text query-image similarity from the scraping procedure.
    4. Semi-automatic cleanup
        4a. To separate the negative (irrelevant) samples from the positive (relevant) samples, we initially sort them, using the VLM captioning of multiple attributes, into negative and positive versions of the dataset.
            - DONE
        4b. Manually search for images in the negative dataset that should be moved to the positive dataset.
            For NEGATIVE samples, remove any image files that we know should be moved to the **POSITIVE** dataset. (We care more about this case.)
            For POSITIVE samples, remove any image files that we know should be moved to the **NEGATIVE** dataset.
            - IN PROGRESS
        4c. Manually search for images in the positive dataset that should be moved to the negative dataset.
            - IN PROGRESS
    5. Re-organize labels based on cleanup in step 4 using the positive set of samples
        - TODO
    
    Progress removing negative samples from the below positive samples:
    al_fakher
    amazon
    american_spirit: done
    apotex
    aroamas
    arturo_fuente
    ashton
    black&mild
    blip
    blue_point
    bluepoint
    buddz
    camel
    chesterfield
    copenhagen
    cornell_and_diehl
    creek
    cvs
    davidoff
    deboob
    eclipse
    equate
    evo
    fekux
    flum
    foster&thrive
    geek_bar
    glo
    goat
    gold&mild
    goodsense
    grizzly
    habitrol
    hearth&home
    heb
    heets
    hitaste
    husky
    iqos
    ismod
    john_cotton
    jones
    kaloud
    kayak
    khalil_mamoon
    kirkland_signature
    klint
    kodiak
    kroger
    l&m
    la_aroma_de_cuba
    leader
    lil
    longhorn
    lost_mary
    lucy
    major_pharmaceuticals
    marlboro
    meijer
    members_mark
    mi-pod
    nakhla
    neostik
    newport
    nex
    nexone
    nicabate
    nicassist
    nicoderm
    nicorelief
    nicorette
    nicosure
    nicotex
    nicotinell
    nicotouch
    nicotrol
    niquitin
    nixit
    oliva
    padron
    pall_mall
    pax
    perdomo
    peterson
    ploom
    premier
    publix
    pulze
    puzeku
    quitx
    raz
    red_seal
    rising_health
    rite_aid
    rogue
    romeo_y_julieta
    rothmans
    rubicon
    rugby
    sainsburys
    sampoerna
    signature_care
    skoal
    smok
    sorelax
    sparsha
    starbuzz
    stokers
    sunmark
    tangiers
    telanshare
    terea
    timber_wolf
    trifecta
    tyson_20
    unknown
    up&up
    uwell
    valleylux
    velo
    vild
    vuse
    vuse_solo
    walgreens
    wellness_basics
    winston
    ximonth
    zonnic
    zyn
    '''

    # Step 4a
    #create_pos_neg_datasets() # NOTE: this will create basically another copy of the dataset, so mind the storage requirements

    # Step 4b
    check_second_neg_pos_datasets()
