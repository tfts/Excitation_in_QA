ANALYSIS_PERIOD_DICT = {  # Assuming data is given in hours
    "y": 365 * 24,  # years
    "m": 30 * 24,  # months
    "d": 24  # days
}
BOOTSTRAP_REPETITIONS = 100
DIMENSION_NAMES = ["_posts_freq", "_posts_nfreq", "_replies_freq", "_replies_nfreq"]
FREQUENT_PERCENTILE = 90
MIN_NUMBER_OF_EVENTS = 10
MODE_TO_DATASETS = {
    "GROW_VS_DEC": {  # datasets of at least 3 years
        # Datasets in 20th percentile of total activity growth in percentage (this comes from "02_zeilis_02_application.R" and end of the file "declining_datasets" and "growing_datasets" vectors)
        "declining": ["stackapps", "webapps", "sound", "parenting", "ham", "cooking", "sustainability", "pets", "spanish", "tridion", "boardgames", "productivity", "pm", "skeptics", "expressionengine", "ebooks", "genealogy", "craftcms", "bricks", "cstheory", "fitness", "gardening"],
        # Datasets in 80th percentile of total activity growth in percentage
        "growing": ["tex", "ru", "electronics", "wordpress", "dba", "codereview", "puzzling", "blender", "salesforce", "sharepoint", "crypto", "askubuntu", "gis", "stats", "security", "academia", "opendata", "ux", "codegolf", "money", "chemistry", "unix"]
    }, "STEM_VS_HUMAN": {
        "humanities": ["arabic", "buddhism", "chinese", "christianity", "english", "esperanto", "french", "german", "music", "mythology", "philosophy", "portuguese", "russian", "spanish", "writers"],
        "stem": ["physics", "stats", "astronomy", "biology", "chemistry", "cs", "earthscience", "engineering", "sound", "space", "electronics", "datascience", "cogsci", "reverseengineering", "softwareengineering"],
    }
}
NUMBER_OF_PROCESSES = 2
NUMBER_OF_DIMENSIONS = 4
PATH_TO_DESTINATION_DATASET = "../data/se_users_monthly_rows/"
PATH_TO_SOURCE_DATASETS = "../data/raw/"
