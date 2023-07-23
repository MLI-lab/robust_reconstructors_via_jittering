import configparser
import os

def add_settings_info_to_df(df, run_dir_column="run_dir", d_set_mapping=None, t_set_mapping=None):

    if d_set_mapping is not None:
        for value in d_set_mapping.values():
            df[value] = 0

    if t_set_mapping is not None:
        for value in t_set_mapping.values():
            df[value] = 0

    for index, row in df.iterrows():
        run_dir = row[run_dir_column]
        config_path = os.path.join(run_dir, "settings")
        config = configparser.ConfigParser()
        config.read(config_path)
        try:
            d_set = config["Data"]
            t_set = config["Training"]

            if d_set_mapping is not None:
                for key, value in d_set_mapping.items():
                    df.at[index, value] = d_set[key]

            if t_set_mapping is not None:
                for key, value in t_set_mapping.items():
                    df.at[index, value] = t_set[key]
        except:
            print(f"Exception on run_dir: {run_dir}")