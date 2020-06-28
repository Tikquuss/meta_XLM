import argparse 

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Training")
    
    parser.add_argument("--config_file", type=str, default="", help="")
    return parser

def main(params) :
    import json

    with open(params.config_file) as json_data:
        data_dict = json.load(json_data)
        for key, value in data_dict.items():
            setattr(params, key, value)
            
if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    main(params)