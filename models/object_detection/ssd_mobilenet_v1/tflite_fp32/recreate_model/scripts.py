import collections

def read_label_map(label_map_path):

    item_id = None
    item_name = None
    items = {}
    
    with open(label_map_path, "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "display_name" in line:
                item_name = line.split(":", 1)[1].replace("'", "").strip()

            if item_id is not None and item_name is not None:
                items[item_id] = item_name
                item_id = None
                item_name = None

    return items

def convert_dictionary_to_list(d):
    output_list = []
    # order dictionary by keys
    d = collections.OrderedDict(sorted(d.items()))
    for k, v in d.items():
        output_list.append(v)

    return output_list
    

if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    items = read_label_map(filename)
    items = convert_dictionary_to_list(items)
    with open("temp.txt", "w") as f:
        for item in items:
            f.write("%s\n" % item)
