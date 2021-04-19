# Copyright (C) 2021 Arm Limited or its affiliates. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import collections
import sys

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
    parser = argparse.ArgumentParser(description="Process ImageNet labels.")
    parser.add_argument("--path", type=str, required=True)

    args = parser.parse_args()

    items = read_label_map(args.path)
    items = convert_dictionary_to_list(items)

    with open("temp.txt", "w") as f:
        for item in items:
            f.write("%s\n" % item)
