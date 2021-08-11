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
import ast

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ImageNet labels.")
    parser.add_argument("--path", type=str, required=True)

    args = parser.parse_args()

    with open(args.path, "r") as f:
        data = f.read()

    labels = ast.literal_eval(data)
    
    # Include the background class as there are 1001 classes
    class_labels = ["background"]

    for _, l in labels.items():
        class_labels.append(l)

    with open("labelmappings.txt", "w") as f:
        for l in class_labels:
            f.write("{}\n".format(l))
