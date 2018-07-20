# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""DataSet utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib


def file_md5_check(file, expected):
    md5 = hashlib.md5()
    with open(file, 'rb') as file_obj:
        while True:
            chunk = file_obj.read(4096)

            if len(chunk) == 0:
                break

            md5.update(chunk)

    actual = md5.hexdigest()
    return expected == actual
