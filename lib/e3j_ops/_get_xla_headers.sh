#! /usr/bin/bash
# Copyright (c) 2026 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Download External API headers from XLA/ffi/api/

HEADERS=("api.h" "c_api.h" "c_api_internal.h" "ffi.h")
DEST="xla/ffi/api/"
XLA_RAW="https://raw.githubusercontent.com/openxla/xla/refs/heads/main/"
SOURCE="$XLA_RAW/xla/ffi/api"

for header in ${HEADERS[@]}; do
    echo "get xla/ffi/api/$header ..."
    curl -L "$SOURCE/$header" > "$DEST/$header"
done
