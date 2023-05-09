#!/bin/bash

# Author: Liran Funaro <liran.funaro@gmail.com>
#
# Copyright (C) 2006-2023 Liran Funaro
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

set -e

me=$(realpath "$0")

bin_path="$HOME/bin"
if [ ! -d "$bin_path" ]; then
  bin_path="$HOME/Library/bin"
fi
if [ ! -d "$bin_path" ]; then
  exit 1
fi


for script in scripts/*; do
  script_name=$(basename "$script")
  script_path=$(realpath "$script")
  script_bin_path="$bin_path/$script_name"

  if [ "$script_path" == "$me" ]; then
    continue
  fi

  if [ -f "$script_bin_path" ] && [ ! -L "$script_bin_path" ]; then
    echo "$script_name exists but it is not a link."
    continue
  fi

  rm -f "$script_bin_path"
  ln -s "$script_path" "$script_bin_path"
done
