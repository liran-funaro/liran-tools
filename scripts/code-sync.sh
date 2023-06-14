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

servers=()
additional_args=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--hard)
      additional_args=("--delete-excluded")
      shift
      ;;
    *)
      servers+=("$1")
      shift
      ;;
  esac
done

rel_path=$(perl -e 'use File::Spec; print File::Spec->abs2rel(@ARGV) . "\n"' . ~)

echo "Syncing ${rel_path} to ${servers[*]}"

for s in "${servers[@]}"; do
	rsync -avz \
	  --rsync-path="mkdir -p ~/${rel_path} & rsync" \
	  --exclude ".git" --exclude "*.ipynb" --exclude "*.pdf" --exclude ".DS_Store" --exclude ".idea" \
	  --delete "${additional_args[@]}" --filter=':- .gitignore' \
	  ./ "${s}:~/${rel_path}/"
done
