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

servers=( "$@" )
result_path=workspace-data/results

echo "Syncing results from servers: ${servers[*]}"

for s in "${servers[@]}"; do
	local_server_path=~/${result_path}/${s}
	mkdir -p "${local_server_path}"
	rsync -avz "${s}:~/${result_path}/" "${local_server_path}/" --delete --progress --human-readable \
	  --exclude "bin" --exclude "*.key" --exclude "*.pem" --delete-excluded
done
