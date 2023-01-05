#!/usr/bin/env bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# switch to website directory
pushd "$parent_path/../../website"

set -xe

npm run cypress:run:contract

popd
