#!/bin/sh
# postCreateCommand.sh

echo "START Install"

sudo chown -R vscode:vscode .

poetry config virtualenvs.in-project true
poetry install

echo "SETUP Vespa"

VESPA_CLI_VERSION=8.453.24
curl -L -o .bin/vespa-cli.tar.gz https://github.com/vespa-engine/vespa/releases/download/v${VESPA_CLI_VERSION}/vespa-cli_${VESPA_CLI_VERSION}_linux_amd64.tar.gz
tar -xzvf .bin/vespa-cli.tar.gz -C .bin
mv .bin/vespa-cli_${VESPA_CLI_VERSION}_linux_amd64/bin/vespa .bin/vespa
rm .bin/vespa-cli.tar.gz
rm -rf .bin/vespa-cli_${VESPA_CLI_VERSION}_linux_amd64

vespa config set target http://vespa:19071
vespa status
vespa deploy vespa-config --wait 30

echo "FINISH"