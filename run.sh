#!/bin/bash
export PYTHONPATH=$(pwd)

start() {
    python app/main.py
}

display_stats() {
    python tools/display_stats.py
}

sandbox() {
    python app/sandbox/sandbox.py
}

download_data() {
    tools/download_market_data.sh
}

delete_data() {
    rm -r ./market_data/unzip_data
    rm -r ./market_data/zip_data
}

case "$1" in
    start)
        start
        ;;
    display_stats)
        display_stats
        ;;
    sandbox)
        sandbox
        ;;
    download)
        download_data
        ;;
    delete_data)
        delete_data
        ;;
    *)
        echo "Params: $0 {start|display_stats|sandbox|download|delete_data}"
        ;;
esac

read -p "Press enter to continue"