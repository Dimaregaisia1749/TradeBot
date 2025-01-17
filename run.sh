#!/bin/bash

start() {
    python -m app.main
}

display_stats() {
    python tools/display_stats.py
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
    download)
        download_data
        ;;
    delete_data)
        delete_data
        ;;
    *)
        echo "Params: $0 {start|display_stats|download|delete_data}"
        ;;
esac

read -p "Press enter to continue"