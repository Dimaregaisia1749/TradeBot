#!/bin/bash
export PYTHONPATH=$(pwd)

start() {
    python app/main.py
}

view_stats() {
    python tools/view_stats.py
}

reset_sandbox() {
    python app/sandbox/reset_sandbox.py
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
    view_stats)
        view_stats
        ;;
    reset_sandbox)
        reset_sandbox
        ;;
    download)
        download_data
        ;;
    delete_data)
        delete_data
        ;;
    *)
        echo "Params: $0 {start|view_stats|sandbox|download|delete_data}"
        ;;
esac

read -p "Press enter to continue"