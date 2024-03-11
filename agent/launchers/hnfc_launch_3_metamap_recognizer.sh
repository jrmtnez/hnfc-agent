#!/bin/bash

SECONDS=0

echo "Launching Metamap recognizer..."

cd /root/vsc/hnfc-dataset-agent/

if  ls ./data/metamap_sent/*.xml 1>/dev/null 2>&1;
then
    rm ./data/metamap_sent/*.xml;
fi

if  ls ./data/metamap_sent/*.json 1>/dev/null 2>&1;
then
    rm ./data/metamap_sent/*.json;
fi

if  ls ./data/metamap_sent/*.txt 1>/dev/null 2>&1;
then
    rm ./data/metamap_sent/*.txt;
fi

source /root/miniconda3/bin/activate agent
python -m agent.launchers.hnfc_launch_3_export_sentences_to_metamap
conda deactivate

if  ls ./data/metamap_sent/*.txt 1>/dev/null 2>&1;
then
    echo "Sentences exported to MetaMap"

    echo "Starting MetaMap services..."

    /root/public_mm/bin/skrmedpostctl start
    /root/public_mm/bin/wsdserverctl start

    sleep 3m

    for file in ./data/metamap_sent/*.txt; do
        output_file=${file%.txt}
        /root/public_mm/bin/metamap -y --XMLf $file $output_file.xml
        rm $file
    done


    /root/public_mm/bin/skrmedpostctl stop
    /root/public_mm/bin/wsdserverctl stop

    echo "Retrieving data from MetaMap"

    source /root/miniconda3/bin/activate agent
    python -m agent.launchers.hnfc_launch_3_import_data_from_metamap
    conda deactivate

    echo "MetaMap process finished"
else
    echo "Nothing to process by Metamap!"
fi

duration=$SECONDS

echo "Total time: $(($duration / 60)) minutes and $(($duration % 60)) seconds."
