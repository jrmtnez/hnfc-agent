#!/bin/bash

dir=.
parentdir="$(dirname "$dir")"

perl wikifil.pl ../text_data/enwiki-20170301-pages-articles1.xml-p000000010p000030302 > ../text_data/text8.txt
truncate -s 250000000 ../text_data/text8.txt
