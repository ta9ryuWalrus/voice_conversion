#!/bin/bash

if [ ! -e conf_mgc ];then
	mkdir conf_mgc
fi

ls data/male/data/mgc/ | head -500 | sed -e 's/\.dat//' > conf_mgc/train.list
ls data/male/data/mgc/ | tail -20 | sed -e 's/\.dat//' > conf_mgc/eval.list
