#!/bin/bash

if [ ! -e conf_f0 ];then
	mkdir conf_f0
fi

ls data/male/data/f0/ | head -1 | sed -e 's/\.dat//' > conf_f0/train.list
ls data/male/data/f0/ | tail -1 | sed -e 's/\.dat//' > conf_f0/eval.list
