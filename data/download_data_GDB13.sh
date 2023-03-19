#!/bin/bash
wget http://gdbtools.unibe.ch:8080/cdn/gdb13.tgz
tar xvzfO gdb13.tgz --exclude 13.smi | gzip > gdb-12.smi.gz
tar xvzfO gdb13.tgz | gzip > gdb-13.smi.gz


wget https://github.com/gablg1/ORGAN/raw/master/organ/NP_score.pkl.gz
wget https://github.com/gablg1/ORGAN/raw/master/organ/SA_score.pkl.gz
