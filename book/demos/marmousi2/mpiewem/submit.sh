#!/bin/sh
for i in {1..5}
do
   qsub -l nodes=1:ppn=12,mem=88gb,walltime=4:00:00 02_fdmod.pbs
done
