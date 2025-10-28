#!/bin/bash

## Check if xtb is installed
#if ! command -v xtb &> /dev/null
#then
#    echo "xtb could not be found. Please install xtb to use this script."
#    exit
#fi

### generate molden.input
# Loop through all molecule_*.xyz files in the current directory
for file in DPI_xyz/*
do
    echo "file is: $file"
    for mol_xyz in $file/*_ligand*.xyz
    do
       echo "file is: $mol_xyz"
       # Extract the base name without extension
       echo "$mol_xyz"
       base_name=$(basename "$mol_xyz" .xyz)
       #echo "$base_name"
       # Create a directory for the output
       mkdir -p "molden/$file/$base_name"

       # Run xtb and direct output to the corresponding directory
       xtb "$mol_xyz" --gfn 2 --molden --parallel 5  > molden/$file/$base_name/output.log 2>&1

       # move generated files to "molden/$file"
       mv "molden.input" "molden/$file/$base_name"
       mv "charges" "molden/$file/$base_name"
       mv "wbo" "molden/$file/$base_name"
       mv "xtbrestart" "molden/$file/$base_name"
       mv "xtbtopo.mol" "molden/$file/$base_name"
    done
done
echo "All files processed (generate molden.input)."





