#!/bin/bash


# The density.cub is saved in the same folder as molden.input

# Define the base directory
base_dir="molden"

# Change to the base directory
cd "$base_dir"

# Loop through all molecule_ directories in the base directory
for dir in */
do
    # Change to the current molecule directory
    cd "$dir"
    for subdir in */
    do
      cd "$subdir"
      for subsubdir in */
      do
        cd "$subsubdir"
        # Check if molden.input file exists
        if [ -f "molden.input" ]; then

            # Run the Multifn command
            echo -e "5\n1\n3\n2\n0\nq" | Multiwfn "molden.input" > "density_log.txt"

            echo "Running $command in $subsubdir"
            eval $command
        else
            echo "molden.input not found in $subsubdir"
        fi
      done
    done
done


