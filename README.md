# rush-generation

## Installation

Create a conda environment named `rush-generation` and install dependencies from `requirements.txt`:

```bash
conda env remove -y -n rush-generation
conda create -n rush-generation cudatoolkit python=3.11 -y
conda activate rush-generation
pip install -r requirements.txt
```

## Run

Activate the conda environment and run the main script:

```bash
conda activate rush-generation
python main.py
```

this model simulates tile drain processes
it has 4 storages: static , surface , top layer that produces interflow, bottom layer that produces tile flow and baseflow

the toplayer is 15cm depth. can store the porosity x 15cm, forced as input
the bottom layer goes from 15cm to bedrock. if bedrock <15cm bottomlayer is zero.
the tile drain depth is a parameter
the bottom layer capacity is the average porosity of 15cm to 200cm x bedrock depth.
the average porosity of 15 to 200cm is an input.
if toplayer is full, water does not infiltrate. stays in the surface
if bottomlayer is full, water does not percolate, stays in the toplayer


static has max_static
surface has no max


https://asciiflow.com/#/share/eJyrVspLzE1VslLKzU9JzYkvSVHSUcpJrEwtAgpVxyiVpRYVZ%2BbnxShZGenEKFUAaUtLYyCrEiRiaQBklaRWlAA5MUqPpvQ8mtKAiibExOQ9mtKkAAJAGgtvCoaeGWBxapmlVKtUCwCod2nv)

         ┌─────┐                
         │     │                
         │     │static          
         └─────┘                
         ┌─────┐                
         │     │                
   sflow │     │surface         
     ◄───┼─────┘                
         ┌─────┐                
         │     │                
interflow|     │top layer       
     ◄───┬─────┘                
         ┌─────┐                
         │     │                
   TDflow│     │bottom above TD
     ◄───┼─────┘                
         ┌─────┐                
         │     │                
 baseflow│     │bottom below TD
     ◄───┼─────┘                


