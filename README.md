## Nand2Tetris Course

Compilers and partial OS implementation for the Nand2Tetris course (https://www.nand2tetris.org/)

### Setup

Writen in python 3.9. To install, simply run `pip3 install -r requirements.txt`

### Usage

#### Assembler

Translates HACK assembly files into binary HACK code. Files must be `.asm` files.

Usage: `python3 compilers\assembler.py filename`

#### VM Translator

Translates HACK virtual machine language into HACK assembly. If used on a directory, will translate all `.vm` files. 

Usage: `python3 compilers\vm_tranlator.py filepath`

#### Jack compiler

Compiles high level JACK code into VM language or HACK assembly. If used on a directory, will translate all `.jack` files.

Usage: `python3 compilers\jack_compiler.py filepath mode` where mode is `vm` to create VM file or `asm` to create HACK assembly file