bin_name := 'pyimg'

alias r := run
alias b := build
alias i := install

# Run with optional args
run +args='':
	python {{bin_name}}.py {{args}}

# Define this in case vim calls it
build:
    python {{bin_name}}.py

# Define this in case vim calls it
install:
    python {{bin_name}}.py
