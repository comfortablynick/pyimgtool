bin_name := 'pyimg.py'

alias r := run
alias build := testrun # in case vim calls it
alias install := testrun # in case vim calls it

# Run with optional args
run +args='':
	python {{bin_name}}.py {{args}}

# test args
testrun:
    ./{{bin_name}} test/sunset.jpg test/sunset_edited.jpg -vv -mw 2000
